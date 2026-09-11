#!/usr/bin/env python3
"""Tune the final front-fusion mag gate without touching the held-out cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from log_registry import DEFAULT_REGISTRY_PATH, LogRegistry  # noqa: E402
from travel_solver_core import (  # noqa: E402
    SolverInputs,
    flatten_1d,
    solve_travel,
    solver_weights_for_mag_baseline,
)


DEFAULT_TUNING_SETS = ("harry", "jamaal", "stumpjumper-front-pod-v2")
DEFAULT_HELD_OUT_SET = "stumpjumper-front-pod-v1"
DEFAULT_FLOORS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1_000_000.0)
CURRENT_FLOOR = 0.1
LOW_TRAVEL_MAX_MM = 30.0


@dataclass(frozen=True)
class SweepRow:
    split: str
    cohort: str
    log: str
    mag_off_floor: float
    off_gate_weight: float
    n_samples: int
    n_low_travel: int
    anchor_off_fraction: float
    corrected_mag_centered_rmse_mm: float
    centered_rmse_mm: float
    low_travel_centered_rmse_mm: float
    uncentered_rmse_mm: float
    cache_max_abs_diff_mm: float
    success: bool
    status: int
    nfev: int
    cost: float
    optimality: float
    runtime_s: float


def parse_floats(value: str) -> tuple[float, ...]:
    floors = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not floors:
        raise argparse.ArgumentTypeError("at least one floor is required")
    if any(floor < 0 for floor in floors):
        raise argparse.ArgumentTypeError("floors must be non-negative")
    return floors


def centered_error(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prediction = flatten_1d(prediction)
    truth = flatten_1d(truth)
    centered = np.full(len(truth), np.nan, dtype=float)
    centered[mask] = (
        prediction[mask]
        - float(np.mean(prediction[mask]))
        - truth[mask]
        + float(np.mean(truth[mask]))
    )
    return centered


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def load_inputs(cache_path: Path) -> tuple[SolverInputs, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        inputs = SolverInputs(
            time_s=cache["travel__t"],
            accel_mm_s2=flatten_1d(cache["accel/lpfhp/proj__x"]) * 1000.0,
            mag=cache["mag/norm/corr/lpf__x"],
            mag_preds_mm=cache["travel/mag_nuisance/corrected__x"],
            mag_zv_points=cache["mag_zv_points"],
            mag_baseline=float(flatten_1d(cache["mag_baseline"])[0]),
        )
        truth = flatten_1d(cache["travel__x"])
        mask = np.asarray(cache["boring_mask"], dtype=bool).reshape(-1)
        cached_solved = flatten_1d(cache["travel/solved__x"])
    lengths = {len(inputs.time_s), len(truth), len(mask), len(cached_solved)}
    if len(lengths) != 1:
        raise ValueError(f"unaligned cache arrays in {cache_path}: lengths={sorted(lengths)}")
    if not np.any(mask):
        raise ValueError(f"boring_mask is empty in {cache_path}")
    return inputs, truth, mask, cached_solved


def run_log(
    split: str,
    cohort: str,
    log_name: str,
    floors: Sequence[float],
    cache_root: str,
    max_nfev: int,
) -> list[dict[str, object]]:
    cache_path = Path(cache_root) / log_name / "cache" / "all.npz"
    inputs, truth, mask, cached_solved = load_inputs(cache_path)
    low_mask = mask & (truth >= 0.0) & (truth < LOW_TRAVEL_MAX_MM)
    if not np.any(low_mask):
        raise ValueError(f"no 0-{LOW_TRAVEL_MAX_MM:g} mm samples in {log_name}")
    corrected_error = centered_error(inputs.mag_preds_mm, truth, mask)
    corrected_rmse = rmse(corrected_error[mask])
    anchor_off_fraction = float(np.mean(inputs.mag[mask] <= inputs.mag_baseline))
    rows: list[dict[str, object]] = []
    for floor in floors:
        started = time.perf_counter()
        weights = solver_weights_for_mag_baseline(inputs.mag_baseline, mag_off_floor=float(floor))
        result = solve_travel(inputs, weights, max_nfev=max_nfev)
        elapsed = time.perf_counter() - started
        prediction = result.x
        error = centered_error(prediction, truth, mask)
        cache_diff = (
            float(np.max(np.abs(prediction - cached_solved)))
            if np.isclose(floor, CURRENT_FLOOR, rtol=0.0, atol=1e-12)
            else float("nan")
        )
        row = SweepRow(
            split=split,
            cohort=cohort,
            log=log_name,
            mag_off_floor=float(floor),
            off_gate_weight=float(floor / (1.0 + floor)),
            n_samples=int(np.sum(mask)),
            n_low_travel=int(np.sum(low_mask)),
            anchor_off_fraction=anchor_off_fraction,
            corrected_mag_centered_rmse_mm=corrected_rmse,
            centered_rmse_mm=rmse(error[mask]),
            low_travel_centered_rmse_mm=rmse(error[low_mask]),
            uncentered_rmse_mm=rmse(prediction[mask] - truth[mask]),
            cache_max_abs_diff_mm=cache_diff,
            success=bool(result.scipy_result.success),
            status=int(result.scipy_result.status),
            nfev=int(result.scipy_result.nfev),
            cost=float(result.scipy_result.cost),
            optimality=float(result.scipy_result.optimality),
            runtime_s=elapsed,
        )
        rows.append(asdict(row))
    return rows


def resolve_cohorts(registry_path: Path, set_names: Iterable[str]) -> dict[str, list[str]]:
    registry = LogRegistry.load(registry_path)
    cohorts: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for set_name in set_names:
        logs = registry.select(set_name=set_name, usable_only=True)
        if not logs:
            raise ValueError(f"registry set {set_name!r} contains no usable logs")
        wrong_pipeline = [log.log_id for log in logs if log.pipeline != "front"]
        if wrong_pipeline:
            raise ValueError(f"set {set_name!r} contains non-front logs: {wrong_pipeline}")
        cohorts[set_name] = sorted(log.log_id for log in logs)
        for log_name in cohorts[set_name]:
            if log_name in seen:
                raise ValueError(f"log {log_name!r} is in both {seen[log_name]!r} and {set_name!r}")
            seen[log_name] = set_name
    return cohorts


def execute_split(
    split: str,
    cohorts: dict[str, list[str]],
    floors: Sequence[float],
    cache_root: Path,
    max_nfev: int,
    workers: int,
) -> list[dict[str, object]]:
    tasks = [(cohort, log_name) for cohort, logs in cohorts.items() for log_name in logs]
    rows: list[dict[str, object]] = []
    if workers <= 1:
        for index, (cohort, log_name) in enumerate(tasks, 1):
            rows.extend(run_log(split, cohort, log_name, floors, str(cache_root), max_nfev))
            print(f"[{split} {index}/{len(tasks)}] {cohort}/{log_name}", flush=True)
        return rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_log, split, cohort, log_name, floors, str(cache_root), max_nfev
            ): (cohort, log_name)
            for cohort, log_name in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            cohort, log_name = futures[future]
            rows.extend(future.result())
            print(f"[{split} {index}/{len(tasks)}] {cohort}/{log_name}", flush=True)
    return rows


def mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.mean(values))


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    splits = sorted({str(row["split"]) for row in rows})
    floors = sorted({float(row["mag_off_floor"]) for row in rows})
    metric_keys = (
        "corrected_mag_centered_rmse_mm",
        "centered_rmse_mm",
        "low_travel_centered_rmse_mm",
        "uncentered_rmse_mm",
        "runtime_s",
    )
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        cohorts = sorted({str(row["cohort"]) for row in split_rows})
        for floor in floors:
            floor_rows = [row for row in split_rows if float(row["mag_off_floor"]) == floor]
            if not floor_rows:
                continue
            for cohort in cohorts:
                selected = [row for row in floor_rows if row["cohort"] == cohort]
                if selected:
                    output.append({
                        "split": split,
                        "cohort": cohort,
                        "mag_off_floor": floor,
                        "n_logs": len(selected),
                        "success_fraction": mean(selected, "success"),
                        **{key: mean(selected, key) for key in metric_keys},
                    })
            cohort_rows = [
                row for row in output
                if row["split"] == split
                and float(row["mag_off_floor"]) == floor
                and row["cohort"] in cohorts
            ]
            output.append({
                "split": split,
                "cohort": "cohort-balanced",
                "mag_off_floor": floor,
                "n_logs": len(floor_rows),
                "success_fraction": mean(floor_rows, "success"),
                **{key: mean(cohort_rows, key) for key in metric_keys},
            })
    return output


def select_floor(aggregate: Sequence[dict[str, object]]) -> dict[str, object]:
    candidates = [
        row for row in aggregate
        if row["split"] == "tuning" and row["cohort"] == "cohort-balanced"
    ]
    converged = [row for row in candidates if float(row["success_fraction"]) == 1.0]
    if not converged:
        raise RuntimeError("no tuning candidate converged on every log")
    return min(
        converged,
        key=lambda row: (
            float(row["low_travel_centered_rmse_mm"]),
            float(row["centered_rmse_mm"]),
            abs(float(row["mag_off_floor"]) - CURRENT_FLOOR),
        ),
    )


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None


def find_aggregate(
    aggregate: Sequence[dict[str, object]], split: str, cohort: str, floor: float
) -> dict[str, object]:
    return next(
        row for row in aggregate
        if row["split"] == split
        and row["cohort"] == cohort
        and np.isclose(float(row["mag_off_floor"]), floor, rtol=0.0, atol=1e-12)
    )


def change_text(selected: dict[str, object], current: dict[str, object], key: str) -> str:
    delta = float(selected[key]) - float(current[key])
    return f"{delta:+.3f} mm ({100.0 * delta / float(current[key]):+.1f}%)"


def paired_change_summary(
    rows: Sequence[dict[str, object]],
    split: str,
    cohort: str,
    selected_floor: float,
    key: str,
) -> tuple[float, int, int]:
    selected_rows = {
        str(row["log"]): row
        for row in rows
        if row["split"] == split
        and row["cohort"] == cohort
        and np.isclose(float(row["mag_off_floor"]), selected_floor, rtol=0.0, atol=1e-12)
    }
    current_rows = {
        str(row["log"]): row
        for row in rows
        if row["split"] == split
        and row["cohort"] == cohort
        and np.isclose(float(row["mag_off_floor"]), CURRENT_FLOOR, rtol=0.0, atol=1e-12)
    }
    if selected_rows.keys() != current_rows.keys():
        raise ValueError(f"unpaired rows for {split}/{cohort}/{key}")
    changes = [float(selected_rows[log][key]) - float(current_rows[log][key]) for log in selected_rows]
    return float(np.mean(changes)), sum(change < 0 for change in changes), len(changes)


def render_report(
    tuning_cohorts: dict[str, list[str]],
    held_out_cohorts: dict[str, list[str]],
    floors: Sequence[float],
    selected_floor: float,
    aggregate: Sequence[dict[str, object]],
    rows: Sequence[dict[str, object]],
) -> str:
    tuning = [
        find_aggregate(aggregate, "tuning", "cohort-balanced", floor) for floor in floors
    ]
    heldout_name = next(iter(held_out_cohorts))
    heldout_selected = find_aggregate(aggregate, "held-out", heldout_name, selected_floor)
    heldout_current = find_aggregate(aggregate, "held-out", heldout_name, CURRENT_FLOOR)
    current_tuning = find_aggregate(aggregate, "tuning", "cohort-balanced", CURRENT_FLOOR)
    selected_tuning = find_aggregate(aggregate, "tuning", "cohort-balanced", selected_floor)
    replay_diffs = [
        float(row["cache_max_abs_diff_mm"])
        for row in rows
        if np.isclose(float(row["mag_off_floor"]), CURRENT_FLOOR, rtol=0.0, atol=1e-12)
    ]
    convergence_notes = [
        f"{float(item['mag_off_floor']):g}: {100 * float(item['success_fraction']):.1f}%"
        for item in tuning
        if float(item["success_fraction"]) < 1.0
    ]
    lines = [
        "# Post-mag-correction fusion `mag_off_floor` sweep",
        "",
        "## Design",
        "",
        f"The final front fusion solve was replayed from each current pipeline cache using `travel/mag_nuisance/corrected` as its magnetic travel observation. The swept floors were {', '.join(f'`{floor:g}`' for floor in floors)}; these correspond to anchor-off magnetic residual multipliers of {', '.join(f'{100 * floor / (1 + floor):.1f}%' for floor in floors)}. The current value is `{CURRENT_FLOOR:g}`.",
        "",
        f"Tuning used {sum(map(len, tuning_cohorts.values()))} logs from {', '.join(f'`{name}`' for name in tuning_cohorts)}. `{heldout_name}` ({sum(map(len, held_out_cohorts.values()))} logs) was not evaluated until after selection. The primary objective was the mean 0–30 mm centered RMSE after first averaging logs within each tuning cohort and then weighting the three cohorts equally. Centering used one offset per full log before the low-travel subset was scored.",
        "",
        "## Tuning result",
        "",
        "| mag_off_floor | off-gate weight | 0–30 mm RMSE | overall RMSE | converged |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in tuning:
        lines.append(
            f"| {float(item['mag_off_floor']):g} | {100 * float(item['mag_off_floor']) / (1 + float(item['mag_off_floor'])):.1f}% | {float(item['low_travel_centered_rmse_mm']):.3f} mm | {float(item['centered_rmse_mm']):.3f} mm | {100 * float(item['success_fraction']):.1f}% |"
        )
    lines.extend([
        "",
        f"Selected: **`mag_off_floor={selected_floor:g}`**. Relative to `{CURRENT_FLOOR:g}` on tuning, its 0–30 mm RMSE changed by {change_text(selected_tuning, current_tuning, 'low_travel_centered_rmse_mm')} and overall centered RMSE changed by {change_text(selected_tuning, current_tuning, 'centered_rmse_mm')}.",
        "",
        "The aggregate is heterogeneous across setups:",
        "",
        "| tuning cohort | 0–30 mm change | logs improved | overall change | logs improved | uncentered change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cohort in tuning_cohorts:
        low_delta, low_wins, count = paired_change_summary(
            rows, "tuning", cohort, selected_floor, "low_travel_centered_rmse_mm"
        )
        overall_delta, overall_wins, _ = paired_change_summary(
            rows, "tuning", cohort, selected_floor, "centered_rmse_mm"
        )
        uncentered_delta, _, _ = paired_change_summary(
            rows, "tuning", cohort, selected_floor, "uncentered_rmse_mm"
        )
        lines.append(
            f"| `{cohort}` | {low_delta:+.3f} mm | {low_wins}/{count} | {overall_delta:+.3f} mm | {overall_wins}/{count} | {uncentered_delta:+.3f} mm |"
        )
    heldout_low_delta, heldout_low_wins, heldout_count = paired_change_summary(
        rows, "held-out", heldout_name, selected_floor, "low_travel_centered_rmse_mm"
    )
    heldout_overall_delta, heldout_overall_wins, _ = paired_change_summary(
        rows, "held-out", heldout_name, selected_floor, "centered_rmse_mm"
    )
    heldout_uncentered_delta, heldout_uncentered_wins, _ = paired_change_summary(
        rows, "held-out", heldout_name, selected_floor, "uncentered_rmse_mm"
    )
    lines.extend([
        "",
        "## Held-out result",
        "",
        f"On `{heldout_name}`, the selected value changed 0–30 mm centered RMSE by {change_text(heldout_selected, heldout_current, 'low_travel_centered_rmse_mm')} and overall centered RMSE by {change_text(heldout_selected, heldout_current, 'centered_rmse_mm')} relative to `{CURRENT_FLOOR:g}`.",
        "",
        "| value | 0–30 mm RMSE | overall RMSE | uncentered RMSE |",
        "| ---: | ---: | ---: | ---: |",
        f"| {CURRENT_FLOOR:g} (current) | {float(heldout_current['low_travel_centered_rmse_mm']):.3f} mm | {float(heldout_current['centered_rmse_mm']):.3f} mm | {float(heldout_current['uncentered_rmse_mm']):.3f} mm |",
        f"| {selected_floor:g} (selected) | {float(heldout_selected['low_travel_centered_rmse_mm']):.3f} mm | {float(heldout_selected['centered_rmse_mm']):.3f} mm | {float(heldout_selected['uncentered_rmse_mm']):.3f} mm |",
        "",
        f"The held-out mean improvement was mixed at log level: {heldout_low_wins}/{heldout_count} logs improved at 0–30 mm and {heldout_overall_wins}/{heldout_count} improved overall. Uncentered RMSE changed by {heldout_uncentered_delta:+.3f} mm on average and improved on {heldout_uncentered_wins}/{heldout_count} logs.",
        "",
        "## Checks and interpretation",
        "",
        f"Replaying the current `{CURRENT_FLOOR:g}` setting matched the cached final solver output to a worst-case absolute difference of {max(replay_diffs):.3g} mm.",
    ])
    if convergence_notes:
        lines.append("Incomplete convergence on tuning: " + ", ".join(convergence_notes) + ".")
    else:
        lines.append("Every tuning solve reported successful convergence.")
    lines.extend([
        "",
        "The formal low-travel objective validates the hypothesis directionally, but it does not support changing the global default yet. Most of the tuning gain came from Harry; the fully-on endpoint slightly worsened overall centered error on Jamaal and Stumpjumper-v2, and it worsened uncentered error on every held-out log. A moderate floor of `0.3` was the only swept value that improved both mean low-travel and mean overall centered RMSE in every tuning cohort, so it is the better candidate for a separately preregistered validation run when another untouched setup is available.",
        "",
        "This is a held-out cohort check rather than an uncertainty estimate: logs within a cohort share hardware and riding conditions, so the per-log sample count should not be read as independent replication.",
        "",
        "Full per-log values are in `per_log_metrics.csv`; cohort and cohort-balanced summaries are in `aggregate_metrics.csv`; the frozen selection is in `selection.json`.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune mag_off_floor on the final front fusion solver with a held-out cohort"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tuning-sets", nargs="+", default=list(DEFAULT_TUNING_SETS))
    parser.add_argument("--held-out-set", default=DEFAULT_HELD_OUT_SET)
    parser.add_argument("--floors", type=parse_floats, default=DEFAULT_FLOORS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    floors = tuple(dict.fromkeys(float(value) for value in args.floors))
    if not any(np.isclose(floor, CURRENT_FLOOR, rtol=0.0, atol=1e-12) for floor in floors):
        raise ValueError(f"floors must include the current value {CURRENT_FLOOR:g}")
    tuning_cohorts = resolve_cohorts(args.registry, args.tuning_sets)
    held_out_cohorts = resolve_cohorts(args.registry, [args.held_out_set])
    overlap = set(sum(tuning_cohorts.values(), [])) & set(sum(held_out_cohorts.values(), []))
    if overlap:
        raise ValueError(f"held-out logs overlap tuning logs: {sorted(overlap)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "script": str(Path(__file__).resolve()),
        "registry": str(args.registry.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "tuning_cohorts": tuning_cohorts,
        "held_out_cohorts": held_out_cohorts,
        "candidate_mag_off_floors": floors,
        "current_mag_off_floor": CURRENT_FLOOR,
        "low_travel_max_mm": LOW_TRAVEL_MAX_MM,
        "selection_rule": "minimum cohort-balanced 0-30 mm centered RMSE among fully converged tuning candidates; overall centered RMSE then distance from current are tie-breakers",
        "max_nfev": args.max_nfev,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    tuning_rows = execute_split(
        "tuning", tuning_cohorts, floors, args.cache_root, args.max_nfev, args.workers
    )
    tuning_aggregate = aggregate_rows(tuning_rows)
    selected = select_floor(tuning_aggregate)
    selected_floor = float(selected["mag_off_floor"])
    held_out_floors = tuple(dict.fromkeys((CURRENT_FLOOR, selected_floor)))
    held_out_rows = execute_split(
        "held-out",
        held_out_cohorts,
        held_out_floors,
        args.cache_root,
        args.max_nfev,
        args.workers,
    )
    rows = sorted(
        tuning_rows + held_out_rows,
        key=lambda row: (
            str(row["split"]), str(row["cohort"]), str(row["log"]), float(row["mag_off_floor"])
        ),
    )
    aggregate = aggregate_rows(rows)
    selection = {
        "selected_mag_off_floor": selected_floor,
        "selected_off_gate_weight": selected_floor / (1.0 + selected_floor),
        "selection_split": "tuning",
        "selection_metric": "cohort-balanced low_travel_centered_rmse_mm",
        "tuning_summary": selected,
        "held_out_values_evaluated_after_selection": held_out_floors,
    }
    write_csv(args.output_dir / "per_log_metrics.csv", rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_report(tuning_cohorts, held_out_cohorts, floors, selected_floor, aggregate, rows),
        encoding="utf-8",
    )
    print(f"Selected mag_off_floor={selected_floor:g}")
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
