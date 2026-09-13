#!/usr/bin/env python3
"""Tune the final front-fusion magnetic cutoff after nuisance correction."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.front.mag_offset_calibration.sweep_post_mag_correction_fusion import (  # noqa: E402
    CURRENT_FLOOR,
    DEFAULT_HELD_OUT_SET,
    DEFAULT_TUNING_SETS,
    LOW_TRAVEL_MAX_MM,
    centered_error,
    git_revision,
    load_inputs,
    mean,
    resolve_cohorts,
    rmse,
    write_csv,
)
from backend.log_registry import DEFAULT_REGISTRY_PATH  # noqa: E402
from backend.travel_solver_core import solve_travel, solver_weights_for_mag_baseline  # noqa: E402


CURRENT_MODE = "baseline"
DEFAULT_THRESHOLD_MODES = ("baseline", "1250", "1000", "750", "500", "disabled")


@dataclass(frozen=True)
class ThresholdRow:
    split: str
    cohort: str
    log: str
    threshold_mode: str
    resolved_mag_x_thresh_mg: float
    mag_off_floor: float
    full_weight_fraction: float
    n_samples: int
    n_low_travel: int
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


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip().casefold() for part in value.split(",") if part.strip())
    for mode in modes:
        if mode in {"baseline", "disabled"}:
            continue
        try:
            threshold = float(mode)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid threshold mode {mode!r}") from exc
        if threshold < 0:
            raise argparse.ArgumentTypeError("thresholds must be non-negative")
    if not modes:
        raise argparse.ArgumentTypeError("at least one threshold mode is required")
    return modes


def resolve_threshold(mode: str, mag_baseline: float) -> float | None:
    if mode == "baseline":
        return float(mag_baseline)
    if mode == "disabled":
        return None
    return float(mode)


def run_log(
    split: str,
    cohort: str,
    log_name: str,
    modes: Sequence[str],
    cache_root: str,
    max_nfev: int,
) -> list[dict[str, object]]:
    inputs, truth, mask, cached_solved = load_inputs(
        Path(cache_root) / log_name / "cache" / "all.npz"
    )
    low_mask = mask & (truth >= 0.0) & (truth < LOW_TRAVEL_MAX_MM)
    if not np.any(low_mask):
        raise ValueError(f"no 0-{LOW_TRAVEL_MAX_MM:g} mm samples in {log_name}")
    corrected_error = centered_error(inputs.mag_preds_mm, truth, mask)
    rows: list[dict[str, object]] = []
    for mode in modes:
        threshold = resolve_threshold(mode, float(inputs.mag_baseline))
        weights = replace(
            solver_weights_for_mag_baseline(inputs.mag_baseline),
            mag_x_thresh=threshold,
            mag_off_floor=CURRENT_FLOOR,
        )
        started = time.perf_counter()
        result = solve_travel(inputs, weights, max_nfev=max_nfev)
        runtime_s = time.perf_counter() - started
        prediction = result.x
        error = centered_error(prediction, truth, mask)
        full_weight_fraction = (
            1.0 if threshold is None else float(np.mean(inputs.mag[mask] > threshold))
        )
        cache_diff = (
            float(np.max(np.abs(prediction - cached_solved)))
            if mode == CURRENT_MODE
            else float("nan")
        )
        rows.append(
            asdict(
                ThresholdRow(
                    split=split,
                    cohort=cohort,
                    log=log_name,
                    threshold_mode=mode,
                    resolved_mag_x_thresh_mg=(float("nan") if threshold is None else threshold),
                    mag_off_floor=CURRENT_FLOOR,
                    full_weight_fraction=full_weight_fraction,
                    n_samples=int(np.sum(mask)),
                    n_low_travel=int(np.sum(low_mask)),
                    corrected_mag_centered_rmse_mm=rmse(corrected_error[mask]),
                    centered_rmse_mm=rmse(error[mask]),
                    low_travel_centered_rmse_mm=rmse(error[low_mask]),
                    uncentered_rmse_mm=rmse(prediction[mask] - truth[mask]),
                    cache_max_abs_diff_mm=cache_diff,
                    success=bool(result.scipy_result.success),
                    status=int(result.scipy_result.status),
                    nfev=int(result.scipy_result.nfev),
                    cost=float(result.scipy_result.cost),
                    optimality=float(result.scipy_result.optimality),
                    runtime_s=runtime_s,
                )
            )
        )
    return rows


def execute_split(
    split: str,
    cohorts: dict[str, list[str]],
    modes: Sequence[str],
    cache_root: Path,
    max_nfev: int,
    workers: int,
) -> list[dict[str, object]]:
    tasks = [(cohort, log_name) for cohort, logs in cohorts.items() for log_name in logs]
    rows: list[dict[str, object]] = []
    if workers <= 1:
        for index, (cohort, log_name) in enumerate(tasks, 1):
            rows.extend(run_log(split, cohort, log_name, modes, str(cache_root), max_nfev))
            print(f"[{split} {index}/{len(tasks)}] {cohort}/{log_name}", flush=True)
        return rows
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_log, split, cohort, log_name, modes, str(cache_root), max_nfev
            ): (cohort, log_name)
            for cohort, log_name in tasks
        }
        for index, future in enumerate(as_completed(futures), 1):
            cohort, log_name = futures[future]
            rows.extend(future.result())
            print(f"[{split} {index}/{len(tasks)}] {cohort}/{log_name}", flush=True)
    return rows


def aggregate_rows(
    rows: Sequence[dict[str, object]], modes: Sequence[str]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = (
        "resolved_mag_x_thresh_mg",
        "full_weight_fraction",
        "corrected_mag_centered_rmse_mm",
        "centered_rmse_mm",
        "low_travel_centered_rmse_mm",
        "uncentered_rmse_mm",
        "runtime_s",
    )
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        cohorts = sorted({str(row["cohort"]) for row in split_rows})
        for mode in modes:
            mode_rows = [row for row in split_rows if row["threshold_mode"] == mode]
            if not mode_rows:
                continue
            cohort_summaries = []
            for cohort in cohorts:
                selected = [row for row in mode_rows if row["cohort"] == cohort]
                if not selected:
                    continue
                summary = {
                    "split": split,
                    "cohort": cohort,
                    "threshold_mode": mode,
                    "n_logs": len(selected),
                    "success_fraction": mean(selected, "success"),
                    **{key: mean(selected, key) for key in metrics},
                }
                cohort_summaries.append(summary)
                output.append(summary)
            output.append(
                {
                    "split": split,
                    "cohort": "cohort-balanced",
                    "threshold_mode": mode,
                    "n_logs": len(mode_rows),
                    "success_fraction": mean(mode_rows, "success"),
                    **{key: mean(cohort_summaries, key) for key in metrics},
                }
            )
    return output


def find_summary(
    aggregate: Sequence[dict[str, object]], split: str, cohort: str, mode: str
) -> dict[str, object]:
    return next(
        row for row in aggregate
        if row["split"] == split
        and row["cohort"] == cohort
        and row["threshold_mode"] == mode
    )


def select_mode(
    aggregate: Sequence[dict[str, object]], tuning_cohorts: Sequence[str]
) -> tuple[dict[str, object], list[str]]:
    current = {
        cohort: find_summary(aggregate, "tuning", cohort, CURRENT_MODE)
        for cohort in tuning_cohorts
    }
    balanced = [
        row for row in aggregate
        if row["split"] == "tuning" and row["cohort"] == "cohort-balanced"
    ]
    eligible: list[dict[str, object]] = []
    for candidate in balanced:
        mode = str(candidate["threshold_mode"])
        if float(candidate["success_fraction"]) != 1.0:
            continue
        if all(
            float(find_summary(aggregate, "tuning", cohort, mode)[metric])
            <= float(current[cohort][metric]) + 1e-12
            for cohort in tuning_cohorts
            for metric in ("low_travel_centered_rmse_mm", "centered_rmse_mm")
        ):
            eligible.append(candidate)
    if not eligible:
        raise RuntimeError("no candidate, including current, passed the non-regression rule")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["low_travel_centered_rmse_mm"]),
            float(row["centered_rmse_mm"]),
        ),
    )
    return selected, [str(row["threshold_mode"]) for row in eligible]


def paired_change(
    rows: Sequence[dict[str, object]], split: str, cohort: str, mode: str, metric: str
) -> tuple[float, int, int]:
    current = {
        str(row["log"]): float(row[metric])
        for row in rows
        if row["split"] == split
        and row["cohort"] == cohort
        and row["threshold_mode"] == CURRENT_MODE
    }
    candidate = {
        str(row["log"]): float(row[metric])
        for row in rows
        if row["split"] == split
        and row["cohort"] == cohort
        and row["threshold_mode"] == mode
    }
    if current.keys() != candidate.keys():
        raise ValueError(f"unpaired rows for {split}/{cohort}/{mode}/{metric}")
    changes = [candidate[log] - current[log] for log in current]
    return float(np.mean(changes)), sum(change < 0 for change in changes), len(changes)


def render_report(
    modes: Sequence[str],
    tuning_cohorts: dict[str, list[str]],
    held_out_cohorts: dict[str, list[str]],
    selected_mode: str,
    eligible_modes: Sequence[str],
    aggregate: Sequence[dict[str, object]],
    rows: Sequence[dict[str, object]],
) -> str:
    current = find_summary(aggregate, "tuning", "cohort-balanced", CURRENT_MODE)
    selected = find_summary(aggregate, "tuning", "cohort-balanced", selected_mode)
    heldout_name = next(iter(held_out_cohorts))
    heldout_current = find_summary(aggregate, "held-out", heldout_name, CURRENT_MODE)
    heldout_selected = find_summary(aggregate, "held-out", heldout_name, selected_mode)
    replay_diffs = [
        float(row["cache_max_abs_diff_mm"])
        for row in rows
        if row["threshold_mode"] == CURRENT_MODE
    ]
    lines = [
        "# Post-mag-correction fusion `mag_x_thresh` sweep",
        "",
        "## Design",
        "",
        f"The final front fusion solver was replayed with `mag_off_floor={CURRENT_FLOOR:g}` fixed. Threshold modes were {', '.join(f'`{mode}`' for mode in modes)}; `baseline` is the current per-log magnetic baseline and `disabled` gives every sample full magnetic residual weight. Numeric modes are absolute mG thresholds.",
        "",
        f"Tuning used {sum(map(len, tuning_cohorts.values()))} logs from {', '.join(f'`{name}`' for name in tuning_cohorts)}. The robust selection rule required a candidate to avoid worsening both 0–30 mm and overall centered cohort means versus current in every tuning cohort, then minimized cohort-balanced 0–30 mm RMSE. `{heldout_name}` was evaluated only after selection, but this cohort has been examined by an earlier hyperparameter experiment, so its result is exploratory rather than pristine validation.",
        "",
        "## Tuning result",
        "",
        "| threshold mode | mean resolved threshold | full-weight samples | 0–30 mm RMSE | overall RMSE | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in modes:
        item = find_summary(aggregate, "tuning", "cohort-balanced", mode)
        threshold = "disabled" if mode == "disabled" else f"{float(item['resolved_mag_x_thresh_mg']):.0f} mG"
        lines.append(
            f"| `{mode}` | {threshold} | {100 * float(item['full_weight_fraction']):.1f}% | {float(item['low_travel_centered_rmse_mm']):.3f} mm | {float(item['centered_rmse_mm']):.3f} mm | {'yes' if mode in eligible_modes else 'no'} |"
        )
    low_delta = float(selected["low_travel_centered_rmse_mm"]) - float(current["low_travel_centered_rmse_mm"])
    overall_delta = float(selected["centered_rmse_mm"]) - float(current["centered_rmse_mm"])
    lines.extend([
        "",
        f"Selected: **`{selected_mode}`**. On tuning, cohort-balanced 0–30 mm RMSE changed by {low_delta:+.3f} mm and overall centered RMSE changed by {overall_delta:+.3f} mm.",
        "",
        "| tuning cohort | 0–30 mm change | logs improved | overall change | logs improved |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for cohort in tuning_cohorts:
        low_change, low_wins, count = paired_change(
            rows, "tuning", cohort, selected_mode, "low_travel_centered_rmse_mm"
        )
        overall_change, overall_wins, _ = paired_change(
            rows, "tuning", cohort, selected_mode, "centered_rmse_mm"
        )
        lines.append(
            f"| `{cohort}` | {low_change:+.3f} mm | {low_wins}/{count} | {overall_change:+.3f} mm | {overall_wins}/{count} |"
        )
    lines.extend(["", "## Exploratory pod-v1 result", ""])
    if selected_mode == CURRENT_MODE:
        lines.append(
            "No distinct held-out comparison was required because the strict selection retained current behavior."
        )
    else:
        held_low_change, held_low_wins, held_count = paired_change(
            rows, "held-out", heldout_name, selected_mode, "low_travel_centered_rmse_mm"
        )
        held_overall_change, held_overall_wins, _ = paired_change(
            rows, "held-out", heldout_name, selected_mode, "centered_rmse_mm"
        )
        held_uncentered_change, held_uncentered_wins, _ = paired_change(
            rows, "held-out", heldout_name, selected_mode, "uncentered_rmse_mm"
        )
        lines.extend([
            "| mode | 0–30 mm RMSE | overall centered RMSE | uncentered RMSE |",
            "| --- | ---: | ---: | ---: |",
            f"| `baseline` | {float(heldout_current['low_travel_centered_rmse_mm']):.3f} mm | {float(heldout_current['centered_rmse_mm']):.3f} mm | {float(heldout_current['uncentered_rmse_mm']):.3f} mm |",
            f"| `{selected_mode}` | {float(heldout_selected['low_travel_centered_rmse_mm']):.3f} mm | {float(heldout_selected['centered_rmse_mm']):.3f} mm | {float(heldout_selected['uncentered_rmse_mm']):.3f} mm |",
            "",
            f"The selected mode changed mean 0–30 mm RMSE by {held_low_change:+.3f} mm ({held_low_wins}/{held_count} logs improved), overall centered RMSE by {held_overall_change:+.3f} mm ({held_overall_wins}/{held_count}), and uncentered RMSE by {held_uncentered_change:+.3f} mm ({held_uncentered_wins}/{held_count}).",
        ])
    all_tuning_converged = all(
        bool(row["success"] if isinstance(row["success"], bool) else row["success"] == "True")
        for row in rows
        if row["split"] == "tuning"
    )
    lines.extend([
        "",
        "## Checks",
        "",
        f"The current `baseline` replay matched cached final solver output to a worst-case absolute difference of {max(replay_diffs):.3g} mm. {'Every tuning solve converged.' if all_tuning_converged else 'At least one tuning solve did not converge; see per-log metrics.'}",
        "",
        "Full per-log values are in `per_log_metrics.csv`; cohort summaries are in `aggregate_metrics.csv`; the frozen decision is in `selection.json`.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune mag_x_thresh on the final front fusion solver"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "backend" / "run_artifacts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tuning-sets", nargs="+", default=list(DEFAULT_TUNING_SETS))
    parser.add_argument("--held-out-set", default=DEFAULT_HELD_OUT_SET)
    parser.add_argument("--thresholds", type=parse_modes, default=DEFAULT_THRESHOLD_MODES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = tuple(dict.fromkeys(args.thresholds))
    if CURRENT_MODE not in modes:
        raise ValueError(f"threshold modes must include {CURRENT_MODE!r}")
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
        "threshold_modes": modes,
        "fixed_mag_off_floor": CURRENT_FLOOR,
        "low_travel_max_mm": LOW_TRAVEL_MAX_MM,
        "selection_rule": "require no increase in either 0-30 mm or overall centered mean in any tuning cohort, then minimize cohort-balanced 0-30 mm centered RMSE",
        "held_out_note": "pod-v1 is isolated from this sweep's selection but was examined in the preceding mag_off_floor experiment",
        "max_nfev": args.max_nfev,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tuning_rows = execute_split(
        "tuning", tuning_cohorts, modes, args.cache_root, args.max_nfev, args.workers
    )
    tuning_aggregate = aggregate_rows(tuning_rows, modes)
    selected, eligible_modes = select_mode(tuning_aggregate, tuple(tuning_cohorts))
    selected_mode = str(selected["threshold_mode"])
    heldout_modes = tuple(dict.fromkeys((CURRENT_MODE, selected_mode)))
    heldout_rows = execute_split(
        "held-out", held_out_cohorts, heldout_modes, args.cache_root, args.max_nfev, args.workers
    )
    rows = sorted(
        tuning_rows + heldout_rows,
        key=lambda row: (
            str(row["split"]), str(row["cohort"]), str(row["log"]), modes.index(str(row["threshold_mode"]))
        ),
    )
    aggregate = aggregate_rows(rows, modes)
    write_csv(args.output_dir / "per_log_metrics.csv", rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    selection = {
        "selected_threshold_mode": selected_mode,
        "selection_split": "tuning",
        "eligible_threshold_modes": eligible_modes,
        "selection_summary": selected,
        "held_out_modes_evaluated_after_selection": heldout_modes,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_report(
            modes,
            tuning_cohorts,
            held_out_cohorts,
            selected_mode,
            eligible_modes,
            aggregate,
            rows,
        ),
        encoding="utf-8",
    )
    print(f"Selected mag_x_thresh mode={selected_mode}")
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
