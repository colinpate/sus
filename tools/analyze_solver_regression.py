from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from travel_solver_core import (  # noqa: E402
    SolverInputs,
    flatten_1d,
    prepare_solver,
    solve_prepared_travel,
    solver_weights_for_mag_baseline,
    term_costs,
)


def centered_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    pred = flatten_1d(pred)[mask]
    gt = flatten_1d(gt)[mask]
    pred = pred - np.mean(pred)
    gt = gt - np.mean(gt)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def centered_std(x: np.ndarray, mask: np.ndarray) -> float:
    x = flatten_1d(x)[mask]
    x = x - np.mean(x)
    return float(np.std(x))


def mean_pct(mask: np.ndarray) -> float:
    if len(mask) == 0:
        return float("nan")
    return 100.0 * float(np.mean(mask))


@dataclass
class SolverReplay:
    travel: np.ndarray
    time_s: np.ndarray
    mask: np.ndarray
    mag: np.ndarray
    mag_preds: np.ndarray
    solved_cache: np.ndarray
    zv: np.ndarray
    mag_baseline: float
    inputs: SolverInputs

    @classmethod
    def from_cache(cls, cache: np.lib.npyio.NpzFile) -> "SolverReplay":
        mag_baseline = float(flatten_1d(cache["mag_baseline"])[0])
        inputs = SolverInputs(
            time_s=cache["travel__t"],
            accel_mm_s2=flatten_1d(cache["accel/lpfhp/proj__x"]) * 1000.0,
            mag=cache["mag/proj/corr/lpf__x"],
            mag_preds_mm=cache["travel/mag_model/adj__x"],
            mag_zv_points=cache["mag_zv_points"],
            mag_baseline=mag_baseline,
        )
        return cls(
            travel=flatten_1d(cache["travel__x"]),
            time_s=inputs.time_s,
            mask=np.asarray(cache["boring_mask"]).astype(bool).reshape(-1),
            mag=inputs.mag,
            mag_preds=inputs.mag_preds_mm,
            solved_cache=flatten_1d(cache["travel/solved__x"]),
            zv=inputs.mag_zv_mask,
            mag_baseline=mag_baseline,
            inputs=inputs,
        )


def masked_centered_err(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred = flatten_1d(pred)[mask]
    gt = flatten_1d(gt)[mask]
    pred = pred - np.mean(pred)
    gt = gt - np.mean(gt)
    return pred - gt


def summarize_windows(
    time_s: np.ndarray,
    mask: np.ndarray,
    travel: np.ndarray,
    mag: np.ndarray,
    solved: np.ndarray,
    mag_preds: np.ndarray,
    zv: np.ndarray,
    anchor_mask: np.ndarray,
    window_s: float,
    top_k: int,
) -> list[dict[str, float]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    dt = float(np.median(np.diff(time_s[idx]))) if len(idx) > 1 else 0.01
    win = max(1, int(round(window_s / dt)))
    out: list[dict[str, float]] = []
    for start in range(0, len(idx) - win + 1, win):
        w_idx = idx[start : start + win]
        gt_w = travel[w_idx]
        mag_w = mag_preds[w_idx]
        solved_w = solved[w_idx]
        gt_c = gt_w - np.mean(gt_w)
        mag_c = mag_w - np.mean(mag_w)
        solved_c = solved_w - np.mean(solved_w)
        mag_rmse = float(np.sqrt(np.mean((mag_c - gt_c) ** 2)))
        solved_rmse = float(np.sqrt(np.mean((solved_c - gt_c) ** 2)))
        delta = solved_rmse - mag_rmse
        correction = solved_w - mag_w
        out.append(
            {
                "t0": float(time_s[w_idx[0]]),
                "t1": float(time_s[w_idx[-1]]),
                "delta_rmse": delta,
                "solved_rmse": solved_rmse,
                "mag_rmse": mag_rmse,
                "anchor_on_pct": mean_pct(anchor_mask[w_idx]),
                "zv_pct": mean_pct(zv[w_idx]),
                "travel_mean": float(np.mean(travel[w_idx])),
                "travel_std": float(np.std(gt_c)),
                "mag_mean": float(np.mean(mag[w_idx])),
                "corr_mean": float(np.mean(correction)),
                "corr_std": float(np.std(correction)),
            }
        )
    out.sort(key=lambda row: row["delta_rmse"], reverse=True)
    return out[:top_k]


def print_cost_row(label: str, costs: dict[str, float]) -> None:
    parts = [f"{k}={v:.1f}" for k, v in costs.items()]
    print(f"{label}: " + ", ".join(parts))


def run_log(log_name: str, cache_root: Path, window_s: float, top_k: int, run_ablation: bool) -> None:
    cache = np.load(cache_root / log_name / "cache" / "all.npz")
    replay = SolverReplay.from_cache(cache)
    current = solver_weights_for_mag_baseline(replay.mag_baseline)
    prepared = prepare_solver(replay.inputs, current)

    result = solve_prepared_travel(prepared)
    solved = result.x
    init_terms = result.init_terms
    opt_terms = result.opt_terms
    mask = replay.mask
    mag_rmse = centered_rmse(replay.mag_preds, replay.travel, mask)
    solved_rmse = centered_rmse(solved, replay.travel, mask)
    cache_rmse = centered_rmse(replay.solved_cache, replay.travel, mask)
    correction = solved - replay.mag_preds
    cache_diff = solved - replay.solved_cache

    print(f"\n== {log_name} ==")
    print(
        "Solver config:",
        f"mag_x_thresh={current.mag_x_thresh:.3f}",
        f"mag_off_floor={current.mag_off_floor:.3f}",
    )
    print(
        "RMSE:",
        f"mag_model_adj={mag_rmse:.3f}",
        f"solver={solved_rmse:.3f}",
        f"cache_solver={cache_rmse:.3f}",
        f"delta={solved_rmse - mag_rmse:.3f}",
    )
    print(
        "Cache diff:",
        f"rmse_delta={solved_rmse - cache_rmse:.6f}",
        f"max_abs={np.max(np.abs(cache_diff)):.6f}",
        f"mean_abs={np.mean(np.abs(cache_diff)):.6f}",
    )
    print(
        "Amplitude:",
        f"travel_std={centered_std(replay.travel, mask):.3f}",
        f"mag_std={centered_std(replay.mag_preds, mask):.3f}",
        f"solver_std={centered_std(solved, mask):.3f}",
    )
    print(
        "Correction:",
        f"mean={np.mean(correction[mask]):.3f}",
        f"std={np.std(correction[mask]):.3f}",
        f"mean_abs={np.mean(np.abs(correction[mask])):.3f}",
        f"bias_b={opt_terms.b:.3f}",
    )

    print_cost_row("Init weighted costs", term_costs(init_terms, current))
    print_cost_row("Opt weighted costs", term_costs(opt_terms, current))

    masked_travel = replay.travel[mask]
    masked_mag = replay.mag[mask]
    masked_zv = replay.zv[mask]
    masked_corr = correction[mask]
    masked_solver_err = np.abs(masked_centered_err(solved, replay.travel, mask))
    masked_mag_err = np.abs(masked_centered_err(replay.mag_preds, replay.travel, mask))
    anchor_on = prepared.mag_anchor_mask[mask]
    low_travel = masked_travel < 20.0
    high_travel = masked_travel > np.percentile(masked_travel, 80.0)

    for label, cond in (
        ("anchor_on", anchor_on),
        ("anchor_off", ~anchor_on),
        ("zv", masked_zv),
        ("non_zv", ~masked_zv),
        ("travel<20", low_travel),
        ("travel>p80", high_travel),
    ):
        if not np.any(cond):
            continue
        print(
            f"{label}:",
            f"n={int(np.sum(cond))}",
            f"delta_rmse={np.sqrt(np.mean(masked_solver_err[cond] ** 2)) - np.sqrt(np.mean(masked_mag_err[cond] ** 2)):.3f}",
            f"mean_abs_corr={np.mean(np.abs(masked_corr[cond])):.3f}",
            f"anchor_pct={mean_pct(anchor_on[cond]):.1f}",
            f"zv_pct={mean_pct(masked_zv[cond]):.1f}",
        )

    print("Worst windows by solver delta:")
    for row in summarize_windows(
        replay.time_s,
        mask,
        replay.travel,
        replay.mag,
        solved,
        replay.mag_preds,
        replay.zv,
        prepared.mag_anchor_mask,
        window_s,
        top_k,
    ):
        print(
            f"  {row['t0']:.1f}-{row['t1']:.1f}s",
            f"delta={row['delta_rmse']:.3f}",
            f"solver={row['solved_rmse']:.3f}",
            f"mag={row['mag_rmse']:.3f}",
            f"anchor_on={row['anchor_on_pct']:.1f}%",
            f"zv={row['zv_pct']:.1f}%",
            f"travel_mean={row['travel_mean']:.1f}",
            f"corr_std={row['corr_std']:.3f}",
        )

    if not run_ablation:
        return

    ablations = [
        ("no_oob", replace(current, oob=0.0)),
        ("no_zupt", replace(current, zupt_v=0.0)),
        ("strong_mag", replace(current, mag_x=400.0)),
        ("weak_dyn", replace(current, v0=1.25, x0=250.0)),
        ("strong_dyn", replace(current, v0=5.0, x0=1000.0)),
        ("hard_off", replace(current, mag_off_floor=0.0)),
    ]
    print("Ablations:")
    for name, weights in ablations:
        ablation_result = solve_prepared_travel(prepare_solver(replay.inputs, weights))
        rmse = centered_rmse(ablation_result.x, replay.travel, mask)
        print(f"  {name}: rmse={rmse:.3f}, delta_vs_mag={rmse - mag_rmse:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze solver regressions versus mag-model predictions")
    parser.add_argument("logs", nargs="+", help="Log names to analyze")
    parser.add_argument("--cache-root", default="backend/run_artifacts", help="Cache root directory")
    parser.add_argument("--window-s", type=float, default=5.0, help="Window size in seconds for local summaries")
    parser.add_argument("--top-k", type=int, default=5, help="How many worst windows to print")
    parser.add_argument("--with-ablation", action="store_true", help="Run a small set of solver ablations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    for log_name in args.logs:
        run_log(log_name, cache_root, window_s=args.window_s, top_k=args.top_k, run_ablation=args.with_ablation)


if __name__ == "__main__":
    main()
