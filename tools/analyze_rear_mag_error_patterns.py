#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from rear_mag_model import load_ws  # noqa: E402


TRAVEL_QUARTILES = 4
SPEED_QUARTILES = 4
ACCEL_QUARTILES = 4
PLOT_SAMPLE_COUNT = 8000


@dataclass
class LogSummary:
    name: str
    n_roi: int
    static_rmse: float
    static_mae: float
    static_corr: float
    direction_split_rmse: float
    direction_split_gain: float
    slow_train_rmse: float
    slow_train_gain: float
    lag_best_samples: int
    lag_best_rmse: float
    lag_gain: float
    speed_err_spearman: float
    accel_err_spearman: float
    speed_partial_err_spearman: float
    accel_partial_err_spearman: float
    travel_err_spearman: float
    time_err_spearman: float
    signed_resid_vs_velocity_spearman: float
    low_sensitivity_err_spearman: float
    speed_bin_rmses: list[float]
    accel_bin_rmses: list[float]
    travel_bin_rmses: list[float]
    time_bin_rmses: list[float]
    sensitivity_bin_rmses: list[float]
    travel_speed_heatmap_sse: list[list[float]]
    travel_speed_heatmap_counts: list[list[int]]
    travel_accel_heatmap_sse: list[list[float]]
    travel_accel_heatmap_counts: list[list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze error patterns in rear travel-vs-mag ground truth.")
    parser.add_argument(
        "--logs",
        nargs="*",
        default=[
            "log136_rear",
            "log137_rear",
            "log140_rear",
            "log141_rear",
            "log142_rear",
            "log143_rear",
            "log144_rear",
            "log145_rear",
        ],
        help="Log names without extension.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/rear_mag_error_patterns"),
        help="Directory for report artifacts.",
    )
    return parser.parse_args()


def smooth_derivatives(travel: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = float(np.median(np.diff(t)))
    window = min(len(travel) - (1 - len(travel) % 2), 31)
    window = max(window, 7)
    if window % 2 == 0:
        window -= 1
    vel = savgol_filter(travel, window, 3, deriv=1, delta=dt, mode="interp")
    accel = savgol_filter(travel, window, 3, deriv=2, delta=dt, mode="interp")
    return vel, accel


def rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def fit_isotonic(mag: np.ndarray, travel: np.ndarray) -> tuple[IsotonicRegression, np.ndarray, np.ndarray]:
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    pred = iso.fit_transform(mag, travel)
    resid = travel - pred
    return iso, pred, resid


def quartile_rmses(values: np.ndarray, resid: np.ndarray) -> list[float]:
    q = np.quantile(values, [0.25, 0.5, 0.75])
    bins = np.digitize(values, q)
    return [rmse(resid[bins == i]) for i in range(4)]


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    x_rank = scipy.stats.rankdata(np.asarray(x, dtype=float))
    y_rank = scipy.stats.rankdata(np.asarray(y, dtype=float))
    ctrl = [scipy.stats.rankdata(np.asarray(c, dtype=float)) for c in controls]
    design = np.column_stack([np.ones_like(x_rank), *ctrl])
    beta_x, *_ = np.linalg.lstsq(design, x_rank, rcond=None)
    beta_y, *_ = np.linalg.lstsq(design, y_rank, rcond=None)
    resid_x = x_rank - design @ beta_x
    resid_y = y_rank - design @ beta_y
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def estimate_local_sensitivity(travel: np.ndarray, mag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(float(np.min(travel)), float(np.max(travel)), 80)
    centers: list[float] = []
    mag_med: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (travel >= lo) & (travel < hi)
        if int(np.sum(mask)) < 50:
            continue
        centers.append(0.5 * (lo + hi))
        mag_med.append(float(np.median(mag[mask])))

    travel_centers = np.asarray(centers, dtype=float)
    mag_medians = np.asarray(mag_med, dtype=float)
    abs_dmag_dtravel = np.abs(np.gradient(mag_medians, travel_centers))
    return travel_centers, mag_medians, abs_dmag_dtravel


def best_lag_rmse(mag: np.ndarray, travel: np.ndarray, max_lag: int = 12) -> tuple[int, float]:
    best = (0, rmse(travel - fit_isotonic(mag, travel)[1]))
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            mag_l = mag[-lag:]
            travel_l = travel[:lag]
        elif lag > 0:
            mag_l = mag[:-lag]
            travel_l = travel[lag:]
        else:
            mag_l = mag
            travel_l = travel

        if len(mag_l) < 100:
            continue
        _, pred_l, _ = fit_isotonic(mag_l, travel_l)
        lag_rmse = rmse(travel_l - pred_l)
        if lag_rmse < best[1]:
            best = (lag, lag_rmse)
    return best


def direction_split_rmse(mag: np.ndarray, travel: np.ndarray, vel: np.ndarray) -> float:
    vel_thresh = float(np.percentile(np.abs(vel), 35))
    comp = vel > vel_thresh
    reb = vel < -vel_thresh
    use_mask = comp | reb
    pred = np.full_like(travel, np.nan, dtype=float)

    if int(np.sum(comp)) >= 50:
        iso_comp = IsotonicRegression(increasing=False, out_of_bounds="clip")
        pred[comp] = iso_comp.fit_transform(mag[comp], travel[comp])
    if int(np.sum(reb)) >= 50:
        iso_reb = IsotonicRegression(increasing=False, out_of_bounds="clip")
        pred[reb] = iso_reb.fit_transform(mag[reb], travel[reb])

    return rmse(travel[use_mask] - pred[use_mask])


def slow_train_rmse(mag: np.ndarray, travel: np.ndarray, vel: np.ndarray) -> float:
    slow_thresh = float(np.percentile(np.abs(vel), 40))
    slow_mask = np.abs(vel) <= slow_thresh
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(mag[slow_mask], travel[slow_mask])
    pred = iso.predict(mag)
    return rmse(travel - pred)


def heatmap_sse(
    travel: np.ndarray,
    y_values: np.ndarray,
    sqerr: np.ndarray,
    *,
    y_quartiles: int,
) -> tuple[np.ndarray, np.ndarray]:
    travel_pct = travel / max(float(np.max(travel)), 1e-6)
    travel_q = np.quantile(travel_pct, [0.25, 0.5, 0.75])
    y_q = np.quantile(np.abs(y_values), [0.25, 0.5, 0.75])
    travel_bins = np.digitize(travel_pct, travel_q)
    y_bins = np.digitize(np.abs(y_values), y_q)
    sqerr_sum = np.zeros((TRAVEL_QUARTILES, y_quartiles), dtype=float)
    counts = np.zeros((TRAVEL_QUARTILES, y_quartiles), dtype=int)
    for i in range(TRAVEL_QUARTILES):
        for j in range(y_quartiles):
            mask = (travel_bins == i) & (y_bins == j)
            if np.any(mask):
                sqerr_sum[i, j] = float(np.sum(sqerr[mask]))
                counts[i, j] = int(np.sum(mask))
    return sqerr_sum, counts


def analyze_log(log_name: str) -> tuple[LogSummary, dict[str, np.ndarray]]:
    _, mag, t, travel, _, _, _, roi_mask = load_ws(log_name)
    roi = roi_mask.astype(bool)
    travel_roi = travel[roi]
    mag_roi = mag[roi]
    t_roi = t[roi]
    vel, accel = smooth_derivatives(travel, t)
    vel_roi = vel[roi]
    accel_roi = accel[roi]

    iso, pred, resid = fit_isotonic(mag_roi, travel_roi)
    abs_err = np.abs(resid)
    sqerr = resid**2

    sensitivity_travel, sensitivity_mag, abs_dmag_dtravel = estimate_local_sensitivity(travel_roi, mag_roi)
    local_abs_slope = np.interp(
        travel_roi,
        sensitivity_travel,
        abs_dmag_dtravel,
        left=float(abs_dmag_dtravel[0]),
        right=float(abs_dmag_dtravel[-1]),
    )

    static_rmse = rmse(resid)
    static_mae = float(np.mean(abs_err))
    static_corr = float(np.corrcoef(pred, travel_roi)[0, 1])

    dir_rmse = direction_split_rmse(mag_roi, travel_roi, vel_roi)
    slow_rmse = slow_train_rmse(mag_roi, travel_roi, vel_roi)
    lag_samples, lag_rmse = best_lag_rmse(mag_roi, travel_roi)

    speed_sse_grid, speed_counts_grid = heatmap_sse(
        travel_roi,
        vel_roi,
        sqerr,
        y_quartiles=SPEED_QUARTILES,
    )
    accel_sse_grid, accel_counts_grid = heatmap_sse(
        travel_roi,
        accel_roi,
        sqerr,
        y_quartiles=ACCEL_QUARTILES,
    )
    summary = LogSummary(
        name=log_name,
        n_roi=int(np.sum(roi)),
        static_rmse=static_rmse,
        static_mae=static_mae,
        static_corr=static_corr,
        direction_split_rmse=dir_rmse,
        direction_split_gain=static_rmse - dir_rmse,
        slow_train_rmse=slow_rmse,
        slow_train_gain=static_rmse - slow_rmse,
        lag_best_samples=int(lag_samples),
        lag_best_rmse=lag_rmse,
        lag_gain=static_rmse - lag_rmse,
        speed_err_spearman=float(scipy.stats.spearmanr(abs_err, np.abs(vel_roi)).correlation),
        accel_err_spearman=float(scipy.stats.spearmanr(abs_err, np.abs(accel_roi)).correlation),
        speed_partial_err_spearman=partial_spearman(
            abs_err,
            np.abs(vel_roi),
            [travel_roi, np.abs(accel_roi)],
        ),
        accel_partial_err_spearman=partial_spearman(
            abs_err,
            np.abs(accel_roi),
            [travel_roi, np.abs(vel_roi)],
        ),
        travel_err_spearman=float(scipy.stats.spearmanr(abs_err, travel_roi).correlation),
        time_err_spearman=float(scipy.stats.spearmanr(abs_err, t_roi).correlation),
        signed_resid_vs_velocity_spearman=float(scipy.stats.spearmanr(resid, vel_roi).correlation),
        low_sensitivity_err_spearman=float(scipy.stats.spearmanr(abs_err, -local_abs_slope).correlation),
        speed_bin_rmses=quartile_rmses(np.abs(vel_roi), resid),
        accel_bin_rmses=quartile_rmses(np.abs(accel_roi), resid),
        travel_bin_rmses=quartile_rmses(travel_roi, resid),
        time_bin_rmses=quartile_rmses(t_roi, resid),
        sensitivity_bin_rmses=quartile_rmses(-local_abs_slope, resid),
        travel_speed_heatmap_sse=speed_sse_grid.tolist(),
        travel_speed_heatmap_counts=speed_counts_grid.tolist(),
        travel_accel_heatmap_sse=accel_sse_grid.tolist(),
        travel_accel_heatmap_counts=accel_counts_grid.tolist(),
    )

    detail = {
        "travel_roi": travel_roi,
        "mag_roi": mag_roi,
        "pred_roi": pred,
        "resid_roi": resid,
        "vel_roi": vel_roi,
        "accel_roi": accel_roi,
        "time_roi": t_roi,
        "sensitivity_travel": sensitivity_travel,
        "sensitivity_mag": sensitivity_mag,
        "abs_dmag_dtravel": abs_dmag_dtravel,
    }
    return summary, detail


def plot_log(summary: LogSummary, detail: dict[str, np.ndarray], out_path: Path) -> None:
    travel = detail["travel_roi"]
    mag = detail["mag_roi"]
    pred = detail["pred_roi"]
    resid = detail["resid_roi"]
    vel = detail["vel_roi"]
    sensitivity_travel = detail["sensitivity_travel"]
    abs_dmag_dtravel = detail["abs_dmag_dtravel"]

    rng = np.random.default_rng(0)
    idx = np.arange(len(travel))
    if len(idx) > PLOT_SAMPLE_COUNT:
        idx = np.sort(rng.choice(idx, size=PLOT_SAMPLE_COUNT, replace=False))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sc = axes[0, 0].scatter(
        mag[idx],
        travel[idx],
        c=np.abs(vel[idx]),
        s=4,
        cmap="viridis",
        alpha=0.4,
        linewidths=0,
    )
    axes[0, 0].scatter(mag[idx], pred[idx], s=3, alpha=0.25, color="tab:red", linewidths=0)
    axes[0, 0].set_title(f"{summary.name}: Travel vs Mag")
    axes[0, 0].set_xlabel("Projected mag (mG)")
    axes[0, 0].set_ylabel("Travel (mm)")
    fig.colorbar(sc, ax=axes[0, 0], label="|velocity| (mm/s)")

    axes[0, 1].scatter(
        travel[idx],
        resid[idx],
        c=np.abs(vel[idx]),
        s=4,
        cmap="plasma",
        alpha=0.4,
        linewidths=0,
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_title("Residual vs Travel")
    axes[0, 1].set_xlabel("Travel (mm)")
    axes[0, 1].set_ylabel("Travel residual (mm)")

    axes[1, 0].scatter(
        np.abs(vel[idx]),
        np.abs(resid[idx]),
        s=4,
        alpha=0.35,
        color="tab:blue",
        linewidths=0,
    )
    axes[1, 0].set_title("Absolute Error vs Speed")
    axes[1, 0].set_xlabel("|velocity| (mm/s)")
    axes[1, 0].set_ylabel("|travel residual| (mm)")

    ax2 = axes[1, 1]
    ax2.plot(sensitivity_travel, abs_dmag_dtravel, color="tab:green", linewidth=2)
    ax2.set_title("Mag Sensitivity vs Travel")
    ax2.set_xlabel("Travel (mm)")
    ax2.set_ylabel("|dmag/dtravel| (mG/mm)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2b = ax2.twinx()
    travel_q = np.quantile(travel, [0.25, 0.5, 0.75])
    travel_bins = np.digitize(travel, travel_q)
    bin_centers = [float(np.median(travel[travel_bins == i])) for i in range(4)]
    ax2b.plot(bin_centers, summary.travel_bin_rmses, color="tab:red", marker="o")
    ax2b.set_ylabel("Travel-bin RMSE (mm)", color="tab:red")
    ax2b.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle(
        f"{summary.name}: static RMSE {summary.static_rmse:.3f} mm, "
        f"lag gain {summary.lag_gain:.3f} mm"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_aggregate_heatmap(
    log_summaries: list[LogSummary],
    *,
    sqerr_attr: str,
    count_attr: str,
    out_path: Path,
    x_label: str,
    title: str,
) -> None:
    sqerr = np.zeros((TRAVEL_QUARTILES, SPEED_QUARTILES), dtype=float)
    counts = np.zeros((TRAVEL_QUARTILES, SPEED_QUARTILES), dtype=int)
    for summary in log_summaries:
        sqerr += np.asarray(getattr(summary, sqerr_attr), dtype=float)
        counts += np.asarray(getattr(summary, count_attr), dtype=int)
    rmse_grid = np.sqrt(sqerr / np.maximum(counts, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rmse_grid, origin="lower", cmap="magma")
    for i in range(TRAVEL_QUARTILES):
        for j in range(SPEED_QUARTILES):
            ax.text(j, i, f"{rmse_grid[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_xlabel(x_label)
    ax.set_ylabel("travel quartile")
    ax.set_xticks(range(SPEED_QUARTILES))
    ax.set_yticks(range(TRAVEL_QUARTILES))
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="RMSE (mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def write_report(log_summaries: list[LogSummary], out_dir: Path) -> None:
    sorted_by_rmse = sorted(log_summaries, key=lambda x: x.static_rmse, reverse=True)
    worst_log = sorted_by_rmse[0].name
    best_log = sorted(log_summaries, key=lambda x: x.static_rmse)[0].name

    speed_sqerr = np.zeros((TRAVEL_QUARTILES, SPEED_QUARTILES), dtype=float)
    speed_counts = np.zeros((TRAVEL_QUARTILES, SPEED_QUARTILES), dtype=int)
    accel_sqerr = np.zeros((TRAVEL_QUARTILES, ACCEL_QUARTILES), dtype=float)
    accel_counts = np.zeros((TRAVEL_QUARTILES, ACCEL_QUARTILES), dtype=int)
    for summary in log_summaries:
        speed_sqerr += np.asarray(summary.travel_speed_heatmap_sse, dtype=float)
        speed_counts += np.asarray(summary.travel_speed_heatmap_counts, dtype=int)
        accel_sqerr += np.asarray(summary.travel_accel_heatmap_sse, dtype=float)
        accel_counts += np.asarray(summary.travel_accel_heatmap_counts, dtype=int)
    speed_rmse_grid = np.sqrt(speed_sqerr / np.maximum(speed_counts, 1))
    accel_rmse_grid = np.sqrt(accel_sqerr / np.maximum(accel_counts, 1))

    lines = [
        "# Rear Mag Error Pattern Analysis",
        "",
        "Logs used:",
        "",
    ]
    lines.extend([f"- `{summary.name}`" for summary in log_summaries])
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            f"- The strongest recurring pattern is **error growth at deep travel**. Across logs, the top travel quartile has the highest static monotonic RMSE.",
            f"- **Speed matters too**, but especially at deep travel. The aggregate travel/speed heatmap peaks in the highest-travel, highest-speed cell at `{speed_rmse_grid[3, 3]:.2f} mm`.",
            f"- **Acceleration matters**, but a bit less cleanly. The aggregate travel/acceleration heatmap peaks in the highest-travel, highest-acceleration cell at `{accel_rmse_grid[3, 3]:.2f} mm`.",
            f"- After controlling for travel and acceleration, the mean partial Spearman correlation between `|err|` and `|v|` is `{mean([s.speed_partial_err_spearman for s in log_summaries]):.3f}`.",
            f"- After controlling for travel and speed, the mean partial Spearman correlation between `|err|` and `|a|` is `{mean([s.accel_partial_err_spearman for s in log_summaries]):.3f}`.",
            f"- A simple **compression vs rebound split does not help**. Mean gain from direction-specific isotonic fits is `{mean([s.direction_split_gain for s in log_summaries]):.3f} mm`, which is slightly negative.",
            f"- A small **timing lag** is present but secondary. The best lag is almost always `-1` sample, with mean RMSE gain `{mean([s.lag_gain for s in log_summaries]):.3f} mm`.",
            f"- Error tracks **low mag sensitivity** as well as speed. Mean Spearman correlations: `|err|` vs `|v|` = `{mean([s.speed_err_spearman for s in log_summaries]):.3f}`, `|err|` vs `|a|` = `{mean([s.accel_err_spearman for s in log_summaries]):.3f}`, `|err|` vs low sensitivity = `{mean([s.low_sensitivity_err_spearman for s in log_summaries]):.3f}`.",
            f"- Training a static oracle only on slower points does **not** materially improve all-point RMSE. Mean gain from slow-point-only training is `{mean([s.slow_train_gain for s in log_summaries]):.3f} mm`.",
            "",
            "## Interpretation",
            "",
            "- The data does not mainly look like a two-branch hysteresis problem. If it were, separate compression/rebound fits would help noticeably, but they do not.",
            "- The data looks more like a **quasi-static monotonic map with a weak-sensitivity region near high travel**, where travel becomes hard to infer accurately from mag because `|dmag/dtravel|` gets small.",
            "- High speed makes that weak-sensitivity region worse. Acceleration also tags the bad region, but its independent effect is smaller once travel and speed are accounted for.",
            "- Time/drift effects exist in some logs but are not consistent enough to look like the primary global issue.",
            "",
            "## What Probably Won't Help Much",
            "",
            "- Training the same single static map only on slow or “clean” points.",
            "- Splitting only by motion direction.",
            "",
            "## What Might Help Next",
            "",
            "- Improve the **mag projection / feature** so the high-travel region has more sensitivity.",
            "- Allow a slightly richer static map than the current power-law model.",
            "- Add a small dynamic correction term keyed off speed or a tiny lag compensation, since the lag signal is weak but consistent.",
            "",
            "## Per-Log Summary",
            "",
            "| Log | Static RMSE (mm) | Lag Gain | `rho(|err|,|v|)` | `rho(|err|,|a|)` | Partial `rho(|err|,|v|)` | Partial `rho(|err|,|a|)` |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        [
            f"| `{s.name}` | {s.static_rmse:.3f} | {s.lag_gain:.3f} | {s.speed_err_spearman:.3f} | {s.accel_err_spearman:.3f} | {s.speed_partial_err_spearman:.3f} | {s.accel_partial_err_spearman:.3f} |"
            for s in log_summaries
        ]
    )
    lines.extend(
        [
            "",
            "Aggregate travel/speed heatmap:",
            "",
            f"![Aggregate Heatmap]({(out_dir / 'aggregate_travel_speed_heatmap.png').resolve()})",
            "",
            "Aggregate travel/acceleration heatmap:",
            "",
            f"![Aggregate Accel Heatmap]({(out_dir / 'aggregate_travel_accel_heatmap.png').resolve()})",
            "",
            f"Representative lowest static-RMSE log: `{best_log}`",
            "",
            f"![{best_log}]({(out_dir / f'{best_log}_patterns.png').resolve()})",
            "",
            f"Representative highest static-RMSE log: `{worst_log}`",
            "",
            f"![{worst_log}]({(out_dir / f'{worst_log}_patterns.png').resolve()})",
            "",
        ]
    )

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[LogSummary] = []
    details_by_log: dict[str, dict[str, np.ndarray]] = {}
    for log_name in args.logs:
        summary, detail = analyze_log(log_name)
        summaries.append(summary)
        details_by_log[log_name] = detail

    plot_aggregate_heatmap(
        summaries,
        sqerr_attr="travel_speed_heatmap_sse",
        count_attr="travel_speed_heatmap_counts",
        out_path=out_dir / "aggregate_travel_speed_heatmap.png",
        x_label="|velocity| quartile",
        title="Static monotonic RMSE by travel and speed quartile",
    )
    plot_aggregate_heatmap(
        summaries,
        sqerr_attr="travel_accel_heatmap_sse",
        count_attr="travel_accel_heatmap_counts",
        out_path=out_dir / "aggregate_travel_accel_heatmap.png",
        x_label="|acceleration| quartile",
        title="Static monotonic RMSE by travel and acceleration quartile",
    )
    for summary in summaries:
        plot_log(summary, details_by_log[summary.name], out_dir / f"{summary.name}_patterns.png")

    payload = {
        "logs": [asdict(summary) for summary in summaries],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(summaries, out_dir)


if __name__ == "__main__":
    main()
