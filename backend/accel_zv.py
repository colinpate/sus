from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from classes.sensor_loader import Workspace
from classes.step import Step
from classes.time_series import TimeSeries


ZVCorrectionMode = Literal["linear_velocity", "pchip_velocity", "smoothed_bias"]


def mag_zv_prominence(mag: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """Estimate each mag turning point's local prominence against neighboring ZV points."""
    mag = np.asarray(mag, dtype=float).reshape(-1)
    idxs = np.asarray(idxs, dtype=int).reshape(-1)
    scores = np.zeros(len(idxs), dtype=float)
    if len(idxs) < 3:
        return scores

    center = mag[idxs[1:-1]]
    left = np.abs(center - mag[idxs[:-2]])
    right = np.abs(center - mag[idxs[2:]])
    scores[1:-1] = np.minimum(left, right)
    scores[0] = scores[1]
    scores[-1] = scores[-2]
    return scores


def filter_mag_zv_points(
    idxs: np.ndarray,
    mag: np.ndarray,
    t: np.ndarray,
    *,
    min_prominence_mg: float = 0.0,
    min_separation_s: float = 0.0,
) -> np.ndarray:
    """Filter mag ZV points using local mag prominence and optional time suppression."""
    mag = np.asarray(mag, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    idxs = np.unique(np.asarray(idxs, dtype=int).reshape(-1))
    idxs = idxs[(idxs >= 0) & (idxs < len(mag))]
    if len(idxs) == 0:
        return idxs

    scores = mag_zv_prominence(mag, idxs)
    keep = scores >= float(min_prominence_mg)
    idxs = idxs[keep]
    scores = scores[keep]

    if min_separation_s <= 0.0 or len(idxs) <= 1:
        return idxs

    dt = float(np.median(np.diff(t)))
    min_sep_samples = max(1, int(round(float(min_separation_s) / dt)))

    # Non-maximum suppression keeps the strongest extrema in dense noisy clusters.
    kept: list[int] = []
    for order_i in np.argsort(scores)[::-1]:
        idx = int(idxs[order_i])
        if all(abs(idx - kept_idx) >= min_sep_samples for kept_idx in kept):
            kept.append(idx)

    return np.asarray(sorted(kept), dtype=int)


def correct_accel_with_zv(
    accel: np.ndarray,
    t: np.ndarray,
    zv_points: np.ndarray,
    *,
    mode: ZVCorrectionMode = "linear_velocity",
    smooth_bias_s: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """Use ZV anchors to remove acceleration drift from a full 1D acceleration signal.

    The correction integrates acceleration once, treats the integrated velocity at
    mag ZV points as drift, and subtracts an interpolated drift derivative.
    """
    accel = np.asarray(accel, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    zv_points = np.unique(np.asarray(zv_points, dtype=int).reshape(-1))
    zv_points = zv_points[(zv_points >= 0) & (zv_points < len(accel))]

    if len(zv_points) < 2:
        return accel.copy(), {
            "zv_count": float(len(zv_points)),
            "bias_std": 0.0,
            "bias_abs_p95": 0.0,
        }

    raw_v = cumulative_trapezoid(accel, t, initial=0.0)
    anchor_t = t[zv_points]
    anchor_v = raw_v[zv_points]

    if mode == "linear_velocity":
        drift_v = np.interp(
            t,
            anchor_t,
            anchor_v,
            left=float(anchor_v[0]),
            right=float(anchor_v[-1]),
        )
        drift_a = np.gradient(drift_v, t)
    elif mode == "pchip_velocity":
        interp = PchipInterpolator(anchor_t, anchor_v, extrapolate=False)
        drift_v = interp(t)
        drift_v[t < anchor_t[0]] = anchor_v[0]
        drift_v[t > anchor_t[-1]] = anchor_v[-1]
        drift_a = np.gradient(drift_v, t)
    elif mode == "smoothed_bias":
        drift_a = np.zeros_like(accel)
        for start, stop in zip(zv_points[:-1], zv_points[1:]):
            duration = t[stop] - t[start]
            if duration <= 0.0:
                continue
            drift_a[start : stop + 1] = (raw_v[stop] - raw_v[start]) / duration
        sigma = max(1.0, float(smooth_bias_s) / float(np.median(np.diff(t))))
        drift_a = gaussian_filter1d(drift_a, sigma=sigma, mode="nearest")
    else:
        raise ValueError(f"Unknown ZV correction mode {mode!r}")

    corrected = accel - drift_a
    stats = {
        "zv_count": float(len(zv_points)),
        "bias_std": float(np.std(drift_a)),
        "bias_abs_p95": float(np.percentile(np.abs(drift_a), 95.0)),
    }
    return corrected, stats


@dataclass
class CorrectAccelWithMagZV(Step):
    """Correct projected acceleration with magnetometer zero-velocity anchors."""

    mode: ZVCorrectionMode = "linear_velocity"
    min_prominence_mg: float = 0.0
    min_separation_s: float = 0.0
    smooth_bias_s: float = 0.05

    def run(self, ws: Workspace) -> None:
        accel_ts: TimeSeries = ws[self.inputs[0]]
        mag_ts: TimeSeries = ws[self.inputs[1]]
        zv_points: np.ndarray = ws[self.inputs[2]]

        accel = accel_ts.x[:, 0]
        mag = mag_ts.x[:, 0]
        idxs = filter_mag_zv_points(
            zv_points,
            mag,
            accel_ts.t,
            min_prominence_mg=float(self.param(ws, "min_prominence_mg")),
            min_separation_s=float(self.param(ws, "min_separation_s")),
        )
        corrected, stats = correct_accel_with_zv(
            accel,
            accel_ts.t,
            idxs,
            mode=self.param(ws, "mode"),
            smooth_bias_s=float(self.param(ws, "smooth_bias_s")),
        )

        print(
            "ZV accel correction:",
            f"mode={self.param(ws, 'mode')}",
            f"zv={len(idxs)}",
            f"bias_std={stats['bias_std']:.3f}",
            f"bias_abs_p95={stats['bias_abs_p95']:.3f}",
        )

        ws[self.outputs[0]] = TimeSeries(
            t=accel_ts.t,
            x=corrected,
            units=accel_ts.units,
            frame=accel_ts.frame,
            meta={
                **accel_ts.meta,
                "zv_accel_correction": {
                    **stats,
                    "mode": self.param(ws, "mode"),
                    "min_prominence_mg": float(self.param(ws, "min_prominence_mg")),
                    "min_separation_s": float(self.param(ws, "min_separation_s")),
                    "smooth_bias_s": float(self.param(ws, "smooth_bias_s")),
                },
            },
        )
        if len(self.outputs) > 1:
            ws[self.outputs[1]] = idxs
