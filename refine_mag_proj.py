import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def fit_curve(mag, travel):
    def gt_res(vec):
        x0, y_scale, power, y0 = vec[0], vec[1], vec[2], vec[3]
        return pred_x(mag, x0, y_scale, power=power, y0=y0) - travel

    guess_vec = [0, 5, 1/3, 0]
    gt_result = least_squares(
                    fun=gt_res,
                    x0=guess_vec, 
                    method="trf",
                    verbose=1,
                    max_nfev=1000,
                    #loss='huber',
                )

    gt_coeffs = gt_result.x
    gt_preds = pred_x(mag, gt_coeffs[0], gt_coeffs[1], gt_coeffs[2], gt_coeffs[3])
    return gt_coeffs, gt_preds


def pred_x(mag_i, x0, y_scale, power, y0=0):
    mag_i = np.copy(mag_i)
    mag_i[(mag_i - x0) < 0] = x0
    return ((mag_i - x0) ** power) * y_scale + y0


def get_mag_vector(mag, min_mag, max_mag, normalize=False):
    # Threshold magnet data and project along the direction of travel
    mag_mag = np.linalg.norm(mag, axis=1)
    thresh_mask = (mag_mag >= min_mag) & (mag_mag < max_mag)
    mag_filt = mag[thresh_mask]
    if normalize:
        mag_filt /= np.linalg.norm(mag_filt, axis=1)
    mean_vector = np.mean(mag_filt, axis=0)
    mag_travel_vector = mean_vector / np.linalg.norm(mean_vector)
    return mag_travel_vector


def project_mag(mag, mag_travel_vector):
    # Project mag data onto mean vector to get travel along that direction
    return mag @ mag_travel_vector


def summarize_log(
    log_name: str,
    cache_root: Path,
) -> None:
    cache_path = cache_root / log_name / "cache" / "all.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    cache = np.load(cache_path)

    mag_lpf = cache["mag/lpf__x"]
    travel = cache["travel__x"]
    boring_mask = cache["boring_mask"].astype(bool)

    

    mag_min = 



