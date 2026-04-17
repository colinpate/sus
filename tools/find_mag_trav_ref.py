import os
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stats_aggregator import DEFAULT_LOGS, load_cache

run_pipeline = True
cache_root = Path("backend/run_artifacts/")
logs = DEFAULT_LOGS
#logs = ["log079", "log080"]

pred_soft_mg = 50
def pred_x(mag_i, x0, y_scale, power):
    dx = np.asarray(mag_i, dtype=float) - x0
    soft = (np.abs(dx) + pred_soft_mg) ** power - (pred_soft_mg ** power)
    return np.sign(dx) * soft * y_scale

def load_cache_keys(log_filename):
    cache = load_cache(log_filename, cache_root)
    mag_proj = cache["mag/proj/corr/lpf__x"][:, 0]
    coeffs = cache["mag_model_coeffs"]
    travel = cache["travel__x"][:, 0]
    err_mask = cache["boring_mask"]
    return mag_proj, coeffs, travel, err_mask

def main():
    zm_rows = []
    me_rows = []
    total_error = np.zeros((5,))
    for log_filename in logs:
        try:
            mag_proj, coeffs, travel, err_mask = load_cache_keys(log_filename)
        except:
            if run_pipeline:
                print(f"Running pipeline for {log_filename}...")
                os.system("venv/bin/python3 backend/pipeline.py " + log_filename)
            mag_proj, coeffs, travel, err_mask = load_cache_keys(log_filename)

        # Try different zeros
        zero_mags = []
        mean_errs = []
        for percentile in (4, 6, 8, 12, 16):
            zero_mag = np.percentile(mag_proj, percentile)
            zero_mag_pred_x = pred_x(zero_mag, coeffs[0], coeffs[1], coeffs[2])
            mag_proj_pred_x = pred_x(mag_proj, coeffs[0], coeffs[1], coeffs[2])
            mag_proj_pred_x -= zero_mag_pred_x
            mean_err = np.mean((mag_proj_pred_x - travel)[err_mask])
            zero_mags.append(zero_mag)
            mean_errs.append(mean_err)

        total_error += np.abs(np.array(mean_errs)**2)
        zm_row = "Zero mags:"
        me_row = "Mean errs:"
        for (zm, me) in zip(zero_mags, mean_errs):
            zm_row += f" {zm:>8.1f},"
            me_row += f" {me:>8.1f},"

        zm_rows.append(zm_row)
        me_rows.append(me_row)

    for (log, zm_row, me_row) in zip(logs, zm_rows, me_rows):
        print("Log", log)
        print(zm_row)
        print(me_row)

    print(np.sqrt(total_error/len(logs)))

if __name__ == "__main__":
    main()
