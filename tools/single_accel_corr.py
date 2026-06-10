from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy.signal import butter, sosfiltfilt
import scipy

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(".").resolve()))
    sys.path.insert(0, str(Path("backend/").resolve()))

from tools.analyze_accel_mismatch import derive_gt
from backend.fusion import print_err_stats


def print_err_stats(x, gt, center=False, prefix=""):
    if center:
        x = x.copy() - np.mean(x)
        gt = gt.copy() - np.mean(gt)
    error = x - gt
    rmse = np.mean(error ** 2) ** 0.5
    mae = np.mean(abs(error))
    me = np.mean(error)
    print(f"{prefix} RMSE: {rmse:.3f}, MAE: {mae:.3f}, ME: {me:.3f}")


def main():
    log_filename = "log104"
    out_dir = f"backend/run_artifacts/{log_filename}/cache/"
    ws_file = out_dir + "/all.npz"
    ws = np.load(ws_file)
    a = ws["accel/lpf/lis2__x"]
    b_proj = ws["mag/proj/lpf__x"][:, 0]
    b_raw = ws["mag/lpf__x"]
    gyro = ws["gyro/lpf/gyro1__x"]
    boring_mask = ws["boring_mask"]
    #t = ws["accel/lpf/lis2__t"]
    t, a_meas, a_gt, v = derive_gt(ws, use_gradient=True, use_raw=False)

    # Highpass accel
    hp_fc_hz=1
    sos_hp = butter(N=4, Wn=hp_fc_hz, btype="high", fs=100, output="sos")
    a_hp = sosfiltfilt(sos_hp, a, axis=0)
    a_hp_norm = np.linalg.norm(a_hp, axis=1)

    mask = (a_hp_norm > 10)
    # dilate mask
    mask = np.convolve(mask, np.ones(200), mode="same") > 0
    print(np.mean(mask))
    print(np.mean(boring_mask))
    print(np.mean(np.abs(mask ^ boring_mask)))
    mask = boring_mask

    a_hp_m = a_hp[mask]
    accel_axis = np.mean(a_hp_m[a_hp_m[:, 0] > 0], axis=0)
    accel_axis /= np.linalg.norm(accel_axis)
    print(accel_axis)
    print_err_stats(np.zeros_like(a_gt[mask]), a_gt[mask], center=True, prefix="Accel 0")
    print_err_stats(a_meas[mask], a_gt[mask], center=True, prefix="Accel proj")

    for fc in [0.5, 1, 2, 4, 8, 16]:
        sos_hp = butter(N=4, Wn=fc, btype="high", fs=100, output="sos")
        a_proj = a @ -accel_axis
        a_hp_proj = sosfiltfilt(sos_hp, a_proj, axis=0)

        print_err_stats(a_hp_proj[mask], a_gt[mask], center=True, prefix="Accel")

if __name__ == "__main__":
    main()