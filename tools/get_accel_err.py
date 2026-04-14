import numpy as np

log_filename = "log079"

out_dir = f"backend/run_artifacts/{log_filename}/cache/"
ws_file = out_dir + "/all.npz"
ws = np.load(ws_file)
t = ws["accel/lpf/proj__t"]
travel = ws["travel__x"][:, 0]
a_proj = ws["accel/lpfhp/proj__x"][:, 0]
dt_s = np.diff(t, prepend=t[0]-0.01)
v = np.diff(travel, prepend=travel[0]) / dt_s
a_gt = np.diff(v, prepend=v[0]) / dt_s

# Histogram of accelerometer error vs magnitude
n_bins = 8

a_gt_ms = a_gt / 1000
start = np.percentile(a_gt_ms, 0.01)
stop = np.percentile(a_gt_ms, 99.99)
bin_size = (stop - start) / n_bins
bin_centers = [(i + 0.5) * bin_size + start for i in range(n_bins)]

bin_means = []
bin_stds = []
bin_rmses = []
for i in range(n_bins):
    bin_min = (i * bin_size) + start
    bin_max = bin_min + bin_size
    mask = (a_gt_ms >= bin_min) * (a_gt_ms < bin_max)
    bin_err = a_proj[mask] - a_gt_ms[mask]
    bin_means.append(np.mean(bin_err))
    bin_stds.append(np.std(bin_err))
    bin_rmses.append(np.sqrt(np.mean(np.square(bin_err))))
print("Bin centers:", bin_centers)
print("Bin means:", bin_means)
print("Bin stds:", bin_stds)
print("Bin RMSEs:", bin_rmses)