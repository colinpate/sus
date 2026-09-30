# Rear ZV Acceleration Correction

Logs:

- `log148_rear`
- `log149_rear`
- `log150_rear`
- `log151_rear`
- `log152_rear`
- `log153_rear`
- `log154_rear`

## Method

For each acceleration signal, integrate projected acceleration once, sample the raw integrated velocity at mag ZV points, interpolate that drift velocity, and subtract its derivative from the full acceleration series.

- Chunk cap during this run: `none`.
- Solver evaluated: `False`.

## Main Findings

- Best score in this run: `zv_smooth_all` with mean accel RMSE `22.080 m/s^2` and mean mag bin RMSE `7.339 mm`.
- The all-extrema linear velocity-drift correction is the most important baseline because it uses exactly the same ZV evidence as `centered_zv`, but applies it to the whole acceleration series before any downstream integration.
- Prominence/separation filters trade off noisy anchor removal against losing real small-amplitude turning points. Compare `mean_zv_points`, `mean_accel_rmse`, and bin RMSE together rather than optimizing only one metric.

## Aggregate Metrics

| Variant | Mode | Chunking | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Chunks | ZV | Bias p95 | Score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zv_smooth_all` | `smoothed_bias` | `centered_zv` | 22.080 | 0.805 | 6.445 | 7.339 | 12.575 | nan | nan | 8369 | 19518 | 5.363 | 8.001 |
| `raw_zv_s80ms` | `raw` | `centered_zv` | 22.645 | 0.796 | 5.351 | 8.375 | 15.300 | nan | nan | 3638 | 7502 | 0.000 | 9.055 |
| `zv_linear_p200_s50ms` | `linear_velocity` | `centered_zv` | 21.764 | 0.809 | 8.258 | 8.912 | 14.579 | nan | nan | 2043 | 2328 | 7.997 | 9.565 |
| `raw_zv_p200_s50ms` | `raw` | `centered_zv` | 22.645 | 0.796 | 5.047 | 8.971 | 17.181 | nan | nan | 1828 | 2328 | 0.000 | 9.650 |
| `raw_centered` | `raw` | `centered_zv` | 22.645 | 0.796 | 4.716 | 9.222 | 18.328 | nan | nan | 8309 | 19518 | 0.000 | 9.901 |
| `zv_linear_s80ms` | `linear_velocity` | `centered_zv` | 21.705 | 0.809 | 8.733 | 9.429 | 15.009 | nan | nan | 3781 | 7502 | 9.873 | 10.080 |

## Selected Per-Log Metrics

| Log | Variant | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Chunks | ZV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `log148_rear` | `zv_smooth_all` | 21.842 | 0.805 | 7.451 | 7.407 | 12.647 | nan | nan | 8542 | 16086 |
| `log148_rear` | `raw_centered` | 22.395 | 0.795 | 4.854 | 8.927 | 15.759 | nan | nan | 8481 | 16086 |
| `log149_rear` | `zv_smooth_all` | 19.418 | 0.801 | 9.908 | 9.589 | 14.063 | nan | nan | 8927 | 22041 |
| `log149_rear` | `raw_centered` | 19.966 | 0.791 | 6.490 | 10.702 | 19.440 | nan | nan | 8937 | 22041 |
| `log150_rear` | `zv_smooth_all` | 19.941 | 0.787 | 6.775 | 7.806 | 11.838 | nan | nan | 9864 | 17231 |
| `log150_rear` | `raw_centered` | 20.492 | 0.777 | 5.364 | 9.937 | 17.456 | nan | nan | 9797 | 17231 |
| `log151_rear` | `zv_smooth_all` | 23.315 | 0.802 | 6.115 | 6.715 | 9.921 | nan | nan | 8062 | 26430 |
| `log151_rear` | `raw_centered` | 24.002 | 0.791 | 4.035 | 9.778 | 18.122 | nan | nan | 7953 | 26430 |
| `log152_rear` | `zv_smooth_all` | 22.435 | 0.795 | 5.970 | 6.930 | 11.336 | nan | nan | 8956 | 19629 |
| `log152_rear` | `raw_centered` | 22.730 | 0.789 | 4.061 | 6.557 | 10.867 | nan | nan | 8909 | 19629 |
| `log153_rear` | `zv_smooth_all` | 22.983 | 0.807 | 4.720 | 6.681 | 9.788 | nan | nan | 8547 | 18812 |
| `log153_rear` | `raw_centered` | 23.367 | 0.800 | 3.723 | 9.376 | 17.795 | nan | nan | 8475 | 18812 |
| `log154_rear` | `zv_smooth_all` | 24.629 | 0.841 | 4.174 | 6.242 | 18.428 | nan | nan | 5687 | 16397 |
| `log154_rear` | `raw_centered` | 25.560 | 0.830 | 4.485 | 9.277 | 28.854 | nan | nan | 5610 | 16397 |