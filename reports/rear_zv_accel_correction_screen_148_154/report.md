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

- Chunk cap during this run: `2500`.
- Solver evaluated: `False`.

## Main Findings

- Best score in this run: `raw_zv_s80ms` with mean accel RMSE `22.645 m/s^2` and mean mag bin RMSE `8.424 mm`.
- The all-extrema linear velocity-drift correction is the most important baseline because it uses exactly the same ZV evidence as `centered_zv`, but applies it to the whole acceleration series before any downstream integration.
- Prominence/separation filters trade off noisy anchor removal against losing real small-amplitude turning points. Compare `mean_zv_points`, `mean_accel_rmse`, and bin RMSE together rather than optimizing only one metric.

## Aggregate Metrics

| Variant | Mode | Chunking | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Chunks | ZV | Bias p95 | Score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_zv_s80ms` | `raw` | `centered_zv` | 22.645 | 0.796 | 5.428 | 8.424 | 15.610 | nan | nan | 2496 | 7502 | 0.000 | 9.104 |
| `zv_linear_p200_s50ms` | `linear_velocity` | `centered_zv` | 21.764 | 0.809 | 8.258 | 8.912 | 14.579 | nan | nan | 2043 | 2328 | 7.997 | 9.565 |
| `raw_zv_p200_s50ms` | `raw` | `centered_zv` | 22.645 | 0.796 | 5.047 | 8.971 | 17.181 | nan | nan | 1828 | 2328 | 0.000 | 9.650 |
| `zv_linear_s80ms` | `linear_velocity` | `centered_zv` | 21.705 | 0.809 | 8.750 | 9.447 | 14.960 | nan | nan | 2500 | 7502 | 9.873 | 10.098 |
| `raw_centered` | `raw` | `centered_zv` | 22.645 | 0.796 | 4.583 | 9.688 | 19.303 | nan | nan | 2500 | 19518 | 0.000 | 10.367 |

## Selected Per-Log Metrics

| Log | Variant | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Chunks | ZV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `log148_rear` | `raw_zv_s80ms` | 22.395 | 0.795 | 5.162 | 8.660 | 14.919 | nan | nan | 2500 | 6395 |
| `log148_rear` | `raw_centered` | 22.395 | 0.795 | 4.497 | 9.603 | 17.425 | nan | nan | 2500 | 16086 |
| `log149_rear` | `raw_zv_s80ms` | 19.966 | 0.791 | 9.750 | 9.598 | 13.911 | nan | nan | 2500 | 8597 |
| `log149_rear` | `raw_centered` | 19.966 | 0.791 | 5.532 | 13.742 | 26.138 | nan | nan | 2500 | 22041 |
| `log150_rear` | `raw_zv_s80ms` | 20.492 | 0.777 | 6.006 | 8.805 | 13.774 | nan | nan | 2500 | 7473 |
| `log150_rear` | `raw_centered` | 20.492 | 0.777 | 5.946 | 8.613 | 13.380 | nan | nan | 2500 | 17231 |
| `log151_rear` | `raw_zv_s80ms` | 24.002 | 0.791 | 5.279 | 7.689 | 12.557 | nan | nan | 2500 | 9523 |
| `log151_rear` | `raw_centered` | 24.002 | 0.791 | 3.992 | 10.293 | 19.170 | nan | nan | 2500 | 26430 |
| `log152_rear` | `raw_zv_s80ms` | 22.730 | 0.789 | 4.031 | 6.735 | 11.380 | nan | nan | 2500 | 7362 |
| `log152_rear` | `raw_centered` | 22.730 | 0.789 | 3.782 | 7.044 | 12.502 | nan | nan | 2500 | 19629 |
| `log153_rear` | `raw_zv_s80ms` | 23.367 | 0.800 | 3.692 | 10.082 | 19.268 | nan | nan | 2500 | 7012 |
| `log153_rear` | `raw_centered` | 23.367 | 0.800 | 3.747 | 8.965 | 16.905 | nan | nan | 2500 | 18812 |
| `log154_rear` | `raw_zv_s80ms` | 25.560 | 0.830 | 4.078 | 7.400 | 23.458 | nan | nan | 2473 | 6154 |
| `log154_rear` | `raw_centered` | 25.560 | 0.830 | 4.582 | 9.555 | 29.600 | nan | nan | 2500 | 16397 |