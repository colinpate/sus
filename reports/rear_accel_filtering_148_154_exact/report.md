# Rear Accel Filtering Sweep

Logs:

- `log148_rear`
- `log149_rear`
- `log150_rear`
- `log151_rear`
- `log152_rear`
- `log153_rear`
- `log154_rear`

## Method

Each variant recomputes the rear LIS2 lowpass/highpass, re-estimates the rear travel axis from high-accel samples, projects to 1D, applies the current smoothed-bias mag-ZV correction, then retrains the rear mag model with `centered_zv` chunks and `x0_weight=0.0`.

- Chunk cap during this run: `999999`.
- Solver evaluated: `False`.

## Main Findings

- Best score in this run: `hpf4_o2_s50` with mean mag bin RMSE `6.142 mm` and mean accel RMSE `21.478 m/s^2`.
- Current 2 Hz HPF reference: mean mag bin RMSE `6.742 mm`, mean accel RMSE `22.080 m/s^2`.

## Aggregate Metrics

| Variant | LPF | HPF | Order | ZV smooth | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Axis Dot | Chunks | Bias p95 | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hpf4_o2_s50` | 40.0 | 4.00 | 2 | 50 ms | 21.478 | 0.809 | 5.037 | 6.142 | 9.717 | nan | nan | 1.000 | 8240 | 3.857 | 6.786 |
| `hpf2_o1_s50` | 40.0 | 2.00 | 1 | 50 ms | 21.397 | 0.808 | 5.240 | 6.394 | 10.213 | nan | nan | 1.000 | 8307 | 5.123 | 7.036 |
| `hpf3_o2_s50` | 40.0 | 3.00 | 2 | 50 ms | 21.856 | 0.807 | 5.613 | 6.409 | 9.997 | nan | nan | 1.000 | 8313 | 4.406 | 7.065 |
| `hpf2p5_o2_s50` | 40.0 | 2.50 | 2 | 50 ms | 21.987 | 0.806 | 5.836 | 6.616 | 10.237 | nan | nan | 1.000 | 8349 | 4.817 | 7.276 |
| `hpf1_o2_s50` | 40.0 | 1.00 | 2 | 50 ms | 22.180 | 0.804 | 5.886 | 6.735 | 11.026 | nan | nan | 1.000 | 8380 | 6.799 | 7.400 |
| `current_lpf40_hpf2_o2_s50` | 40.0 | 2.00 | 2 | 50 ms | 22.080 | 0.805 | 5.953 | 6.742 | 10.608 | nan | nan | 1.000 | 8369 | 5.363 | 7.405 |
| `hpf0p75_o2_s50` | 40.0 | 0.75 | 2 | 50 ms | 22.192 | 0.804 | 5.849 | 6.760 | 11.107 | nan | nan | 1.000 | 8381 | 7.186 | 7.426 |
| `lpf20_hpf1_o2_s50` | 20.0 | 1.00 | 2 | 50 ms | 21.552 | 0.646 | 5.234 | 7.166 | 12.166 | nan | nan | 1.000 | 8406 | 6.199 | 7.813 |

## Selected Per-Log Metrics

| Log | Variant | Accel RMSE | Accel Corr | Mag RMSE | Mag Bin | Mag Worst | Solver RMSE | Solver Bin | Axis Dot | Chunks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `log148_rear` | `hpf4_o2_s50` | 21.223 | 0.809 | 5.191 | 5.752 | 8.334 | nan | nan | 1.000 | 8420 |
| `log148_rear` | `current_lpf40_hpf2_o2_s50` | 21.842 | 0.805 | 6.820 | 6.758 | 11.394 | nan | nan | 1.000 | 8542 |
| `log148_rear` | `hpf1_o2_s50` | 21.947 | 0.803 | 6.897 | 6.975 | 11.606 | nan | nan | 1.000 | 8568 |
| `log149_rear` | `hpf4_o2_s50` | 18.842 | 0.805 | 7.712 | 8.109 | 9.960 | nan | nan | 1.000 | 8704 |
| `log149_rear` | `current_lpf40_hpf2_o2_s50` | 19.418 | 0.801 | 8.875 | 9.233 | 11.950 | nan | nan | 1.000 | 8927 |
| `log149_rear` | `hpf1_o2_s50` | 19.522 | 0.800 | 8.733 | 8.790 | 11.871 | nan | nan | 1.000 | 8960 |
| `log150_rear` | `hpf4_o2_s50` | 19.314 | 0.791 | 5.726 | 6.257 | 8.286 | nan | nan | 1.000 | 9672 |
| `log150_rear` | `current_lpf40_hpf2_o2_s50` | 19.941 | 0.787 | 6.436 | 6.874 | 10.394 | nan | nan | 1.000 | 9864 |
| `log150_rear` | `hpf1_o2_s50` | 20.038 | 0.786 | 6.452 | 7.077 | 10.644 | nan | nan | 1.000 | 9887 |
| `log151_rear` | `hpf4_o2_s50` | 22.611 | 0.806 | 4.140 | 4.989 | 6.344 | nan | nan | 1.000 | 7952 |
| `log151_rear` | `current_lpf40_hpf2_o2_s50` | 23.315 | 0.802 | 5.179 | 5.488 | 7.849 | nan | nan | 1.000 | 8062 |
| `log151_rear` | `hpf1_o2_s50` | 23.423 | 0.800 | 5.138 | 5.487 | 7.839 | nan | nan | 1.000 | 8070 |
| `log152_rear` | `hpf4_o2_s50` | 21.989 | 0.797 | 4.804 | 5.669 | 8.547 | nan | nan | 1.000 | 8835 |
| `log152_rear` | `current_lpf40_hpf2_o2_s50` | 22.435 | 0.795 | 5.860 | 7.137 | 10.737 | nan | nan | 1.000 | 8956 |
| `log152_rear` | `hpf1_o2_s50` | 22.516 | 0.794 | 5.571 | 6.517 | 10.350 | nan | nan | 1.000 | 8951 |
| `log153_rear` | `hpf4_o2_s50` | 22.507 | 0.810 | 3.926 | 6.364 | 10.483 | nan | nan | 1.000 | 8449 |
| `log153_rear` | `current_lpf40_hpf2_o2_s50` | 22.983 | 0.807 | 4.639 | 6.406 | 9.044 | nan | nan | 1.000 | 8547 |
| `log153_rear` | `hpf1_o2_s50` | 23.075 | 0.806 | 4.527 | 6.811 | 10.641 | nan | nan | 1.000 | 8544 |
| `log154_rear` | `hpf4_o2_s50` | 23.858 | 0.845 | 3.757 | 5.854 | 16.069 | nan | nan | 1.000 | 5647 |
| `log154_rear` | `current_lpf40_hpf2_o2_s50` | 24.629 | 0.841 | 3.861 | 5.298 | 12.885 | nan | nan | 1.000 | 5687 |
| `log154_rear` | `hpf1_o2_s50` | 24.735 | 0.840 | 3.885 | 5.490 | 14.229 | nan | nan | 1.000 | 5678 |