# Rear Slope-Guided Method Analysis

Logs used:

- `log149_rear`
- `log150_rear`
- `log151_rear`
- `log152_rear`
- `log153_rear`

## Main Findings

- None of the slope-guided methods produced a meaningful RMSE win over the current rear fit. Best mean masked aligned RMSE was `bin_prior_w10` at `5.329 mm`, versus baseline `5.329 mm`.
- The only objective that moved the curve materially was the strong absolute bin prior `bin_prior_w100`: it improved mean GT bin-slope MAE from `23.945` to `18.420 mG/mm`, but it also worsened mean RMSE to `6.296 mm`.
- The RANSAC-like consensus objectives and the shape-only priors were essentially no-ops. Their mean first-last `|dmag/dx|` ratio stayed at about `1.119`, the same as baseline.
- That points to a specific limitation: the chunk slope proxy seems to carry some trend information, but not enough leverage to teach the solver appreciably more curvature inside the current model family.

## Mean Metrics

| Method | Mean masked aligned RMSE (mm) | Mean GT bin-slope MAE (mG/mm) | Mean first-last |dmag/dx| ratio |
|---|---:|---:|---:|
| `bin_prior_w10` | 5.329 | 23.842 | 1.119 |
| `bin_shape_prior_w100` | 5.329 | 23.945 | 1.119 |
| `bin_shape_prior_w30` | 5.329 | 23.945 | 1.119 |
| `bin_shape_prior_w10` | 5.329 | 23.945 | 1.119 |
| `baseline` | 5.329 | 23.945 | 1.119 |
| `raw_proxy_consensus_w0.2` | 5.329 | 23.945 | 1.119 |
| `bin_interp_consensus_w0.2` | 5.329 | 23.945 | 1.119 |
| `local_median_consensus_w0.2` | 5.329 | 23.945 | 1.119 |
| `raw_proxy_consensus_w0.5` | 5.329 | 23.945 | 1.119 |
| `bin_interp_consensus_w0.5` | 5.329 | 23.945 | 1.119 |
| `local_median_consensus_w0.5` | 5.329 | 23.945 | 1.119 |
| `local_median_consensus_w1.0` | 5.330 | 23.943 | 1.119 |
| `bin_interp_consensus_w1.0` | 5.330 | 23.943 | 1.119 |
| `bin_prior_w30` | 5.341 | 23.093 | 1.119 |
| `bin_prior_w100` | 6.296 | 18.420 | 1.119 |

## Per-Log RMSE

| Log | `baseline` | `bin_prior_w10` | `bin_prior_w30` | `bin_prior_w100` | `bin_shape_prior_w10` | `bin_shape_prior_w30` | `bin_shape_prior_w100` | `bin_interp_consensus_w0.2` | `bin_interp_consensus_w0.5` | `bin_interp_consensus_w1.0` | `local_median_consensus_w0.2` | `local_median_consensus_w0.5` | `local_median_consensus_w1.0` | `raw_proxy_consensus_w0.2` | `raw_proxy_consensus_w0.5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `log149_rear` | 6.115 | 6.140 | 6.343 | 8.411 | 6.115 | 6.115 | 6.114 | 6.115 | 6.115 | 6.116 | 6.115 | 6.115 | 6.116 | 6.115 | 6.115 |
| `log150_rear` | 4.886 | 4.891 | 4.930 | 5.553 | 4.886 | 4.886 | 4.885 | 4.886 | 4.886 | 4.886 | 4.886 | 4.886 | 4.886 | 4.886 | 4.886 |
| `log151_rear` | 5.130 | 5.160 | 5.396 | 7.347 | 5.130 | 5.130 | 5.128 | 5.130 | 5.130 | 5.130 | 5.130 | 5.130 | 5.130 | 5.130 | 5.130 |
| `log152_rear` | 4.760 | 4.759 | 4.764 | 5.420 | 4.760 | 4.760 | 4.760 | 4.760 | 4.760 | 4.760 | 4.760 | 4.760 | 4.760 | 4.760 | 4.760 |
| `log153_rear` | 5.755 | 5.693 | 5.273 | 4.750 | 5.755 | 5.755 | 5.756 | 5.755 | 5.755 | 5.755 | 5.755 | 5.755 | 5.755 | 5.755 | 5.755 |

## Per-Log Slope MAE

| Log | `baseline` | `bin_prior_w10` | `bin_prior_w30` | `bin_prior_w100` | `bin_shape_prior_w10` | `bin_shape_prior_w30` | `bin_shape_prior_w100` | `bin_interp_consensus_w0.2` | `bin_interp_consensus_w0.5` | `bin_interp_consensus_w1.0` | `local_median_consensus_w0.2` | `local_median_consensus_w0.5` | `local_median_consensus_w1.0` | `raw_proxy_consensus_w0.2` | `raw_proxy_consensus_w0.5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `log149_rear` | 28.931 | 28.832 | 28.099 | 23.385 | 28.931 | 28.931 | 28.931 | 28.931 | 28.930 | 28.929 | 28.931 | 28.930 | 28.929 | 28.931 | 28.930 |
| `log150_rear` | 28.010 | 27.931 | 27.336 | 23.247 | 28.010 | 28.010 | 28.010 | 28.010 | 28.010 | 28.008 | 28.010 | 28.010 | 28.008 | 28.010 | 28.010 |
| `log151_rear` | 18.592 | 18.527 | 18.038 | 14.512 | 18.592 | 18.592 | 18.592 | 18.591 | 18.591 | 18.590 | 18.591 | 18.591 | 18.590 | 18.591 | 18.591 |
| `log152_rear` | 22.041 | 21.944 | 21.233 | 16.712 | 22.041 | 22.041 | 22.042 | 22.041 | 22.040 | 22.039 | 22.041 | 22.040 | 22.039 | 22.041 | 22.040 |
| `log153_rear` | 22.152 | 21.979 | 20.760 | 14.247 | 22.152 | 22.152 | 22.152 | 22.152 | 22.151 | 22.150 | 22.152 | 22.151 | 22.150 | 22.152 | 22.151 |