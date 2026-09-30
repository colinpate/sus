# Rear Mag Model Centering Effects

This compares constant-offset choices for evaluating `travel/mag_model/adj` against GT travel.

## Offset Methods

- `sample_mean`: current centered metric; offset minimizes sample-weighted RMSE over the boring mask.
- `sample_median`: robust global offset.
- `bin_mean_eligible`: offset minimizes equal-bin RMSE over bins with at least 100 samples.
- `bin_mean_all`: offset minimizes equal-bin RMSE over every observed bin, including sparse tails.
- `tail_mean`: offset averages the 0-30 and 120-150 bin residual means.

## Aggregate Metrics

| Method | Mean sample RMSE | Mean eligible-bin RMSE | Mean all-bin RMSE | Mean 0-30 RMSE | Mean 120-150 RMSE |
|---|---:|---:|---:|---:|---:|
| `sample_mean` | 5.18 | 6.50 | 7.15 | 7.00 | 10.31 |
| `sample_median` | 5.21 | 6.58 | 7.24 | 7.22 | 10.47 |
| `bin_mean_eligible` | 5.62 | 6.15 | 6.70 | 6.31 | 8.80 |
| `bin_mean_all` | 5.94 | 6.27 | 6.61 | 6.37 | 8.34 |
| `tail_mean` | 8.35 | 7.74 | 7.75 | 6.20 | 6.60 |

## Biggest High-Tail Changes Under All-Bin Offset

| Log | Current 120-150 | All-bin 120-150 | Delta | Current 0-30 | All-bin 0-30 | Offset delta |
|---|---:|---:|---:|---:|---:|---:|
| `log144_rear` | 21.34 | 16.64 | -4.70 | 6.34 | 4.78 | +5.59 |
| `log154_rear` | 16.07 | 11.76 | -4.30 | 3.33 | 5.28 | +4.79 |
| `log141_rear` | 15.33 | 11.43 | -3.90 | 5.10 | 5.13 | +4.28 |
| `log143_rear` | 10.41 | 8.02 | -2.38 | 4.91 | 4.55 | +2.98 |
| `log153_rear` | 10.48 | 8.28 | -2.21 | 6.21 | 4.68 | +2.66 |
| `log142_rear` | 8.26 | 6.14 | -2.11 | 7.56 | 6.07 | +2.48 |
| `log145_rear` | 11.22 | 9.41 | -1.81 | 9.71 | 8.57 | +2.63 |
| `log140_rear` | 7.91 | 6.63 | -1.29 | 7.08 | 5.94 | +1.59 |

## Bias Decomposition Under Current Offset

| Log | Bin | RMSE | Mean bias | Within-bin RMSE | Bias fraction |
|---|---|---:|---:|---:|---:|
| `log141_rear` | `120-150` | 15.33 | -14.34 | 5.41 | 88% |
| `log154_rear` | `120-150` | 16.07 | -14.89 | 6.03 | 86% |
| `log142_rear` | `120-150` | 8.26 | -7.38 | 3.71 | 80% |
| `log144_rear` | `120-150` | 21.34 | -18.76 | 10.17 | 77% |
| `log153_rear` | `120-150` | 10.48 | -9.11 | 5.19 | 76% |
| `log143_rear` | `120-150` | 10.41 | -8.85 | 5.47 | 72% |
| `log140_rear` | `120-150` | 7.91 | -6.67 | 4.25 | 71% |
| `log152_rear` | `0-30` | 8.55 | -7.08 | 4.79 | 69% |
| `log148_rear` | `0-30` | 8.33 | -6.69 | 4.97 | 64% |
| `log151_rear` | `0-30` | 5.68 | -4.56 | 3.39 | 64% |
| `log140_rear` | `0-30` | 7.08 | -5.48 | 4.49 | 60% |
| `log145_rear` | `120-150` | 11.22 | -8.41 | 7.42 | 56% |
| `log149_rear` | `0-30` | 9.96 | -7.43 | 6.63 | 56% |
| `log151_rear` | `120-150` | 6.34 | -4.65 | 4.32 | 54% |

## Takeaway

The current centering is the optimal single offset for sample-weighted RMSE, but it is not aligned with an equal-bin metric. Dense mid-travel samples dominate the offset, so sparse tails can inherit a large bin bias.
For shape evaluation independent of Y-offset, use an offset fitted with the same weighting as the metric: `bin_mean_eligible` for the current eligible-bin score, or `bin_mean_all` / a separate tail metric when sparse endpoints matter.

Full tables:

- `centering_metrics.csv`
- `bin_bias_decomposition.csv`