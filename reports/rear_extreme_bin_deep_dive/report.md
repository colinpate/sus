# Rear Mag Model Extreme-Bin Deep Dive

This analysis compares mag-model errors against raw/cache-level signals and reconstructed training chunks.

Data note: I regenerated a self-consistent stats bundle from the current caches at
`experiments/legacy_stats/rear/rear_extreme_bin_current_stats` because the earlier
legacy tables did not match the current cache contents for at least `log140_rear`.

## Headline

- High-travel failures are mostly endpoint/tail failures: very little time above 120 mm, very few filtered training chunks centered there, weak local mag/travel correlation in several logs, and a systematic high-travel underprediction.
- Low-travel failures are less about missing samples and more about the learned zero-end shape: bad low bins usually have plenty of samples, but the selected chunks and fitted curve create a signed offset at the bottom of travel.
- The solver only weakly changes these patterns, so the mag-model learning/coverage problem is visible before solver fusion.
- A gotcha: the aggregate `bin_rmse` excludes bins with fewer than 100 samples. That means `log144_rear`, `log154_rear`, and `log142_rear` can have ugly 120-150 mm tails without fully paying for them in the overall bin score.

## Interpretation

- The 120-150 mm bin is sparse everywhere: 59-514 samples and only 6-42 filtered chunks depending on log. Once everything is this sparse, sample count alone does not rank the failures well, but the support level is still low enough that the fitted power-law curve is mostly extrapolating the endpoint.
- Every high-travel bin has negative signed error, so the model is consistently underpredicting the top of travel after centering. The worst cases are not random scatter; they are biased low at the tail.
- `log144_rear` is the clearest oddball: only 59 high-bin samples, 6 filtered high-bin chunks, and an empirical high-bin `dx/dmag` near zero, so the model/empirical slope ratio explodes to `96x`.
- `log154_rear` is the split-personality log: it is the best low-travel log (`3.06 mm` at 0-30), but one of the worst high-travel logs (`15.36 mm` at 120-150) because the high end has only 87 samples and 6 chunks.
- Low-travel error is much more diagnostic. The Spearman correlation between low-bin RMSE and model/empirical slope ratio is `0.764`; the bad low logs tend to fit a bottom-end curve that is too steep relative to the local mag/travel cloud.
- Chunk survival is not the whole story. Low-bin chunk survival ranges from about 9-32%, but the logs with poor low-bin RMSE are differentiated more by slope mismatch and signed bias than by raw chunk count.

## Curve Plots

- Combined high-travel best/worst plot: `curve_plots/high_travel_best_worst_curves.png`
- Individual plots live in `curve_plots/`.
- Worst high-travel logs plotted: `log144_rear`, `log141_rear`, `log140_rear`.
- Best high-travel logs plotted: `log151_rear`, `log152_rear`, `log150_rear`.

## Worst High-Travel Excess

| Log | Overall bin RMSE | 120-150 RMSE | Excess | Samples | Filtered chunks | Signed err | Slope ratio | Chunk survival |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `log144_rear` | 6.06 | 19.16 | 13.10 | 59 | 6 | -15.96 | 96.12 | 85.71% |
| `log154_rear` | 5.80 | 15.36 | 9.56 | 87 | 6 | -14.00 | 2.83 | 85.71% |
| `log141_rear` | 10.24 | 18.40 | 8.17 | 200 | 22 | -17.66 | 2.05 | 95.65% |
| `log140_rear` | 9.26 | 16.96 | 7.70 | 514 | 42 | -16.53 | 1.63 | 89.36% |
| `log153_rear` | 8.49 | 15.86 | 7.38 | 326 | 28 | -15.14 | 3.17 | 100.00% |
| `log142_rear` | 5.37 | 11.72 | 6.35 | 97 | 8 | -11.12 | 1.00 | 88.89% |

## Worst Low-Travel Excess

| Log | Overall bin RMSE | 0-30 RMSE | Excess | Samples | Filtered chunks | Signed err | Slope ratio | Chunk survival |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `log152_rear` | 5.75 | 7.91 | 2.16 | 16441 | 1170 | -6.30 | 1.60 | 21.67% |
| `log149_rear` | 8.37 | 10.17 | 1.80 | 50208 | 1130 | -7.46 | 1.58 | 8.91% |
| `log142_rear` | 5.37 | 7.08 | 1.71 | 22911 | 946 | -4.74 | 1.53 | 8.83% |
| `log148_rear` | 7.60 | 8.87 | 1.27 | 32907 | 1154 | -6.89 | 1.45 | 22.13% |
| `log150_rear` | 7.65 | 8.79 | 1.14 | 21508 | 964 | -5.16 | 1.69 | 17.22% |
| `log151_rear` | 5.45 | 6.26 | 0.81 | 29883 | 1284 | -5.06 | 1.25 | 8.36% |

## Best Overall Logs

| Log | Overall bin RMSE | 0-30 | 120-150 | Low chunks | High chunks | Power | y_scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `log142_rear` | 5.37 | 7.08 | 11.72 | 946 | 8 | 0.296 | -55.83 |
| `log151_rear` | 5.45 | 6.26 | 7.51 | 1284 | 23 | 0.246 | -75.79 |
| `log152_rear` | 5.75 | 7.91 | 7.75 | 1170 | 17 | 0.291 | -67.15 |
| `log154_rear` | 5.80 | 3.06 | 15.36 | 929 | 6 | 0.254 | -52.81 |
| `log144_rear` | 6.06 | 6.25 | 19.16 | 371 | 6 | 0.308 | -39.59 |
| `log148_rear` | 7.60 | 8.87 | 12.04 | 1154 | 21 | 0.312 | -49.37 |

## Worst Overall Logs

| Log | Overall bin RMSE | 0-30 | 120-150 | Low chunks | High chunks | Power | y_scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `log141_rear` | 10.24 | 5.23 | 18.40 | 823 | 22 | 0.330 | -81.11 |
| `log140_rear` | 9.26 | 4.53 | 16.96 | 1305 | 42 | 0.269 | -75.00 |
| `log145_rear` | 9.03 | 8.88 | 14.84 | 973 | 28 | 0.322 | -74.30 |
| `log153_rear` | 8.49 | 5.28 | 15.86 | 1454 | 28 | 0.300 | -61.95 |
| `log143_rear` | 8.44 | 4.76 | 14.46 | 861 | 28 | 0.275 | -67.73 |
| `log149_rear` | 8.37 | 10.17 | 11.24 | 1130 | 26 | 0.270 | -64.58 |

## Correlation Clues

- `high bin` Spearman `filtered_chunk_count` vs `rmse`: `0.083`
- `high bin` Spearman `sample_count` vs `rmse`: `0.000`
- `high bin` Spearman `model_to_empirical_slope_ratio` vs `rmse`: `0.170`
- `high bin` Spearman `chunk_survival_pct` vs `rmse`: `-0.210`
- `low bin` Spearman `filtered_chunk_count` vs `rmse`: `0.181`
- `low bin` Spearman `sample_count` vs `rmse`: `0.352`
- `low bin` Spearman `model_to_empirical_slope_ratio` vs `rmse`: `0.764`
- `low bin` Spearman `chunk_survival_pct` vs `rmse`: `0.044`

## Notes

- `Slope ratio` is learned model `dx/dmag` divided by empirical local `dx/dmag`. Values above 1 mean the learned curve is steeper than the observed mag/travel cloud in that bin.
- `Signed err` is centered prediction error in the bin. Positive means predicted travel is high relative to GT after centering; negative means low.
- `Filtered chunks` counts chunks centered in that travel bin after the same `RearMagModel` dx filter used by the pipeline.

Full tables:

- `log_summary.csv`
- `bin_metrics.csv`
