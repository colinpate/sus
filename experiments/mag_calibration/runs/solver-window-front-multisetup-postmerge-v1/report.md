# solver-window-front-multisetup-postmerge-v1

Status: **90/90 successful** (0 failed).

## Design

- Front pipeline; 15 logs from three setup-balanced pod-v2 cohorts (Jamaal TR11, Harry TR11, and Slayer).
- 2 deterministic nested window(s) per log (source repeats 1, 2) at 10 s, 40 s, 60 s.
- The saved self-supervised curve is injected before the two full-log fusion solves.
- Scoring is applied afterward on the training window, a fixed centered common core, the full log, and the full log excluding training.
- Fixed-offset window metrics use the corresponding stage's full-log alignment, so window-local recentering cannot hide bias.

## Main result

On the full log, median raw magnetic-model RMSE changes from 9.19 to 7.43 mm; the paired 60-10 s change is -0.80 mm (60% of logs improve, exploratory Wilcoxon p=0.303).
For the final solved output, median full-log RMSE changes from 7.35 to 5.47 mm; the paired change is -0.75 mm (60% improve, p=0.035).

The suspicious low-travel trend is strongly attenuated downstream. On the varying training windows, raw 0–30 mm fixed-offset RMSE has a paired change of -0.01 mm from 10 to 60 s, while the final solved output changes by -0.90 mm. On the identical centered 10 s core, the corresponding changes are -1.02 and -1.14 mm. This supports the interpretation that fusion/correction removes much of the raw curve's apparent low-travel degradation rather than propagating it to final travel.

## Practical duration

## Median RMSE by duration

| Training | Full log: mag | Full log: solved | Training window: mag | Training window: solved | 0–30 mm training: mag | 0–30 mm training: solved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 s | 9.19 | 7.35 | 6.87 | 5.80 | 9.46 | 7.70 |
| 40 s | 7.47 | 5.42 | 6.96 | 5.30 | 11.03 | 6.72 |
| 60 s | 7.43 | 5.47 | 7.39 | 5.39 | 10.59 | 7.79 |

All values are millimetres. Repeats are first collapsed within each log, then logs are weighted equally. Negative paired changes mean lower error at the longest duration. The p-values are descriptive only: this run remains an exploratory rather than confirmatory significance study.

## Runtime and artifacts

Median runtime was 12.1 s per full-log condition (22.5 solver-minutes total).
Fresh runs write row-level metrics, aggregate tables, and the learning-curve figure beside this report. The frozen schedule links every solve to its exact source calibration and cached-input fingerprint; large execution caches need not be versioned.
