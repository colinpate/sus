# Phase 1 downstream solver window-length experiment

Status: **55/55 successful** (0 failed).

## Design

- Front pipeline; 11 Stumpjumper/pod-v2 logs.
- One deterministic nested window per log (source repeat 0) at 5 s, 10 s, 20 s, 40 s, 120 s.
- The saved self-supervised curve is injected before the two full-log fusion solves.
- Scoring is applied afterward on the training window, a fixed centered common core, the full log, and the full log excluding training.
- Fixed-offset window metrics use the corresponding stage's full-log alignment, so window-local recentering cannot hide bias.

## Main result

On the full log, median raw magnetic-model RMSE changes from 5.40 to 4.12 mm; the paired 120-5 s change is -0.90 mm (73% of logs improve, exploratory Wilcoxon p=0.102).
For the final solved output, median full-log RMSE changes from 5.19 to 3.28 mm; the paired change is -1.33 mm (82% improve, p=0.007).

The suspicious low-travel trend is strongly attenuated downstream. On the varying training windows, raw 0–30 mm fixed-offset RMSE has a paired change of 2.58 mm from 5 to 120 s, while the final solved output changes by -2.25 mm. On the identical centered 5 s core, the corresponding changes are 1.82 and -0.60 mm. This supports the interpretation that fusion/correction removes much of the raw curve's apparent low-travel degradation rather than propagating it to final travel.

## Median RMSE by duration

| Training | Full log: mag | Full log: Cabsolved | Training window: mag | Training window: solved | 0–30 mm training: mag | 0–30 mm training: solved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 s | 5.40 | 5.19 | 4.07 | 4.28 | 5.67 | 6.75 |
| 10 s | 4.71 | 3.93 | 3.70 | 3.56 | 5.83 | 5.59 |
| 20 s | 4.01 | 3.48 | 3.50 | 3.24 | 6.59 | 4.85 |
| 40 s | 4.55 | 3.35 | 4.02 | 3.15 | 7.49 | 5.00 |
| 120 s | 4.12 | 3.28 | 4.34 | 3.19 | 8.00 | 5.25 |

All values are millimetres and are medians across logs. Negative paired changes mean lower error at the longest duration. The p-values are descriptive only: this Phase 1 run uses one random window per log and was not designed as a definitive significance test.

## Runtime and artifacts

Median runtime was 12.6 s per full-log condition (12.5 solver-minutes total).
The prediction arrays were inspected during the pilot and removed after Phase 2 completed because they were large, reproducible execution artifacts. The row-level metrics remain available locally, while the frozen schedule, manifest, aggregate table, report, and learning-curve figure form the versioned record.
