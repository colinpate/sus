# solver-window-front-phase2

Status: **220/220 successful** (0 failed).

## Design

- Front pipeline; 11 Stumpjumper/pod-v2 logs.
- 4 deterministic nested window(s) per log (source repeats 0, 1, 2, 3) at 5 s, 10 s, 20 s, 40 s, 120 s.
- The saved self-supervised curve is injected before the two full-log fusion solves.
- Scoring is applied afterward on the training window, a fixed centered common core, the full log, and the full log excluding training.
- Fixed-offset window metrics use the corresponding stage's full-log alignment, so window-local recentering cannot hide bias.

## Main result

On the full log, median raw magnetic-model RMSE changes from 5.10 to 4.10 mm; the paired 120-5 s change is -0.59 mm (73% of logs improve, exploratory Wilcoxon p=0.083).
For the final solved output, median full-log RMSE changes from 4.52 to 3.24 mm; the paired change is -1.31 mm (100% improve, p=0.001).

The suspicious low-travel trend is strongly attenuated downstream. On the varying training windows, raw 0–30 mm fixed-offset RMSE has a paired change of 2.64 mm from 5 to 120 s, while the final solved output changes by 0.43 mm. On the identical centered 5 s core, the corresponding changes are 2.39 and 0.61 mm. This supports the interpretation that fusion/correction removes much of the raw curve's apparent low-travel degradation rather than propagating it to final travel.

## Practical duration

The typical paired full-log solved improvement is 0.17 mm from 20 to 40 s (p=0.413) and only 0.04 mm from 40 to 120 s (p=0.465). The central-error curve therefore has a practical elbow around 20–40 active seconds.
Longer calibration still improves repeatability and tail risk. Median within-log repeat IQR falls from 0.40 mm at 20 s to 0.25 mm at 40 s and 0.21 mm at 120 s. The across-condition 90th percentile is 5.30, 5.83, and 4.16 mm, respectively. This makes 40 s a reasonable default, while 120 s is preferable when robustness to an unlucky calibration window matters more than calibration latency.

## Median RMSE by duration

| Training | Full log: mag | Full log: solved | Training window: mag | Training window: solved | 0–30 mm training: mag | 0–30 mm training: solved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 s | 5.10 | 4.52 | 4.06 | 3.51 | 4.63 | 4.40 |
| 10 s | 4.39 | 4.17 | 4.03 | 3.65 | 5.14 | 5.31 |
| 20 s | 4.73 | 3.68 | 4.20 | 3.52 | 7.08 | 4.90 |
| 40 s | 4.39 | 3.29 | 4.08 | 3.26 | 7.60 | 4.89 |
| 120 s | 4.10 | 3.24 | 4.31 | 3.11 | 8.35 | 5.32 |

All values are millimetres. Repeats are first collapsed within each log, then logs are weighted equally. Negative paired changes mean lower error at the longest duration. The p-values are descriptive only: this run remains an exploratory rather than confirmatory significance study.

## Runtime and artifacts

Median runtime was 15.5 s per full-log condition (65.6 solver-minutes total).
Raw per-trial metrics, aggregate tables, and the learning-curve figure are stored beside this report. The frozen schedule links every solve to its exact source calibration and cached-input fingerprint.
