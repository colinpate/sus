# solver-window-front-stumpy-postmerge-v1

Status: **220/220 successful** (0 failed).

## Design

- Front pipeline; 11 logs from the 11-log Stumpjumper/pod-v2 cohort.
- 4 deterministic nested window(s) per log (source repeats 0, 1, 2, 3) at 5 s, 10 s, 20 s, 40 s, 120 s.
- The saved self-supervised curve is injected before the two full-log fusion solves.
- Scoring is applied afterward on the training window, a fixed centered common core, the full log, and the full log excluding training.
- Fixed-offset window metrics use the corresponding stage's full-log alignment, so window-local recentering cannot hide bias.

## Main result

On the full log, median raw magnetic-model RMSE changes from 4.62 to 4.23 mm; the paired 120-5 s change is -0.56 mm (91% of logs improve, exploratory Wilcoxon p=0.005).
For the final solved output, median full-log RMSE changes from 3.98 to 2.65 mm; the paired change is -1.23 mm (100% improve, p=0.001).

The suspicious low-travel trend is strongly attenuated downstream. On the varying training windows, raw 0–30 mm fixed-offset RMSE has a paired change of 2.01 mm from 5 to 120 s, while the final solved output changes by 0.56 mm. On the identical centered 5 s core, the corresponding changes are 0.46 and -0.72 mm. This supports the interpretation that fusion/correction removes much of the raw curve's apparent low-travel degradation rather than propagating it to final travel.

## Practical duration

The typical paired full-log solved improvement is 0.13 mm from 20 to 40 s (p=0.032) and only 0.11 mm from 40 to 120 s (p=0.005). The central-error curve therefore has a practical elbow around 20–40 active seconds.
Longer calibration still improves repeatability and tail risk. Median within-log repeat IQR falls from 0.52 mm at 20 s to 0.37 mm at 40 s and 0.15 mm at 120 s. The across-condition 90th percentile is 6.25, 4.03, and 3.57 mm, respectively. This makes 40 s a reasonable default, while 120 s is preferable when robustness to an unlucky calibration window matters more than calibration latency.

## Median RMSE by duration

| Training | Full log: mag | Full log: solved | Training window: mag | Training window: solved | 0–30 mm training: mag | 0–30 mm training: solved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 s | 4.62 | 3.98 | 4.11 | 3.62 | 5.26 | 4.21 |
| 10 s | 4.29 | 3.20 | 3.44 | 3.45 | 5.21 | 4.64 |
| 20 s | 4.17 | 3.36 | 3.56 | 3.08 | 6.17 | 4.49 |
| 40 s | 3.78 | 2.93 | 3.81 | 2.65 | 7.15 | 4.44 |
| 120 s | 4.23 | 2.65 | 4.28 | 2.77 | 7.28 | 4.90 |

All values are millimetres. Repeats are first collapsed within each log, then logs are weighted equally. Negative paired changes mean lower error at the longest duration. The p-values are descriptive only: this run remains an exploratory rather than confirmatory significance study.

## Runtime and artifacts

Median runtime was 15.3 s per full-log condition (68.8 solver-minutes total).
Fresh runs write row-level metrics, aggregate tables, and the learning-curve figure beside this report. The frozen schedule links every solve to its exact source calibration and cached-input fingerprint; large execution caches need not be versioned.
