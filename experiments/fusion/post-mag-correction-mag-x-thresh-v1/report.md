# Post-mag-correction fusion `mag_x_thresh` sweep

## Design

The final front fusion solver was replayed with `mag_off_floor=0.1` fixed. Threshold modes were `baseline`, `1250`, `1000`, `750`, `500`, `disabled`; `baseline` is the current per-log magnetic baseline and `disabled` gives every sample full magnetic residual weight. Numeric modes are absolute mG thresholds.

Tuning used 31 logs from `harry`, `jamaal`, `stumpjumper-front-pod-v2`. The robust selection rule required a candidate to avoid worsening both 0–30 mm and overall centered cohort means versus current in every tuning cohort, then minimized cohort-balanced 0–30 mm RMSE. `stumpjumper-front-pod-v1` was evaluated only after selection, but this cohort has been examined by an earlier hyperparameter experiment, so its result is exploratory rather than pristine validation.

## Tuning result

| threshold mode | mean resolved threshold | full-weight samples | 0–30 mm RMSE | overall RMSE | eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 1189 mG | 76.7% | 8.822 mm | 5.879 mm | yes |
| `1250` | 1250 mG | 72.0% | 8.628 mm | 5.854 mm | no |
| `1000` | 1000 mG | 91.4% | 7.413 mm | 5.639 mm | no |
| `750` | 750 mG | 96.1% | 7.312 mm | 5.630 mm | no |
| `500` | 500 mG | 98.6% | 7.338 mm | 5.662 mm | no |
| `disabled` | disabled | 100.0% | 7.321 mm | 5.662 mm | no |

Selected: **`baseline`**. On tuning, cohort-balanced 0–30 mm RMSE changed by +0.000 mm and overall centered RMSE changed by +0.000 mm.

The strict selector retained current behavior because every alternative caused some cohort-level regression. However, `750` was the best cohort-balanced candidate on both centered objectives, improving 0–30 mm RMSE by 1.510 mm (17.1%) and overall centered RMSE by 0.249 mm (4.2%). Its setup breakdown was:

| tuning cohort for `750` | 0–30 mm change | logs improved | overall change | logs improved | uncentered change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `harry` | -3.679 mm | 14/14 | -0.834 mm | 14/14 | +0.116 mm |
| `jamaal` | -0.822 mm | 3/6 | +0.046 mm | 2/6 | +0.385 mm |
| `stumpjumper-front-pod-v2` | -0.029 mm | 7/11 | +0.040 mm | 7/11 | +0.199 mm |

## Exploratory pod-v1 result

Because the strict winner was the unchanged baseline, no distinct selected condition existed to validate. The aggregate-best `750` mode was therefore run afterward as an explicitly post-selection exploratory check.

| mode | 0–30 mm RMSE | overall centered RMSE | uncentered RMSE |
| --- | ---: | ---: | ---: |
| `baseline` | 7.058 mm | 3.960 mm | 11.201 mm |
| `750` | 6.618 mm | 3.827 mm | 11.521 mm |

The 750 mG cutoff changed mean 0–30 mm RMSE by -0.440 mm (5/7 logs improved), overall centered RMSE by -0.132 mm (5/7), and uncentered RMSE by +0.320 mm (2/7 improved). It gave 93.1% of pod-v1 evaluation samples full magnetic weight.

## Interpretation

Decreasing `mag_x_thresh` is directionally useful, and 750 mG is the best practical point in this grid for centered shape accuracy. It is preferable to disabling the gate: it slightly outperformed the fully-on endpoint on tuning and had a smaller uncentered penalty on pod-v1. The remaining concern is absolute offset—uncentered RMSE rose across every tuning cohort and by 0.320 mm on pod-v1—so this experiment alone does not justify changing the production default unless centered error is the governing metric.

## Checks

The current `baseline` replay matched cached final solver output to a worst-case absolute difference of 0 mm. Every tuning solve converged.

Full per-log values are in `per_log_metrics.csv`; cohort summaries are in `aggregate_metrics.csv`; the frozen decision is in `selection.json`.
