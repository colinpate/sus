# Iterative Tangent-Weight Ablation

## Result

Tangent residuals are useful—and, on this cohort, necessary—for the iterative
weak-field body/world correction. Four iterations with full tangent weight
remain the most consistent setting, improving all 15 logs. Moderate tangent
downweighting gives a slightly better cohort mean but loses a small amount of
cross-log and cross-fork consistency.

The best cohort mean is tangent sigma ratio 2 with four iterations: weak-field
RMSE falls from 10.12 to 8.93 mm. It improves 14 of 15 logs; the remaining
change is a practically negligible +0.005 mm on `log-0063`. Full tangent
weight, ratio 1, reaches 9.12 mm and strictly improves every log.

Nearly removing the tangent observation is clearly harmful. Ratio 100 raises
mean RMSE to 13.19 mm after four iterations and regresses 9 of 15 logs, by as
much as 11.29 mm.

## Controlled experiment

Every candidate uses the same:

- current magnitude-pipeline caches and encoder-blind XYZ model;
- initial pipeline travel;
- 1500 mG predicted-field observation gate;
- 1500 mG measured-field application gate;
- body/world process model and gyro propagation;
- 40 mG curve-normal measurement sigma; and
- 75% final travel-space update.

Only outer iteration count and tangent sigma ratio change. At each outer
iteration, the tangent is recomputed as the derivative of the XYZ model at the
current travel estimate. A ratio of `r` gives the tangent residual `1/r²` as
much measurement weight as either normal direction:

| Sigma ratio | Relative tangent weight |
| ---: | ---: |
| 1 | 100% |
| 2 | 25% |
| 5 | 4% |
| 10 | 1% |
| 100 | 0.01% |

Encoder travel is loaded only after all candidate predictions for a log have
been generated and is used only for scoring.

## Cohort weak-field RMSE

All RMSE and delta values are millimeters. `Improved` uses a strict comparison
against the pipeline. `Worst delta` is candidate minus pipeline for the most
regressed log; a negative value means every log improved.

| Tangent ratio | Iterations | Mean RMSE | Median RMSE | Improved | Worst delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 9.68 | 7.93 | 14/15 | +0.221 |
| 1 | 2 | 9.41 | 7.76 | 14/15 | +0.106 |
| **1** | **4** | **9.12** | 7.64 | **15/15** | **-0.013** |
| 2 | 1 | 9.49 | 7.86 | 15/15 | -0.046 |
| 2 | 2 | 9.21 | 7.76 | 15/15 | -0.082 |
| **2** | **4** | **8.93** | 7.63 | 14/15 | +0.005 |
| 5 | 1 | 9.34 | 7.92 | 12/15 | +0.131 |
| 5 | 2 | 9.14 | 7.86 | 13/15 | +0.281 |
| 5 | 4 | 8.94 | **7.51** | 13/15 | +0.562 |
| 10 | 1 | 9.53 | 8.45 | 9/15 | +0.990 |
| 10 | 2 | 9.40 | 8.27 | 9/15 | +0.964 |
| 10 | 4 | 9.28 | 7.81 | 11/15 | +1.419 |
| 100 | 1 | 14.58 | 14.73 | 5/15 | +16.857 |
| 100 | 2 | 12.87 | 13.03 | 6/15 | +9.504 |
| 100 | 4 | 13.19 | 13.97 | 6/15 | +11.287 |

Iteration independently helps when useful tangent information is retained.
For ratio 1, mean RMSE improves from 9.68 to 9.41 to 9.12 mm as iteration count
increases from one to two to four. Ratios 2, 5, and 10 show the same monotonic
mean improvement. Ratio 100 is the exception and remains worse than the 10.12
mm pipeline baseline.

## Four-iteration per-log RMSE

| Log | Setup | Pipeline | Ratio 1 | Ratio 2 | Ratio 5 | Ratio 10 | Ratio 100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| log-0056 | Fox 36 | 7.60 | 5.58 | 5.24 | 5.15 | 5.84 | 17.33 |
| log-0063 | Fox 36 | 5.32 | 5.13 | 5.33 | 5.88 | 5.92 | 4.71 |
| log-0046 | Fox 36 | 14.76 | 12.83 | 12.26 | 11.67 | 11.42 | 14.32 |
| log-0048 | Fox 36 | 6.91 | 5.65 | 5.08 | 4.48 | 4.34 | 5.94 |
| log-0049 | Fox 36 | 7.79 | 7.64 | 7.64 | 8.13 | 9.07 | 14.86 |
| log-0054 | Fox 36 | 8.08 | 6.71 | 6.34 | 6.05 | 6.61 | 13.97 |
| log-0055 | Fox 36 | 8.37 | 7.75 | 7.63 | 7.27 | 6.78 | 6.20 |
| log-0058 | Fox 36 | 3.93 | 3.59 | 3.59 | 3.65 | 3.86 | 7.27 |
| log-0071_183 | Fox 36 | 8.04 | 7.50 | 7.47 | 7.51 | 7.97 | 13.14 |
| log-0072_184 | Fox 36 | 4.95 | 4.75 | 3.87 | 4.19 | 4.49 | 6.89 |
| log-0073_185 | Fox 36 | 14.05 | 14.03 | 14.01 | 14.04 | 14.24 | 15.36 |
| log-0078-valid | Boxxer | 16.05 | 13.02 | 13.11 | 13.90 | 15.98 | 27.34 |
| log-0079 | Boxxer | 12.50 | 11.41 | 10.94 | 9.30 | 7.81 | 10.21 |
| log-0080-valid | Boxxer | 16.90 | 15.80 | 16.00 | 16.79 | 16.89 | 16.61 |
| log-0081 | Boxxer | 16.58 | 15.45 | 15.40 | 16.01 | 18.00 | 23.78 |

## Setup dependence

| Four-iteration method | Fox 36 median | Boxxer median |
| --- | ---: | ---: |
| Pipeline | 7.79 | 16.32 |
| Ratio 1 | 6.71 | **14.24** |
| Ratio 2 | 6.34 | 14.25 |
| Ratio 5 | **6.05** | 14.95 |
| Ratio 10 | 6.61 | 16.44 |
| Ratio 100 | 13.14 | 20.19 |

The Fox 36 logs benefit from moderate downweighting, whereas the Boxxer logs
prefer full tangent weight. This explains why ratios 2–5 can improve aggregate
mean or median while becoming less consistent across logs.

## Interpretation

The controlled result supports both parts of the proposed explanation:

1. Re-estimating travel and nuisance fields is useful. With tangent information
   retained, additional outer iterations consistently lower cohort mean error.
2. Low-field tangent residuals contain important information. Curve-normal
   observations alone do not sufficiently constrain the nuisance field for
   these logs, even though gyro-driven rotation can theoretically make more
   field directions observable over time.

The result does not imply that tangent residuals are perfectly trustworthy.
Ratio 2's lower mean shows that some downweighting can help. It instead says
that the useful operating range is near full weight—not near eliminating the
tangent component.

For a conservative production starting point, ratio 1 with four iterations is
still best supported because it improves every log and transfers slightly
better to the Boxxer setup. Ratio 2 is the leading candidate if future unseen
logs confirm that its +0.005 mm `0063` regression is noise rather than a sign of
weaker robustness.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/experiment_iterative_tangent_ablation.py
```

Raw per-log values are in `metrics.csv`, the cohort comparison is in
`summary.csv`, and per-iteration changes are in `details.json`.
