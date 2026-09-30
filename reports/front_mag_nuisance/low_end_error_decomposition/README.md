# Low-end learned-curve versus nuisance-field error

## Scope and centering

This uses the 15 pod-v2 caches regenerated on 2026-08-30 with pure MMC5603
magnitude as the scalar magnet signal. It is intentionally encoder-supervised:
the purpose is to diagnose how the encoder-free learned curve fails, not to
propose a production estimator.

For each 5 mm encoder-travel bin, the sample error is decomposed exactly as:

```text
sample error = travel-bin mean error + within-bin residual
total MSE    = curve MSE           + scatter MSE
```

The primary result is centered. Before analyzing the samples below 30 mm, it
subtracts the mean prediction error over the full active (`boring_mask`)
recording. This removes one constant travel-reference error, such as an error
from the absolute zero-offset calibration section. The remaining curve term is
therefore disagreement in the low-end curve relative to the rest of the
recording, rather than merely a globally shifted curve.

`curve MSE` is the operational disagreement between the learned power curve
and the whole-log encoder-binned magnitude curve. `scatter MSE` includes
time-varying nuisance field, sensor noise, and state dependence not represented
by travel alone.

This is not a perfect causal separation. A travel-dependent or very slowly
changing nuisance field can shift the encoder-binned curve itself and is then
counted as curve disagreement. The curve number is therefore an upper bound on
error caused by the learned mapping alone. A five-time-block decomposition
separately exposes the clearly time-varying component.

## Primary result below 30 mm

Across the logs, the median centered scalar-model RMSE is 9.16 mm. Its median
components are:

- 6.63 mm encoder-binned curve disagreement;
- 5.78 mm within-bin scatter;
- 55% of MSE associated with curve disagreement.

The cohort mean curve share is 52%. Thus, after removing the constant
zero-reference error, learned-curve disagreement and nuisance/scatter are about
equally important overall. They are not equally important in every log.

| Log | Global offset removed | Centered scalar RMSE | Curve RMS | Scatter RMS | Curve share | Oracle curve-fix gain | Centered solved RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| log-0080-valid | +12.31 | 18.55 | 11.56 | 14.51 | 39% | 4.04 | 14.96 |
| log-0054 | -7.40 | 10.62 | 8.40 | 6.49 | 63% | 4.13 | 6.03 |
| log-0056 | -7.95 | 9.97 | 7.21 | 6.89 | 52% | 3.08 | 6.03 |
| log-0081 | -13.21 | 9.87 | 2.42 | 9.57 | 6% | 0.30 | 9.65 |
| log-0046 | -15.95 | 9.79 | 8.46 | 4.93 | 75% | 4.86 | 7.30 |
| log-0078-valid | -15.25 | 9.29 | 4.94 | 7.87 | 28% | 1.42 | 7.43 |
| log-0058 | -2.08 | 9.23 | 4.31 | 8.16 | 22% | 1.07 | 5.48 |
| log-0071_183 | -9.48 | 9.16 | 7.74 | 4.90 | 71% | 4.26 | 3.97 |
| log-0072_184 | -0.55 | 8.80 | 7.18 | 5.10 | 66% | 3.70 | 6.52 |
| log-0055 | +6.36 | 8.80 | 6.63 | 5.78 | 57% | 3.02 | 7.19 |
| log-0073_185 | -10.49 | 8.17 | 7.85 | 2.26 | 92% | 5.90 | 7.55 |
| log-0048 | -6.41 | 8.09 | 5.31 | 6.10 | 43% | 1.98 | 6.42 |
| log-0049 | -6.39 | 6.82 | 6.01 | 3.22 | 78% | 3.60 | 6.23 |
| log-0063 | -2.24 | 5.51 | 4.10 | 3.68 | 55% | 1.83 | 5.24 |
| log-0079 | -11.70 | 5.05 | 3.13 | 3.97 | 38% | 1.08 | 9.07 |

`Oracle curve-fix gain` is the reduction from centered scalar RMSE to the
within-bin residual RMS if every encoder-bin mean error could be removed. It is
an upper bound, not a deployable correction.

The clearest curve-dominated logs are log-0073_185, log-0049, log-0046,
log-0071_183, and log-0072_184. The clearest nuisance-dominated logs are
log-0081, log-0058, and log-0078-valid. Log-0080-valid has both a substantial
curve component and the largest nuisance component by far.

## Centered curve distance

The following compares the encoder-binned magnitude curve to the learned curve
after applying the same global travel centering:

| Log | Typical / maximum travel disagreement below 30 mm | Typical / maximum magnitude disagreement |
| --- | ---: | ---: |
| log-0080-valid | 11.57 / 18.29 mm | 42 / 83 mG |
| log-0054 | 10.89 / 15.54 mm | 133 / 214 mG |
| log-0056 | 8.03 / 13.87 mm | 105 / 139 mG |
| log-0081 | 2.87 / 6.62 mm | 24 / 74 mG |
| log-0046 | 8.91 / 11.67 mm | 157 / 222 mG |
| log-0078-valid | 1.90 / 5.76 mm | 13 / 38 mG |
| log-0058 | 2.16 / 10.48 mm | 16 / 104 mG |
| log-0071_183 | 5.77 / 13.19 mm | 55 / 152 mG |
| log-0072_184 | 9.73 / 12.64 mm | 79 / 172 mG |
| log-0055 | 7.96 / 12.12 mm | 123 / 174 mG |
| log-0073_185 | 5.62 / 11.92 mm | 325 / 376 mG |
| log-0048 | 5.69 / 10.87 mm | 83 / 112 mG |
| log-0049 | 6.22 / 9.68 mm | 155 / 218 mG |
| log-0063 | 4.81 / 5.88 mm | 131 / 268 mG |
| log-0079 | 2.64 / 3.65 mm | 21 / 36 mG |

The same magnitude error produces very different travel error because low-end
sensitivity differs substantially between setups and learned powers.

## Time-varying component

Splitting each log into five time blocks gives an exact three-part decomposition
on sufficiently populated travel/time cells. Centered cohort medians are:

- stable whole-log curve component: 6.63 mm RMS;
- block-to-block drift: 2.21 mm RMS;
- within-block scatter: 5.26 mm RMS.

The median MSE shares are 57% stable curve, 7% block drift, and 37%
within-block scatter (small differences in eligible cells cause rounding and a
slight difference from the two-part result).

For log-0080-valid, the centered terms are 11.60, 8.11, and 12.03 mm, or
39% stable curve and 61% time-varying nuisance. For log-0081, only 6% is stable
curve while 94% is time-varying. By contrast, log-0073_185 is 93% stable curve.

Changing encoder-bin width from 2 to 5 to 10 mm changes the centered cohort
median curve-MSE fraction from 59% to 55% to 50%. The exact split depends
somewhat on resolution, but the conclusion that both mechanisms matter does
not.

## What the uncentered result was measuring

Without removing the global offset, the median low-end RMSE is 15.60 mm, the
curve RMS is 12.17 mm, and the curve share is 82%. The scatter component remains
5.78 mm because subtracting a constant cannot affect within-bin scatter.

The reduction from 82% to 55% is important: much of the initially reported
"curve disagreement" was a constant reference/calibration offset, not evidence
that the learned curve had the wrong low-end shape. The centered result should
be used for deciding whether to improve the curve family versus nuisance-field
handling.

## Check against the archived Jamaal control

The current caches are not reproducing the archived projection-based control.
Re-running `stats_aggregator.py --center-errors` on the current caches gives:

| Log | Old centered mag RMSE | Current magnitude RMSE | Old centered solved RMSE | Current solved RMSE |
| --- | ---: | ---: | ---: | ---: |
| log-0078-valid | 11.12 | 8.21 | 10.75 | 7.39 |
| log-0079 | 13.25 | 7.37 | 11.11 | 7.63 |
| log-0080-valid | 17.22 | 10.77 | 14.87 | 9.49 |
| log-0081 | 17.11 | 11.05 | 16.03 | 10.27 |

For log-0080-valid, the cached pre-filter projection and magnitude arrays are
not equal and differ by 554 mG RMS. The suspicious numerical match was between
the old report's 15.615 mm centered binned statistic and the new diagnostic's
approximately 15.60 mm uncentered, below-30-mm scalar RMSE. They are different
metrics.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/analyze_low_end_curve_error.py
```

Outputs:

- `summary.csv`: raw and centered decompositions at 10, 20, 30, and 50 mm;
- `encoder_binned_curves.csv`: raw and centered learned-versus-encoder curves;
- `temporal_decomposition.csv`: raw and centered curve, block-drift, and within-block terms;
- `bin_width_sensitivity.csv`: raw and centered 2/5/10 mm sensitivity;
- `current_jamaal_centered_stats/`: newly regenerated centered control metrics.
