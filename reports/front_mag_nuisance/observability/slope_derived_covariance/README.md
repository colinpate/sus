# Slope-Derived Magnetometer Covariance

## Result

Slope-derived tangent uncertainty works, including within the existing 1500 mG
gate, but it is not an unambiguous replacement for the current iterative
setting.

The best strictly consistent adaptive iterative setting is one outer iteration,
5 mm assumed travel sigma, and a 75% output update. It lowers cohort mean
weak-field RMSE from 10.12 to 9.10 mm and improves all 15 logs. The existing
four-iteration isotropic method reaches 9.12 mm and also improves all 15 logs.
The difference in cohort mean is only 0.02 mm, and the existing method remains
better on the Boxxer setup.

The all-sample curve-normal experiment supports downweighting high-slope normal
residuals too. With 5 mm travel sigma and no normal inflation, its worst log
regresses by 3.79 mm at a 50% output update. Giving the normal directions 10%
of the slope-derived uncertainty improves every log, with a 9.24 mm cohort mean
and at least 0.17 mm improvement on every log.

## Is slope flat below 1500 mG?

Not in the learned magnitude-era XYZ curves. Using equal travel spacing rather
than sample density, pooled curve-slope percentiles are:

| Region | P10 | P50 | P90 | P90/P10 |
| --- | ---: | ---: | ---: | ---: |
| Predicted norm ≤1500 mG | 6.53 | 14.96 | 43.62 | 6.68× |
| Predicted norm >1500 mG | 61.97 | 526.42 | 11550.92 | 186.4× |

Values are mG/mm. Median weak-field slope is about 20 mG/mm on the Fox 36 logs
and 9 mG/mm on the Boxxer logs. Therefore adaptive covariance changes
meaningfully inside the gate and behaves differently between setups.

## Covariance model

For the learned XYZ curve `f(t)`, the local Jacobian is the 3D vector

```text
J(t) = df/dt
```

The physically propagated covariance is

```text
R = sigma_mag² I + J sigma_travel² Jᵀ
```

with `sigma_mag = 40 mG`. This leaves both curve-normal directions at 40 mG
and gives the tangent direction sigma

```text
sqrt(40² + (norm(J) * sigma_travel)²).
```

For the all-sample experiment, `normal_slope_fraction = c` adds the heuristic
normal variance

```text
(c * norm(J) * sigma_travel)².
```

This represents tangent-direction error or XYZ-model error leaking a fraction
of the travel-induced residual into the normal plane. It is not part of the
first-order travel-uncertainty propagation itself.

## Effective weights at 5 mm travel sigma

Weights are relative to a 40 mG observation. The three columns are evaluated at
the pooled P10/P50/P90 slopes in each region.

| Region | Direction/model | P10 weight | P50 weight | P90 weight |
| --- | --- | ---: | ---: | ---: |
| Weak | Tangent | 60.0% | 22.2% | 3.25% |
| Weak | Normal, 10% leakage | 99.3% | 96.6% | 77.1% |
| Strong | Tangent | 1.64% | 0.023% | 0.000048% |
| Strong | Normal, 10% leakage | 62.5% | 2.26% | 0.0048% |

Thus 10% normal leakage behaves like a soft high-slope gate: it barely changes
typical weak-field normal observations but strongly suppresses normal residuals
where the learned curve becomes steep.

The requested near-isotropic point is 0.5 mm travel sigma. At the median weak
slope it retains 96.6% tangent weight; even at the weak P90 it retains 77.1%.
As expected, its results stay close to the isotropic model.

## Adaptive iterative sweep

The table uses the previously selected 75% output update. All field-state
updates and travel applications retain the existing predicted/measured 1500 mG
gates.

| Travel sigma | 1 iteration mean / improved | 2 iterations | 4 iterations |
| ---: | ---: | ---: | ---: |
| 0 mm | 9.68 / 14 | 9.41 / 14 | **9.12 / 15** |
| 0.5 mm | 9.63 / 14 | 9.35 / 14 | 9.05 / 14 |
| 1 mm | 9.52 / 14 | 9.23 / 14 | 8.93 / 14 |
| 2.5 mm | 9.26 / 15 | 8.99 / 14 | 8.75 / 14 |
| 5 mm | **9.10 / 15** | 8.90 / 13 | 8.73 / 12 |
| 10 mm | 9.04 / 14 | 8.94 / 13 | 8.81 / 11 |

Additional iterations generally reduce aggregate mean but can reinforce a
wrong nuisance/travel solution on a few logs. At four iterations and 5 mm,
`0073`, `0049`, and `0063` regress by 0.70, 0.33, and 0.04 mm respectively.

Reducing output alpha helps but does not fully solve this. Four iterations and
5 mm at alpha 0.5 reach 8.92 mm mean but still regress `0073` by 0.30 mm.

## All-sample curve-normal sweep

These candidates update the body/world field from all magnitudes, then apply
corrected travel only where both predicted and measured magnitude are at most
1500 mG. All rows below use 5 mm travel sigma and alpha 0.5.

| Normal slope fraction | Mean RMSE | Median RMSE | Improved | Worst delta |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 9.25 | 7.72 | 10/15 | +3.79 |
| **10%** | **9.24** | 7.46 | **15/15** | **-0.17** |
| 25% | 9.27 | 7.45 | 15/15 | -0.16 |
| 50% | 9.30 | 7.40 | 15/15 | -0.14 |
| 100% | 9.34 | **7.37** | 15/15 | -0.08 |

Pure tangent propagation leaves high-field curve-normal residuals at full
weight and is unsafe. Even modest slope-linked normal uncertainty removes those
large regressions. This supports the idea that high-field normal residuals
contain slope-dependent model or tangent-direction error, not just additive
nuisance field.

## Selected per-log comparison

| Log | Setup | Pipeline | Existing iterative | Adaptive iterative | All-sample adaptive normal |
| --- | --- | ---: | ---: | ---: | ---: |
| log-0056 | Fox 36 | 7.60 | 5.58 | 5.15 | 5.79 |
| log-0063 | Fox 36 | 5.32 | 5.13 | 4.72 | 4.67 |
| log-0046 | Fox 36 | 14.76 | 12.83 | 12.21 | 13.02 |
| log-0048 | Fox 36 | 6.91 | 5.65 | 5.38 | 5.60 |
| log-0049 | Fox 36 | 7.79 | 7.64 | 7.60 | 7.46 |
| log-0054 | Fox 36 | 8.08 | 6.71 | 6.51 | 6.91 |
| log-0055 | Fox 36 | 8.37 | 7.75 | 8.37 | 8.20 |
| log-0058 | Fox 36 | 3.93 | 3.59 | 3.54 | 3.17 |
| log-0071_183 | Fox 36 | 8.04 | 7.50 | 7.13 | 7.33 |
| log-0072_184 | Fox 36 | 4.95 | 4.75 | 4.69 | 4.54 |
| log-0073_185 | Fox 36 | 14.05 | 14.03 | 13.87 | 13.84 |
| log-0078-valid | Boxxer | 16.05 | 13.02 | 14.07 | 14.30 |
| log-0079 | Boxxer | 12.50 | 11.41 | 11.38 | 11.27 |
| log-0080-valid | Boxxer | 16.90 | 15.80 | 15.88 | 16.45 |
| log-0081 | Boxxer | 16.58 | 15.45 | 16.00 | 16.05 |

`Existing iterative` is four iterations, isotropic tangent weight, alpha 0.75.
`Adaptive iterative` is one iteration, 5 mm travel sigma, alpha 0.75.
`All-sample adaptive normal` is one all-sample solve, 5 mm travel sigma, 10%
normal slope fraction, alpha 0.5.

| Method | Cohort mean | Cohort median | Fox 36 median | Boxxer median | Improved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing iterative | 9.12 | 7.64 | 6.71 | **14.24** | 15/15 |
| Adaptive iterative | **9.10** | 7.60 | **6.51** | 14.97 | 15/15 |
| All-sample adaptive normal | 9.24 | **7.46** | 6.91 | 15.17 | 15/15 |

## Recommendation

The existing four-iteration isotropic setting remains the conservative default.
Its overall mean is essentially tied with the best adaptive result, and it
generalizes better to Boxxer. The 5 mm one-iteration adaptive model is worth
testing on unseen logs because it is simpler and slightly better on Fox 36,
but selecting it from this development cohort would overstate the 0.02 mm mean
advantage.

For observations above 1500 mG, slope-derived tangent covariance should be
paired with slope-linked normal inflation. A 10% normal fraction is the best
starting point tested here. Its benefit appears to come from softly rejecting
high-slope model error rather than extracting substantially more useful signal
than the gated iterative model.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/experiment_slope_derived_covariance.py
```

Raw per-log/alpha metrics are in `metrics.csv`, ranked cohort metrics are in
`summary.csv`, and slope/iteration diagnostics are in `details.json`.
