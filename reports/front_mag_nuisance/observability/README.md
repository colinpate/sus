# Magnitude-Era Curve-Normal Nuisance Observation

## Updated result

The 15 pod-v2 logs were rerun through the current front pipeline after the
scalar magnet signal was changed from projection to magnitude and renamed to
`mag/norm`. The XYZ nuisance-field experiment was then rerun from those fresh
caches. Encoder travel is used only after prediction generation for scoring.

The projection-era curve-normal parameters are no longer uniformly safe on the
magnitude pipeline. The old setting—90% acceleration tangent, tangent sigma
ratio 5, and a 50% weak-field output update—reduces cohort mean weak-field RMSE
from 10.12 to 9.42 mm, but regresses `log-0046` by 0.10 mm and `log-0080-valid`
by 1.44 mm.

A compact magnitude-era sweep found a conservative fixed setting that improves
all 15 logs:

- tangent direction: 100% acceleration-derived;
- measurement sigma: 40 mG perpendicular and 80 mG along the tangent;
- no full-vector field anchors;
- one body/world field solve with no outer field/travel iteration; and
- a 25% travel-space update applied only in the weak-field region.

This setting lowers cohort mean weak-field RMSE from 10.12 to 9.73 mm and the
median from 8.08 to 7.81 mm. The gain is consistent but modest: the median
per-log improvement is 0.31 mm, and the smallest is 0.05 mm.

## Per-log weak-field RMSE

All values are millimeters. `Change` is curve-normal minus pipeline, so a
negative value is an improvement. Weak field is measured primary-field norm
below 1500 mG; the existing boring-region mask is also applied.

| Log | Setup | Pipeline | Conservative curve normal | Change | Old curve-normal setting | Previous iterative |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| log-0056 | Fox 36 | 7.60 | 6.96 | -0.65 | 6.12 | 5.58 |
| log-0063 | Fox 36 | 5.32 | 5.01 | -0.31 | 5.02 | 5.13 |
| log-0046 | Fox 36 | 14.76 | 14.38 | -0.39 | 14.86 | 12.83 |
| log-0048 | Fox 36 | 6.91 | 6.49 | -0.42 | 5.74 | 5.65 |
| log-0049 | Fox 36 | 7.79 | 7.57 | -0.22 | 7.72 | 7.64 |
| log-0054 | Fox 36 | 8.08 | 7.80 | -0.28 | 7.45 | 6.71 |
| log-0055 | Fox 36 | 8.37 | 8.13 | -0.23 | 7.89 | 7.75 |
| log-0058 | Fox 36 | 3.93 | 3.39 | -0.54 | 3.09 | 3.59 |
| log-0071_183 | Fox 36 | 8.05 | 7.81 | -0.24 | 7.44 | 7.50 |
| log-0072_184 | Fox 36 | 4.95 | 4.80 | -0.15 | 3.90 | 4.75 |
| log-0073_185 | Fox 36 | 14.05 | 13.93 | -0.12 | 13.92 | 14.03 |
| log-0078-valid | Boxxer | 16.05 | 14.78 | -1.28 | 13.85 | 13.02 |
| log-0079 | Boxxer | 12.50 | 11.80 | -0.70 | 10.02 | 11.41 |
| log-0080-valid | Boxxer | 16.90 | 16.85 | -0.05 | 18.34 | 15.80 |
| log-0081 | Boxxer | 16.58 | 16.21 | -0.37 | 15.88 | 15.46 |

`Previous iterative` is the earlier weak-only iterative body/world correction
at a 75% output update. It is included as context rather than as part of the
curve-normal comparison.

## Aggregate comparison

| Metric | Pipeline | Conservative curve normal | Old curve-normal setting | Previous iterative |
| --- | ---: | ---: | ---: | ---: |
| Cohort mean | 10.12 | 9.73 | 9.42 | **9.12** |
| Cohort median | 8.08 | 7.81 | 7.72 | **7.64** |
| Fox 36 median, 11 logs | 7.79 | 7.57 | 7.44 | **6.71** |
| Boxxer median, 4 logs | 16.32 | 15.49 | 14.86 | **14.24** |
| Logs improved | - | **15/15** | 13/15 | **15/15** |

The important update to the earlier report is therefore that curve-normal
geometry remains useful, but it is no longer the strongest consistently safe
nuisance estimator on the magnitude pipeline. The previous iterative method
currently has the better aggregate result while also improving all 15 logs.

## Iterative tangent-weight follow-up

A controlled sweep of one, two, and four outer iterations against tangent sigma
ratios 1, 2, 5, 10, and 100 confirms that low-field tangent residuals contain
important information. At four iterations, ratio 1 improves all 15 logs and
reaches 9.12 mm mean weak-field RMSE. Ratio 2 reaches a lower 8.93 mm mean but
has one negligible +0.005 mm regression; ratio 100 is worse than the pipeline
at 13.19 mm.

The full design, per-log table, setup split, and raw-output locations are in
[`iterative_tangent_ablation/README.md`](iterative_tangent_ablation/README.md).

## Slope-derived covariance follow-up

The learned curve slope is not flat inside the 1500 mG gate: pooled P10/P50/P90
values are 6.5, 15.0, and 43.6 mG/mm. Propagating assumed travel uncertainty
through that slope therefore produces meaningfully adaptive tangent weights.

The best strictly consistent adaptive iterative setting reaches 9.10 mm mean
weak-field RMSE versus 9.12 mm for the existing iterative model, but transfers
less well to Boxxer. All-magnitude curve-normal observations become much safer
when their normal directions also receive a small amount of slope-linked
uncertainty. The complete sweep is in
[`slope_derived_covariance/README.md`](slope_derived_covariance/README.md).

## Why the result changed

The pipeline's legacy `mag/proj/...` cache names previously obscured which
scalar was being used. The current pipeline now writes filtered magnitude to
`mag/norm/corr/lpf`, and the experiment explicitly prefers that key while
retaining the old key only as a cache-compatibility fallback.

Changing the scalar signal changes both the baseline travel solution and the
encoder-free XYZ path:

1. Magnitude is binned in 100 mG bins.
2. Median XYZ is fit as a quadratic function of that scalar coordinate.
3. The magnitude-to-travel power model is inverted to create
   `travel -> expected XYZ`.
4. The acceleration-derived tangent determines which part of the XYZ residual
   is treated as a reliable nuisance-field observation.

Consequently, tangent blend, anisotropy ratio, and output alpha selected on the
projection pipeline cannot be assumed to transfer unchanged.

## Interpretation

The conservative result still supports the underlying observation: residuals
perpendicular to the locally measured travel-induced XYZ direction contain
some usable nuisance-field information. But the evidence is weaker than the
projection-era report suggested. Much of the larger gain requires aggressive
settings that fail on `0080`, while the safe fixed setting makes relatively
small corrections.

The hyperparameters above were selected using encoder metrics from this same
15-log cohort. These are development-set results, not an independent estimate
of performance on an unseen setup. The earlier alternating-time-block tangent
check has not yet been repeated for the magnitude-era setting.

## Reproduction

After regenerating each pipeline cache with the current `backend/pipeline.py`:

```bash
venv/bin/python tools/front/mag_nuisance/experiment_mag_nuisance_observability.py \
  --anchor-norms 1500 \
  --tangent-sigma-ratio 2 \
  --accel-blends 1.0 \
  --alphas 0.25 0.75 \
  --output-dir reports/front_mag_nuisance/observability/magnitude_selected
```

The conservative method in `magnitude_selected/metrics.csv` is
`accel_tangent_b1_normal_only_weak` at `alpha=0.25`; the comparison iterative
method is `previous_weak_iterative` at `alpha=0.75`. Fitted tangent and window
diagnostics are in `magnitude_selected/details.json`.

The old curve-normal column comes from a separate magnitude-pipeline rerun:

```bash
venv/bin/python tools/front/mag_nuisance/experiment_mag_nuisance_observability.py \
  --anchor-norms 1500 \
  --tangent-sigma-ratio 5 \
  --accel-blends 0.9 \
  --alphas 0.25 0.5 \
  --output-dir reports/front_mag_nuisance/observability/magnitude_current
```

Its reported method is `accel_tangent_b0.9_normal_only_weak` at `alpha=0.5` in
`magnitude_current/metrics.csv`.
