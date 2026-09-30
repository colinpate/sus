# Joint latent-travel / magnetic-nuisance experiment

> **Magnitude-era follow-up:** rerunning the no-acceleration-factor variant on
> current magnitude caches gives 9.31 mm mean weak-field RMSE for the safest
> joint output, versus 9.12 mm for the simpler iterative correction. See
> [`magnitude_current_no_factors/README.md`](magnitude_current_no_factors/README.md).
> The results below use the earlier projection-era baseline.

## Question

Can short, encoder-free accelerometer displacement windows break the ambiguity
between travel along the magnetometer XYZ curve and a slowly changing additive
magnetic field?

The encoder is not used to construct the XYZ model, select samples, fit the
nuisance field, or solve latent travel. It is loaded only after all predictions
have been produced and is used for the metrics below.

## Model tested

The 10 Hz state is latent travel `x`, a body-fixed field `b`, and a world-fixed
field expressed in the body frame `w`. The sparse Gauss-Newton objective uses:

- magnet residual `mag_xyz - xyz_model(x) - b - w`, with more uncertainty along
  the locally estimated magnet-curve tangent;
- a random walk for `b`;
- gyro transport plus a small random walk for `w`;
- soft 1 mm travel anchors where the modeled field is strong and the scalar and
  fused travel estimates agree;
- a 0.25 mm/sqrt(s) random walk on the correction `x - initial_travel`, which
  prevents an unobservable travel offset from accumulating between anchors;
- optional relative-travel factors from zero-velocity-centered double
  integrations of projected acceleration.

LSMR receives a unit-column-scaled sparse system. With a 2000-iteration limit,
all 60 linear solves stop on the `1e-5` tolerance (257-1532 iterations; 257-603
on the final Gauss-Newton step), rather than depending on the iteration cap.

The useful output is not latent `x` itself. It is travel inferred from
`mag_xyz - b - w`, blended 50% with the original fused travel, and applied only
below the 1500 mG measured-field threshold.

## Main result

Weak-field RMSE across the same 15-log Fox 36 / Boxxer cohort:

| Method | Mean RMSE (mm) | Logs improved vs pipeline |
| --- | ---: | ---: |
| Pipeline | 8.928 | - |
| Previous iterative body/world correction, 50% | 8.047 | 15/15 |
| Accel-tangent curve-normal correction, 50% | 7.956 | 15/15 |
| Joint solver corrected XYZ, 50% | **7.707** | **15/15** |

The joint corrected-XYZ result lowers mean weak-field RMSE by 1.221 mm (13.7%)
from the pipeline and by 0.249 mm (3.1%) from the curve-normal method. Median
RMSE changes from 6.346 to 5.498 mm on the Fox 36 logs and from 12.407 to
10.850 mm on the Boxxer logs.

| Log | Fork | Pipeline | Curve normal | Joint corrected XYZ | Joint gain vs pipeline |
| --- | --- | ---: | ---: | ---: | ---: |
| log-0056 | Fox 36 | 6.250 | 4.938 | 4.865 | 1.385 |
| log-0063 | Fox 36 | 5.758 | 5.193 | 5.333 | 0.425 |
| log-0046 | Fox 36 | 5.559 | 4.989 | 5.498 | 0.061 |
| log-0048 | Fox 36 | 5.602 | 4.426 | 4.127 | 1.475 |
| log-0049 | Fox 36 | 7.240 | 6.308 | 6.554 | 0.686 |
| log-0054 | Fox 36 | 7.566 | 6.918 | 6.349 | 1.218 |
| log-0055 | Fox 36 | 12.705 | 12.186 | 12.262 | 0.444 |
| log-0058 | Fox 36 | 4.109 | 3.189 | 3.179 | 0.930 |
| log-0071_183 | Fox 36 | 8.888 | 8.082 | 7.688 | 1.200 |
| log-0072_184 | Fox 36 | 6.346 | 4.558 | 3.459 | 2.887 |
| log-0073_185 | Fox 36 | 13.840 | 13.816 | 13.567 | 0.273 |
| log-0078-valid | Boxxer | 9.591 | 8.745 | 8.230 | 1.361 |
| log-0079 | Boxxer | 9.305 | 8.043 | 8.940 | 0.365 |
| log-0080-valid | Boxxer | 15.940 | 14.970 | 12.759 | 3.180 |
| log-0081 | Boxxer | 15.222 | 12.983 | 12.802 | 2.420 |

A 75% blend has a slightly better cohort mean of 7.602 mm, but regresses two
logs. The 50% blend is the safer starting point.

## Acceleration-factor ablation

The direct acceleration factors did not explain the improvement. Repeating the
full cohort with the factors omitted changes per-log RMSE by at most 0.0021 mm,
which is not practically meaningful. Longer strict windows are too sparse
(often zero accepted windows); loosening their correlation and rotation filters
supplies more factors but slightly worsens several difficult logs.

This means the current acceleration integrations do not yet add useful
observability to the joint solve. The improvement comes from the continuously
estimated body/world nuisance field, strong-region anchors, correction
continuity, and corrected-XYZ inference. The acceleration-derived tangent from
the earlier curve-normal experiment remains part of the magnetic covariance
model, but the new direct displacement residuals can be omitted for now.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/experiment_joint_mag_accel.py \
  --output-dir reports/front_mag_nuisance/joint_latent/factors

venv/bin/python tools/front/mag_nuisance/experiment_joint_mag_accel.py \
  --disable-accel-factors \
  --output-dir reports/front_mag_nuisance/joint_latent/no_factors
```

Raw per-log metrics and solver diagnostics are in the `factors/` and
`no_factors/` subdirectories.
