# Why the Ambient Correction Initially Failed on the Fox Logs

> **Encoder-free follow-up:** The scalar field curve already learned from
> accelerometer/ZV chunks can parameterize the XYZ path without encoder travel.
> The resulting damped correction improved all 15 tested logs. See
> `../encoder_free_xyz/README.md`.

## Updated conclusion

The body/world nuisance-field hypothesis was not the main problem. The original
experiment gave the field solver a biased expected magnet field:

- travel was initialized from projection even on v2, where magnitude is the
  better initial signal; and
- the expected low-field XYZ path was forced to be a straight line.

The solver cannot distinguish an expected-magnet error from an additive
nuisance field. It therefore absorbed travel/model error into the body/world
states and subtracted the wrong vector. This mattered on both forks, but it was
especially visible on the higher-sensitivity Fox magnet geometry.

Using an encoder-calibrated magnitude estimate for initialization and a
quadratic low-field XYZ path makes the correction improve 10 of 11 Fox logs
(`0062` excluded) and all four Boxxer logs.

No production solver code was changed for this diagnostic. The quadratic model
and alternate initialization were tested as held-out experimental variants.

## Mounting-aware model

With the MMC5603 and IMU1 in the steer tube, and the magnets plus IMU2 on the
lower bridge, the useful measurement model is approximately

```text
mag = fork_magnet_xyz(travel) + body_fixed_field
      + rotate_world_field(gyro1 attitude) + noise
```

If the solver starts with imperfect travel or an imperfect fork-magnet path,
its measurement residual is

```text
mag - fork_model_xyz(predicted_travel)
    = true nuisance field
      + [true fork field - predicted fork field]
      + noise
```

The bracketed term is algebraically indistinguishable from nuisance field to
the body/world smoother. Gyro constraints determine how a field state may
evolve, but they cannot tell the solver that a smooth residual actually came
from the wrong travel or magnet-path model.

## What the diagnostics ruled out

The Fox and Boxxer groups had similar absolute field residuals and motion
excitation:

| Diagnostic, group median | Fox 36 | Boxxer |
| --- | ---: | ---: |
| Low-field XYZ residual, per-axis RMS | 118.5 mG | 120.2 mG |
| Estimated correction magnitude | 140.4 mG | 133.8 mG |
| Gyro RMS | 46.5 deg/s | 46.4 deg/s |
| Median angular span in a 5 s window | 38.1 deg | 37.1 deg |
| Median local body/world design condition number | 18.0 | 19.7 |

Thus the Fox logs do not fail because they lack rotations, have a badly
conditioned body/world split, or lack an ambient-sized residual. The user's
expectation that the same steer-tube pod should see similarly sized nuisance
fields is consistent with the data.

IMU2-relative features explained only a modest amount of held-out Fox residual
variance (roughly 8–11% at the median), while travel-dependent terms explained
about 28%. Fork flex or lower-bridge motion may contribute on individual logs,
but it was not the primary group-level failure.

## Why fork sensitivity changes the outcome

The fitted low-field XYZ slope norm was about 27.5 mG/mm on the Fox and
8.9 mG/mm on the Boxxer. The unexplained absolute field was similar, but its
travel equivalent was very different:

| Original line+projection diagnostic | Fox 36 | Boxxer |
| --- | ---: | ---: |
| Travel-equivalent residual before correction | 4.51 mm | 16.22 mm |
| Travel-equivalent residual after correction | 5.52 mm | 10.19 mm |
| Correction/residual correlation | 0.33 | 0.76 |
| Optimal scale on estimated correction | 0.31 | 1.10 |

On the Fox, the original correction estimate had roughly the right absolute mG
scale but poor alignment with the small travel-relevant component. It was
trying to remove a mixture of ambient field, initial-travel error, and
fork-model error. On the Boxxer, ambient error was a much larger travel error,
so the same field estimate had a favorable signal-to-model-error ratio.

## Targeted variants

Held-out weak-field RMSE, in millimeters:

| Group | Best raw magnitude/projection | Original line + projection init | Line + magnitude init | Quadratic XYZ + magnitude init |
| --- | ---: | ---: | ---: | ---: |
| Fox, 11 logs | 4.01 | 4.51 | 3.50 | **2.85** |
| Boxxer, 4 logs | 8.93 | 7.61 | 6.58 | **6.01** |

Changing only the initialization removes most of the Fox regression. Allowing
gentle XYZ curvature provides another material improvement. A 5 mm binned XYZ
path produced similar Fox results (2.92 mm median) but was worse on the Boxxer
(6.84 mm), so the quadratic is the better low-variance starting model.

The quadratic+magnitude variant improved 10 of 11 Fox logs relative to each
log's best raw signal. `0072_184` was the only regression, from 4.01 to
4.18 mm. It improved all four Boxxer logs.

With the revised model, the estimated correction once again behaves like a
real nuisance estimate:

| Quadratic+magnitude diagnostic | Fox 36 | Boxxer |
| --- | ---: | ---: |
| Travel-equivalent residual before correction | 4.15 mm | 14.41 mm |
| Travel-equivalent residual after correction | 3.01 mm | 7.17 mm |
| Correction/residual correlation | 0.76 | 0.86 |

## Recommended next implementation

1. Use magnitude rather than projection as the initial v2 travel estimate.
2. Replace the low-field linear XYZ model with a constrained three-coefficient
   quadratic per axis, still fitted to 5 mm bin medians.
3. Retain the current body/world weights and weak-only correction guard.
4. Keep a fallback to the raw estimate if the corrected and raw estimates
   disagree implausibly or the quadratic fit is poorly conditioned.
5. Re-run leave-one-log-out tests after implementation; the present alternating
   blocks validate time generalization within each log, not setup calibration
   transfer to a completely unseen fork.

The earlier 15 mG/mm gate should not be treated as the root fix. Sensitivity is
useful for estimating expected benefit and setting safeguards, but the apparent
clean separation was largely caused by the original line+projection model
being a worse nuisance estimator on the Fox cohort.
