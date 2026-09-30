# Front magnetic-nuisance investigation

## Executive summary

The primary magnetometer contains a real, slowly varying additive nuisance
field with both body-fixed and gyro-transported world-fixed structure. Removing
an estimated nuisance vector improves travel, especially when the magnet field
is weak, but the dominant practical difficulty is identifiability: an error in
travel or in the learned `travel -> XYZ` curve looks exactly like an additive
magnetic field to the smoother.

The best-supported current starting point is the simple iterative body/world
model:

1. Build an encoder-blind quadratic XYZ path from the existing
   accelerometer-trained scalar magnitude model.
2. Integrate gyro1 at its 100 Hz filtered rate, sample those rotations at 10 Hz,
   and estimate body-fixed and gyro-transported world-fixed fields there.
3. Update the field state only where predicted magnet norm is at most 1500 mG.
4. Subtract the sum of the two fields and infer travel from corrected XYZ.
5. Apply the travel correction only where both predicted and measured norm are
   at most 1500 mG.
6. Repeat four field/travel iterations and blend 75% toward corrected travel.

On the current 15-log magnitude-era pod-v2 cohort, mean weak-field RMSE falls
from 10.12 to 8.94 mm, with all 15 logs improving. This is a development-set
result, not yet validation on a completely unseen setup. The earlier version
that decimated gyro samples directly reached 9.12 mm.

### Pipeline integration status

The 10 Hz state solve and full-rate application are implemented in
`backend/mag_nuisance_core.py` and `backend/mag_nuisance.py`. The front pipeline
now emits both a delta-lifted full-rate solution and a full-rate corrected
magnetic travel observation, then runs a second fusion pass without replacing
the original `travel/solved`. Gyro1 is integrated at its full filtered rate;
the world field is interpolated in the fixed reference frame and transported
back to the body frame at every full-rate sample.

Across the same 15 logs, the full-rate second fusion pass reduces mean raw
overall RMSE from 9.353 to 8.430 mm and centered RMSE from 5.023 to 4.160 mm.
Centered overall RMSE and standard travel-bin RMSE improve on all 15 logs. A
5 Hz high-pass check finds essentially unchanged high-frequency encoder error:
1.601 mm baseline versus 1.598 mm after refusion.

## Comparison rules

Several reports were generated under different pipeline eras and evaluation
protocols. Raw RMSE should only be compared within a row group below:

- **Current magnitude era:** caches regenerated on 2026-08-30/31 after pod-v2
  scalar travel changed from projection to MMC5603 magnitude. This is the main
  decision set.
- **Earlier projection era:** same broad 15-log cohort, but a different scalar
  signal and baseline. These results are useful algorithm evidence, not current
  rankings.
- **Encoder-supervised held-out diagnostics:** encoder labels train alternating
  calibration blocks. These test model plausibility and an approximate ceiling;
  they are not deployable calibration procedures.

## Current magnitude-era results

All methods below use the same 15 logs and 10.123 mm cohort mean weak-field
pipeline baseline. `Improved` is a strict per-log comparison.

| Method | Mean RMSE | Reduction | Improved | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Four-iteration isotropic body/world, source-rate gyro, alpha 0.75 | **8.940** | **11.7%** | **15/15** | Current pipeline default; 10 Hz field states use gyro rotations integrated at 100 Hz. |
| Four-iteration isotropic body/world, decimated gyro, alpha 0.75 | 9.124 | 9.9% | **15/15** | Previous stage-one behavior and multirate control. |
| One-iteration slope-adaptive, 5 mm travel sigma, alpha 0.75 | **9.100** | 10.1% | **15/15** | Essentially tied overall; better Fox, worse Boxxer. Needs unseen validation. |
| All-sample slope-adaptive curve normal, 10% normal leakage, alpha 0.50 | 9.240 | 8.7% | **15/15** | Safe soft high-slope gate; more complex observation policy without a net win. |
| Current joint latent corrected XYZ, aligned, alpha 0.50 | 9.311 | 8.0% | **15/15** | Does not justify its added complexity on current caches. |
| Conservative acceleration-tangent curve normal, alpha 0.25 | 9.726 | 3.9% | **15/15** | Consistent but modest gain. |
| Fixed tangent ratio 2, four iterations, alpha 0.75 | **8.926** | **11.8%** | 14/15 | Best aggressive mean; remaining regression is only +0.005 mm, but tuned on this cohort. |

Full-rate gyro integration accounts for nearly all of the new gain. Explicit
2--4 Hz residual low-pass filters and 100 ms mean/median aggregation were tied
or slightly worse at the original weights. Increasing magnetometer observation
sigma improved Fox further but progressively hurt Boxxer, so the untuned 40 mG
value remains the production default pending unseen validation.

Detailed current results:

- [Full-rate correction and second-fusion comparison](observability/full_rate_correction/README.md)
- [Standard 10 Hz overall and travel-bin metrics](observability/standard_correction_metrics/README.md)
- [Magnitude-era observability and per-log comparison](observability/README.md)
- [Four-iteration nuisance-field magnitude and direction dynamics](field_dynamics/README.md)
- [Iteration versus tangent-weight ablation](observability/iterative_tangent_ablation/README.md)
- [Slope-derived covariance and high-slope normal weighting](observability/slope_derived_covariance/README.md)
- [Multirate gyro and residual-observation study](observability/multirate_observations/README.md)
- [Current magnitude-era joint rerun](joint_latent/magnitude_current_no_factors/README.md)

## Algorithms and experiments

### Body/world field model

The useful measurement model is

```text
mag_xyz = fork_magnet_xyz(travel)
          + body_fixed_field
          + gyro_rotated_world_field
          + residual
```

The six-state Kalman/RTS smoother lets the body component random-walk slowly
and transports the world component through gyro1 rotation with its own small
random walk. The two components are not individually well identified; their
sum is the useful correction.

Initial supervised tests showed that the full body+world model beat either
component alone on the Boxxer logs: median weak-field RMSE was 7.61 mm versus
9.93 mm body-only and 9.23 mm world-only. Windowed stationarity tests also found
that a world-only model reduced held-out magnetic residual RMS by only 2–10%,
while the combined body+world model reduced it by 22–44% and improved 88–97% of
eligible windows. A purely static world-field hypothesis is therefore too
simple.

Reports: [stationarity diagnostics](ambient_field_stationarity/README.md) and
[supervised continuous correction](supervised_body_world/README.md).

### Encoder-supervised path diagnostics

The early body/world solver worked very well when given an encoder-calibrated
quadratic XYZ path and magnitude-based initialization. Held-out group medians
improved from 4.01 to 2.85 mm on Fox 36 and from 8.93 to 6.01 mm on Boxxer. This
was important evidence that the nuisance model itself was viable.

The original straight-line/projection version failed on many Fox logs because
the smoother absorbed expected-path and initial-travel error into its nuisance
states. The Fox and Boxxer groups actually had similar absolute residual fields,
gyro excitation, and body/world conditioning. Correcting the expected XYZ model
and initialization fixed most of the apparent setup failure.

Reports: [failure diagnosis](failure_diagnosis/README.md),
[original v2 generalization check](v2_generalization/README.md), and
[supervised continuous result](supervised_body_world/README.md).

### Encoder-free XYZ bootstrap

A deployable XYZ path can be learned without encoder travel:

1. Invert the existing accelerometer-trained scalar magnet-to-travel model.
2. Bin measured scalar field in 100 mG bins.
3. Fit median MMC XYZ as a quadratic function of that scalar coordinate.
4. Compose the models to obtain `travel -> expected XYZ`.

On the earlier projection-era caches, a damped four-step correction improved
all 15 logs; at alpha 0.50, Fox median weak RMSE changed from 6.35 to 5.30 mm
and Boxxer from 12.41 to 11.07 mm. Directly fitting XYZ against predicted travel,
free strong-region extrapolation, retraining the scalar model after correction,
and temporal-distance-only gates were less stable.

Reports: [encoder-free XYZ bootstrap](encoder_free_xyz/README.md) and
[corrected scalar output comparison](scalar_output_comparison/README.md).

### Curve-normal observations

For a local travel error `delta_t`, the field residual is approximately

```text
mag_xyz - f(t) = f'(t) * delta_t + nuisance.
```

The travel-error term lies along the XYZ-curve tangent, so perpendicular
residuals provide cleaner first-order nuisance observations. This geometry is
real—the conservative magnitude-era curve-normal method improved all 15
logs—but normal-only information is insufficient. With four iterations,
reducing tangent weight to 0.01% raised mean RMSE from the 10.12 mm baseline to
13.19 mm. Full tangent weight reached 9.12 mm and improved every log.

The practical conclusion is not to discard the tangent. Use it where the
travel/curve ambiguity is manageable, or represent that ambiguity with
covariance.

Report: [observability study](observability/README.md).

### Slope-derived covariance

Propagating assumed travel uncertainty is the principled tangent covariance:

```text
R = sigma_mag² I + J sigma_travel² J.T,
J = df/dt.
```

The learned slope is not flat below 1500 mG. Pooled equal-travel P10/P50/P90
values are 6.5, 15.0, and 43.6 mG/mm, so adaptive weighting changes materially
inside the existing gate. A 5 mm travel sigma gives roughly 60%, 22%, and 3%
tangent weight at those slopes.

For observations above 1500 mG, tangent uncertainty alone was unsafe because
high-field normal residuals also contain model/tangent-direction error. Adding
only 10% of the slope-derived uncertainty to both normal directions changed a
worst +3.79 mm regression into improvement on every log. This acts as a soft
high-slope gate.

Report: [slope-derived covariance](observability/slope_derived_covariance/README.md).

### Joint latent travel solver

The sparse joint solver estimates travel and both nuisance fields together,
with strong-region travel anchors, continuity priors, robust magnetic factors,
and optional accelerometer displacement factors. On projection-era caches it
reduced mean weak RMSE from 8.928 to 7.707 mm and improved all 15 logs at a 50%
output blend.

Direct acceleration factors were not responsible: disabling them changed any
per-log result by at most 0.0021 mm. On current magnitude-era caches, the safe
joint corrected-XYZ output reaches 9.311 mm versus 9.124 mm for the simpler
iterative smoother. The joint architecture remains interesting, but it is not
the current winner.

Reports: [original joint experiment](joint_latent/README.md) and
[current-cache rerun](joint_latent/magnitude_current_no_factors/README.md).

## What we learned about the nuisance field

### Magnitude and structure

- Fox and Boxxer had similar low-field residuals: approximately 118–120 mG
  per-axis RMS, with estimated correction magnitudes around 134–140 mG.
- They also had similar gyro excitation and local body/world conditioning.
  Setup-to-setup performance differences were not caused primarily by missing
  rotation or an absent ambient-sized signal.
- A world-fixed component is present, but a body-fixed/very-slow component is
  also required. Real environmental change, gyro drift, curve error, and
  unmodeled mechanics are absorbed into the allowed random walks.
- The body/world split is gauge-like and should not be overinterpreted. Validate
  and consume `body + world`, not the individual vectors.

### Why the same field causes different travel error

Travel-equivalent error is approximately field error divided by local curve
slope. The revised supervised diagnostic measured typical low-field slopes
around 27.5 mG/mm on Fox and 8.9 mG/mm on Boxxer. Similar nuisance fields thus
represented about 4.15 mm before correction on Fox but 14.41 mm on Boxxer.
After correction those figures fell to 3.01 and 7.17 mm respectively.

This is why low-sensitivity Boxxer logs can gain more from nuisance correction,
even though their absolute magnetic environment is not obviously worse.

### Contribution to total low-end travel error

The current magnitude-era, below-30-mm decomposition gives a centered median
scalar-model RMSE of 9.16 mm:

- stable learned-curve disagreement: 6.63 mm RMS, about 55% of MSE;
- within-bin scatter: 5.78 mm RMS, about 45% of MSE.

A five-time-block decomposition attributes median MSE shares of 57% to stable
curve disagreement, 7% to block drift, and 37% to within-block scatter.
Scatter includes time-varying nuisance field, sensor noise, and state dependence,
so it is an upper bound on nuisance-field error rather than a pure measurement.
Conversely, very slow or travel-correlated nuisance can be absorbed into the
whole-log curve term.

The split is highly log-dependent. `0081` is approximately 94% time-varying;
`0080` is 61% time-varying. `0073_185` is about 93% stable curve. `0078-valid`
and `0058` are also nuisance/scatter-heavy, while `0046`, `0049`, `0071_183`,
and `0072_184` are primarily curve-limited.

Current correction recovers about 1.0 mm of cohort mean weak-field RMSE, or
roughly 10%. This is smaller than the scatter share because not all scatter is
magnetic nuisance and because the encoder-blind field/travel decomposition is
only partially observable.

Report: [low-end error decomposition](low_end_error_decomposition/README.md).

## Other durable findings

- **Orientation:** for both verified pod revisions,
  `gyro_xyz = [[0,0,1],[0,-1,0],[1,0,0]] @ recorded_mmc_xyz`. Thus MMC +X maps
  to gyro +Z, MMC +Y to gyro -Y, and MMC +Z to gyro +X.
- **Scalar choice:** magnitude is the better current pod-v2 scalar and improved
  nearly all v2 logs. Older v1 logs favored projection, so this should remain
  setup metadata rather than a universal sensor rule.
- **Weak-field gating:** estimating and applying correction only below 1500 mG
  is the safest simple policy. Strong-field samples become usable only with
  careful tangent and normal model uncertainty; travel application should still
  remain weak-only until independently validated.
- **Iteration:** relinearization helps, but convergence is not proof of physical
  correctness. Several logs drift toward a wrong self-consistent travel/field
  solution. Bound the number of iterations and damp the final update.
- **Acceleration:** short integrated acceleration windows were useful for an
  alternate tangent estimate, but direct relative-travel factors added no
  measurable value to the joint solver under the tested selection rules.
- **Secondary LIS3MDL:** it showed some common-mode promise, but railing,
  moving-bridge mounting, and weak axis-level agreement prevented a validated
  differential correction. Current production candidates use only the primary
  MMC5603 and gyro1.
- **Gyro bias:** explicit whole-log gyro-bias correction did not improve the
  early stationarity tests. The successful batch models instead allow a small
  world-field/model random walk.

## What did not work reliably

- Treating the ambient field as only a static world vector.
- Fitting expected XYZ directly against poor low-field predicted travel.
- Extrapolating a free polynomial from strong-field samples into weak field.
- Treating all XYZ residual as nuisance when expected travel/path is biased.
- Removing nearly all tangent information.
- Applying aggressive correction at strong field without model uncertainty.
- Retraining the scalar travel curve from corrected magnitude in the same
  self-calibration loop.
- Running outer refinement until numerical convergence.
- Using direct accelerometer factors with the presently available windows.

## Recommended next steps

1. Validate four-iteration isotropic correction at alpha 0.50 and 0.75 on a
   completely unseen fork/magnet setup before selecting production weights.
2. Expose a per-sample travel covariance from the fusion solver. Use it in
   `R = R_mag + J P_travel J.T` instead of a fixed assumed travel sigma.
3. Add a learned or conservative normal-direction model covariance, especially
   above 1500 mG; the 10% slope-leakage experiment is a useful starting form.
4. Preserve a raw-pipeline fallback and reject poorly conditioned XYZ curves,
   large proposed corrections, or insufficient scalar-bin coverage.
5. If an online implementation is required, replace the full-log RTS pass with
   a causal filter/fixed-lag smoother and explicitly test boundary/latency costs.
6. Improve independent acceleration constraints before revisiting the larger
   joint latent solver.

## Directory map

| Directory | Contents/status |
| --- | --- |
| `ambient_field_stationarity/` | Early supervised body/world/world-only stationarity tests and plots. |
| `supervised_body_world/` | Held-out encoder-calibrated continuous correction; model ceiling and weight checks. |
| `failure_diagnosis/` | Why the initial Fox generalization failed. |
| `v2_generalization/` | Superseded straight-line/projection generalization result, retained for provenance. |
| `encoder_free_xyz/` | Encoder-free scalar-parameterized XYZ bootstrap; earlier pipeline era. |
| `scalar_output_comparison/` | Corrected XYZ versus projection/magnitude scalarization. |
| `observability/` | Current magnitude-era curve-normal, tangent, iteration, and slope-covariance studies. |
| `joint_latent/` | Projection-era joint experiment plus current magnitude-era rerun. |
| `low_end_error_decomposition/` | Current supervised error-budget diagnostic. |
| `archive/` | Superseded raw continuous-solver sweep artifacts. |

The corresponding code is grouped in `tools/front/mag_nuisance/`.

Generated `details.json` files retain the output-directory strings recorded at
the time of each original run. Older entries may therefore show their former
pre-consolidation paths; the files themselves and all reproduction commands are
now organized under this directory.
