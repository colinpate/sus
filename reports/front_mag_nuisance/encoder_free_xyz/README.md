# Encoder-Free XYZ Ambient-Correction Experiment

> **Output-mapping follow-up:** Reusing the unchanged scalar model on corrected
> projection/magnitude works on many logs but is less uniformly safe than XYZ
> inversion. A conservative XYZ–magnitude travel blend improves all 15 logs.
> See `../scalar_output_comparison/README.md`.

## Result

A useful XYZ path can be learned without encoder travel by parameterizing it
with the front pipeline's existing accelerometer-trained scalar magnetic
coordinate. A single four-step field solve followed by a damped update improved
all 15 tested logs (`0062` excluded).

The recommended experimental starting point is a 25% update for maximum
conservatism or 50% for the larger observed gain. Do not iterate the entire
self-calibration loop until numerical convergence; some logs drift toward a
wrong but self-consistent field/travel solution.

## Encoder audit

The current front pipeline is a valid encoder-free bootstrap:

- The primary magnetic projection direction is the mean direction of strong
  raw MMC samples.
- Magnetic zero-velocity points come from extrema in that projected signal.
- The scalar power curve is trained from relative displacement obtained by
  double-integrating short accelerometer chunks around zero-velocity points.
- Its absolute reference point comes from selected still-then-bump
  accelerometer integrations.
- The final travel solver combines the scalar magnetic prediction,
  accelerometer dynamics, and zero-velocity constraints.

The pipeline passes encoder travel into a few steps for plots and printed error
diagnostics, but it does not affect the fitted scalar coefficients, reference
point, or solved travel. This experiment uses `travel` and `boring_mask` only
after prediction generation, to score the frozen output.

## Method

1. Read the existing encoder-free scalar model coefficients and absolute
   offset.
2. Invert that scalar curve on a 0–210 mm grid to obtain the scalar field
   coordinate expected at each travel value.
3. Bin raw scalar field in 100 mG bins and fit a quadratic mapping from scalar
   field to MMC XYZ. This step uses magnetometer samples only, not predicted or
   encoder travel labels.
4. Compose the two models to get an encoder-free `travel -> XYZ` path.
5. Initialize travel with the existing encoder-free pipeline solution and run
   the body/world field smoother for four inner iterations.
6. Apply a trust-region update

   ```text
   travel_out = travel_pipeline
                + alpha * (travel_xyz_corrected - travel_pipeline)
   ```

   only where the existing weak-field guards allow the XYZ solver to update.

## Encoder-blind batch scoring results

The model may inspect all magnetometer, gyro, and accelerometer samples in the
recording because the intended solver is full-log/batch. Encoder travel is held
out from the entire fitting and prediction path, then revealed only for these
metrics.

Median per-log weak-field RMSE:

| Fork | Existing encoder-free pipeline | alpha 0.10 | alpha 0.25 | alpha 0.50 | alpha 0.75 | Full update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fox 36, 11 logs | 6.35 | 6.03 | 5.72 | **5.30** | 5.31 | 5.69 |
| Boxxer, 4 logs | 12.41 | 12.07 | 11.62 | 11.07 | **10.80** | 10.81 |

At alpha 0.50, every individual log improved. The changes ranged from
-0.07 mm on `0073_185` to -1.78 mm on `0081`. Alpha 0.25 also improved every
log and had a smaller maximum step, making it the safer unseen-setup default.

Median all-field RMSE at alpha 0.50 changed from 6.29 to 5.84 mm on the Fox and
from 11.67 to 10.56 mm on the Boxxer. Strong-field predictions are unchanged;
the gain comes from weak-field samples.

This remains below the per-log encoder-trained quadratic result (2.85 mm Fox,
6.01 mm Boxxer weak-field medians), but unlike that earlier result the curve,
correction, and prediction here receive no encoder travel.

## Rejected variants and why

### Fit XYZ directly against predicted travel

This creates a circular target: the predicted travel was already derived from
the same magnetometer. Its low-field errors become a self-consistent XYZ path.
Direct quadratic fitting was inconsistent, and unrestricted iterative updates
often amplified error.

### Fit only strong/anchor samples and extrapolate

The strong region did not constrain a free per-axis polynomial well enough at
the low-field end. Quadratic extrapolation frequently had the wrong direction
or scale. Adding samples within 0.25 seconds of an anchor did not reliably fix
the curve because those samples still carried magnet-derived pseudo-labels.

### Retrain corrected magnitude and rerun the entire scalar pipeline

This helped isolated logs but was unstable across the cohort. Uncertainty in
the extrapolated XYZ path contaminated the new scalar curve's absolute
reference and scale.

### Temporal-distance gate

Pipeline error is indeed highest approximately 0.25–2 seconds after leaving an
anchor region. Applying correction only in that interval improved all logs,
but less than the damped global weak-field update. Temporal distance is useful
as a diagnostic or optional safety feature, not as the primary training-label
selector.

## Iteration and identifiability

The four inner field/travel iterations converge toward a fixed XYZ correction.
Repeating the full outer update is different: it changes the travel point about
which body/world field and magnet-path error are separated. Boxxer logs often
continued improving, while several Fox logs bottomed out after one or two outer
steps and then drifted.

Consequently, numerical convergence is not evidence that the physical
decomposition is correct. Until an independent objective can choose outer
iterations, use one bounded update.

## Production-oriented safeguards

- Start with alpha 0.25 on an unseen setup; promote toward 0.50 only after
  independent validation.
- Require positive, finite scalar-model scale and power.
- Require enough populated scalar bins; this cohort had 16–76, median 59.
- Retain both predicted- and measured-field weak guards.
- Clamp or reject unusually large proposed corrections. Proposed update RMS in
  this cohort was 1.64–7.57 mm.
- Run only one outer correction pass initially.
- Preserve the uncorrected pipeline result as a fallback.
- Treat known mechanical stroke as setup metadata; XYZ cannot determine an
  absolute millimeter scale without the accelerometer-derived scalar model or
  another independent constraint.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/experiment_unsupervised_mag_xyz.py \
  --output-dir reports/front_mag_nuisance/encoder_free_xyz
```

`metrics.csv` contains per-log results for every alpha, `aggregate.csv` has
fork-level medians, and `details.json` records the fitted XYZ parameters and
update diagnostics.
