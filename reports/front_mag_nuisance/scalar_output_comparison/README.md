# Encoder-Free Corrected-Scalar Output Comparison

## Question

After estimating and subtracting the same body/world field vector, is it better
to infer travel from the full corrected XYZ path, or convert corrected XYZ back
to a scalar and reuse the existing accelerometer-trained scalar model?

No scalar model is retrained in this comparison. Encoder travel is used only
after all predictions are generated, for scoring.

## Compared outputs

- `xyz_path`: nearest point on the encoder-free expected XYZ path.
- `corrected_projection`: corrected XYZ projected onto the original strong-field
  direction, followed by the unchanged scalar model and absolute offset.
- `corrected_projection_or_norm`: the pipeline's projection fallback rule
  applied to corrected XYZ before the unchanged scalar model.
- `corrected_magnitude`: corrected XYZ norm followed by the unchanged scalar
  model.
- `raw_scalar_control`: unchanged, uncorrected scalar prediction. This isolates
  whether merely moving away from solved pipeline travel is beneficial.
- `xyz_projection_blend` and `xyz_magnitude_blend`: equal-weight averages of the
  corresponding travel proposals before applying alpha.

All proposals replace only samples allowed by the existing weak-field update
mask. Strong-field samples remain at the original pipeline solution.

## Results

Median weak-field RMSE at alpha 0.50:

| Fork | Pipeline | XYZ path | Corrected projection | Corrected magnitude | XYZ–magnitude blend |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fox 36 | 6.35 | 5.30 | 5.68 | 5.14 | **5.08** |
| Boxxer | 12.41 | **11.07** | 11.40 | 11.53 | 11.21 |

Direct XYZ improved all 15 logs at alpha 0.50. Corrected projection and
corrected magnitude each improved 13 of 15; their worst regressions were
+1.47 and +1.62 mm respectively. The raw scalar control improved only five,
confirming that the corrected field—not simply moving toward the scalar model—is
responsible for the gain.

The projection-or-norm fallback produced the same predictions as pure
projection in the applied weak-field region, so it added no value here.

## Conservative blend

At alpha 0.25:

| Method | Logs improved | Overall median weak RMSE | Worst per-log delta |
| --- | ---: | ---: | ---: |
| XYZ path | 15/15 | 6.96 mm | -0.12 mm |
| Corrected projection | 13/15 | 7.00 mm | +0.36 mm |
| Corrected magnitude | 14/15 | 7.07 mm | +0.42 mm |
| XYZ–projection blend | 15/15 | 6.98 mm | -0.06 mm |
| XYZ–magnitude blend | **15/15** | **6.87 mm** | **-0.23 mm** |

The 50/50 XYZ–magnitude blend is therefore an attractive conservative starting
output for an unseen setup. At alpha 0.50, direct XYZ has the better combined
cohort median, while the blend has the better Fox median.

## Interpretation

Corrected scalar output is viable without retraining. Magnitude often helps on
the Fox because the original scalar curve is already close to a magnitude
coordinate there. It fails on some Boxxer logs—especially `0080`—because
subtracting an ambient vector changes norm nonlinearly, and the original scalar
curve is not exactly calibrated for that corrected norm.

Full XYZ inversion retains direction information and is the most consistently
safe standalone method. A travel-space blend uses the scalar estimate when it
agrees while limiting the damage when scalarization loses important directional
information.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/experiment_unsupervised_mag_xyz.py \
  --output-dir reports/front_mag_nuisance/scalar_output_comparison
```

Detailed results are in `metrics.csv`, with fitted models and update diagnostics
in `details.json`.
