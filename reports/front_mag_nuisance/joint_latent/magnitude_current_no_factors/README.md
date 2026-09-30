# Joint latent solver on magnitude-era caches

This rerun uses the 15 current pod-v2 magnitude-pipeline caches. Direct
acceleration factors are disabled because the original ablation found that they
changed per-log RMSE by at most 0.0021 mm. Encoder travel is used only after
prediction generation for scoring.

## Weak-field result

| Method | Alpha | Mean RMSE | Median RMSE | Fox 36 median | Boxxer median | Improved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pipeline | — | 10.123 | 8.078 | 7.789 | 16.317 | — |
| Iterative body/world | 0.75 | **9.124** | 7.643 | **6.713** | **14.236** | 15/15 |
| Joint corrected XYZ, aligned | 0.50 | 9.311 | 7.400 | 7.094 | 15.102 | 15/15 |
| Joint corrected XYZ, aligned | 0.75 | 9.271 | **7.233** | 6.764 | 14.722 | 14/15 |

The safer 50% joint output improves every log but does not beat the simpler
iterative body/world method. The 75% joint output has a lower median but
regresses `log-0073_185` by 0.21 mm.

This supersedes the old joint solver as a current production candidate. The
older projection-era result remains useful evidence about the model family, but
its 7.707 mm mean is not directly comparable with this magnitude-era baseline.

## Reproduction

```bash
venv/bin/python tools/front/mag_nuisance/experiment_joint_mag_accel.py \
  --disable-accel-factors \
  --output-dir reports/front_mag_nuisance/joint_latent/magnitude_current_no_factors
```

See `metrics.csv`, `aggregate.csv`, and `details.json` for the full output.
