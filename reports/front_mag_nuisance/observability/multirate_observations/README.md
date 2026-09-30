# Multirate nuisance-observation experiment

All results use the encoder-blind quadratic XYZ path, four field/travel
iterations, the 1500 mG update/application gate, and the configured output
blend. Encoder travel is used only for these metrics.

| Method | Mean weak RMSE | Median | Mean delta | Improved | Worst delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `point_full_gyro` | 8.940 | 7.656 | -1.183 | 15/15 | -0.064 |
| `point_multirate_full_gyro` | 8.940 | 7.656 | -1.183 | 15/15 | -0.064 |
| `median100ms_full_gyro` | 8.955 | 7.629 | -1.168 | 15/15 | -0.047 |
| `lowpass3_full_gyro` | 8.966 | 7.627 | -1.157 | 14/15 | +0.367 |
| `lowpass2_full_gyro` | 8.978 | 7.604 | -1.144 | 14/15 | +0.304 |
| `lowpass4_full_gyro` | 8.999 | 7.638 | -1.123 | 14/15 | +0.346 |
| `mean100ms_full_gyro` | 9.011 | 7.637 | -1.111 | 14/15 | +0.262 |
| `point_decimated_gyro` | 9.124 | 7.643 | -0.999 | 15/15 | -0.013 |
| `pipeline` | 10.123 | 8.078 | +0.000 | 0/15 | +0.000 |

Per-log and all-region measurements are in `metrics.csv`; state and
iteration diagnostics are in `details.json`.

## Findings

Integrating gyro1 at 100 Hz and sampling the resulting rotations at the 10 Hz
field nodes is the clearest improvement. With the original 40 mG observation
sigma, mean weak RMSE changes from 9.124 mm for decimated gyro to 8.940 mm for
source-rate gyro integration. The source-rate result still improves all 15
logs relative to the 10.123 mm pipeline baseline.

Anti-aliasing the full-rate residual did not improve the aggregate result at
the original weights. A centered 100 ms median was essentially tied at 8.955
mm and improved all logs relative to the pipeline, but the true 2--4 Hz
low-pass and 100 ms mean variants each regressed one log. A point-sampled
multirate control exactly matched the simpler point implementation, proving
that full-rate travel interpolation was not the source of this difference.

The measurement-sigma sweep for point observations was:

| Mag sigma | All mean | Fox mean | Boxxer mean | Improved vs pipeline |
| ---: | ---: | ---: | ---: | ---: |
| 40 mG | 8.940 | 7.312 | 13.417 | 15/15 |
| 80 mG | 8.747 | 7.049 | 13.417 | 15/15 |
| 120 mG | 8.706 | 6.940 | 13.561 | 15/15 |
| 160 mG | 8.702 | 6.873 | 13.731 | 15/15 |
| 240 mG | 8.721 | 6.789 | 14.035 | 15/15 |

Higher sigma increasingly favors Fox while regressing the Boxxer mean. Because
these values were swept on the development cohort and the problematic Boxxer
setup is important, the pipeline retains the original 40 mG value for now. It
does adopt source-rate gyro integration, which is physically correct and can be
disabled through the step configuration for controlled comparisons.
