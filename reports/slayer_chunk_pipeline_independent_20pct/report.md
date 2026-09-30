# Slayer grown chunks as fully independent logs

Each of the seven grown raw-data ranges was run through the complete front pipeline as a separate log. No preprocessing, accelerometer alignment, magnetic baseline, zero-velocity points, magnetic reference, or model outputs were reused from the parent log. The fixed Slayer bike/fork geometry and normal pipeline configuration were retained. Dropout was not masked during training or solving; the masked metrics exclude exact zero-output samples only during evaluation.

Six of seven chunks completed. `log-0152-independent-c01` failed during accelerometer alignment because it contained only one candidate stationary pose and no non-collinear pose pairs; at least two are required.

## Aggregate results for the six successful chunks

| Stage | Evaluation | Pooled RMSE | Pooled chunk-centered RMSE |
|---|---:|---:|---:|
| Acceleration | Unmasked | 14.28 m/s² | 14.22 m/s² |
| Acceleration | Masked | 12.07 m/s² | 11.92 m/s² |
| Magnetic | Unmasked | 20.52 mm | 9.54 mm |
| Magnetic | Masked | 18.85 mm | 8.42 mm |
| First fusion | Unmasked | 20.44 mm | 6.77 mm |
| First fusion | Masked | 18.99 mm | 6.35 mm |
| Final fusion | Unmasked | 20.58 mm | 6.26 mm |
| Final fusion | Masked | 19.16 mm | 5.77 mm |

## Per-chunk final-fusion results

| Chunk | Dropout | Absolute RMSE | Centered RMSE | Masked centered RMSE | Mean error |
|---|---:|---:|---:|---:|---:|
| 0145 | 0.00% | 14.63 mm | 5.37 mm | 5.37 mm | +13.61 mm |
| 0147-A | 20.00% | 13.32 mm | 6.74 mm | 6.60 mm | -11.49 mm |
| 0147-B | 20.00% | 30.60 mm | 7.51 mm | 6.53 mm | +29.66 mm |
| 0151 | 20.00% | 16.13 mm | 5.72 mm | 4.97 mm | +15.09 mm |
| 0152-A | 20.00% | Failed | Failed | Failed | Failed |
| 0152-B | 20.00% | 10.06 mm | 4.55 mm | 4.19 mm | -8.97 mm |
| 0155 | 6.18% | 14.76 mm | 6.13 mm | 6.02 mm | -13.42 mm |

Four of the six successful chunks could not construct a supported encoder-derived absolute magnetic reference and fell back to a data-driven zero-travel reference. This accounts for much of the remaining absolute bias. Nuisance correction improved aggregate centered error from 6.77 to 6.26 mm unmasked and from 6.35 to 5.77 mm masked.
