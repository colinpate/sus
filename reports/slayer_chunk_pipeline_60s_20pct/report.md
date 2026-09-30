# Slayer 60-active-second chunk pipeline experiment

Each chunk contains at least 60 seconds of activity and less than 20% raw core-sensor zero-output dropout. Full-log preprocessing, mounting calibration, magnetic baseline, and absolute reference are reused. The magnetic model, first travel solver, nuisance correction, and second travel solver run independently per chunk. Dropout masks are not used during training or solving; they are applied only to the masked evaluation columns.

## Aggregate results

| Stage | Units | Evaluation | Chunk-mean RMSE | Chunk-mean centered RMSE | Log-balanced RMSE | Pooled RMSE | Pooled centered RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| accel | m/s^2 | unmasked | 13.68 | 13.63 | 13.97 | 14.07 | 14.02 |
| accel | m/s^2 | masked | 11.94 | 11.82 | 11.64 | 12.03 | 11.91 |
| mag | mm | unmasked | 12.92 | 7.29 | 12.17 | 14.42 | 7.94 |
| mag | mm | masked | 12.19 | 6.51 | 11.58 | 12.92 | 6.65 |
| fusion1 | mm | unmasked | 11.59 | 5.70 | 11.21 | 13.43 | 5.95 |
| fusion1 | mm | masked | 11.08 | 5.41 | 10.79 | 12.16 | 5.45 |
| fusion2 | mm | unmasked | 11.47 | 5.41 | 11.10 | 13.40 | 5.72 |
| fusion2 | mm | masked | 10.96 | 5.07 | 10.67 | 12.12 | 5.17 |

## Main observations

- Dropout-only evaluation masking reduces pooled acceleration RMSE from 14.07 to 12.03 m/s².
- First-fusion pooled chunk-centered RMSE changes from 5.95 mm unmasked to 5.45 mm masked.
- Final-fusion pooled chunk-centered RMSE changes from 5.72 mm unmasked to 5.17 mm masked.
- Final fusion improves on first fusion in aggregate in both evaluations: 5.95 to 5.72 mm unmasked and 5.45 to 5.17 mm masked.

## Per-chunk results

Travel columns are centered RMSE in millimetres. Acceleration columns are RMSE in m/s² over samples where the encoder-derived acceleration magnitude exceeds 0.5 m/s². Travel evaluation uses active samples with positive encoder travel. Masked columns additionally exclude exact core-sensor zero-output samples; no safety halo is applied. Values displayed as 20.00% are strictly below the 20% selection threshold before rounding.

| Chunk | Log | Range (s) | Dropout | Train chunks | Accel all | Accel masked | F1 all | F1 masked | F2 all | F2 masked |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| log-0145-c01 | log-0145 | 5.38–79.55 | 0.00% | 101 | 10.25 | 10.25 | 4.40 | 4.40 | 4.15 | 4.15 |
| log-0145-c02 | log-0145 | 79.55–146.71 | 0.00% | 75 | 8.72 | 8.72 | 4.07 | 4.07 | 3.59 | 3.59 |
| log-0145-c03 | log-0145 | 146.71–212.01 | 0.00% | 50 | 9.52 | 9.52 | 5.57 | 5.57 | 4.81 | 4.81 |
| log-0145-c04 | log-0145 | 212.01–273.25 | 0.00% | 68 | 11.55 | 11.55 | 4.31 | 4.31 | 4.19 | 4.19 |
| log-0145-c05 | log-0145 | 273.25–333.26 | 0.00% | 50 | 9.94 | 9.94 | 4.08 | 4.08 | 3.72 | 3.72 |
| log-0147-c01 | log-0147 | 12.65–199.42 | 19.41% | 50 | 15.72 | 11.90 | 6.60 | 6.68 | 6.57 | 6.64 |
| log-0147-c02 | log-0147 | 304.46–388.23 | 20.00% | 126 | 14.83 | 10.37 | 5.21 | 5.28 | 3.89 | 3.51 |
| log-0147-c03 | log-0147 | 388.23–466.76 | 19.60% | 85 | 17.86 | 15.52 | 10.66 | 7.91 | 10.85 | 8.17 |
| log-0147-c04 | log-0147 | 466.76–528.90 | 15.57% | 79 | 19.05 | 17.61 | 6.95 | 6.67 | 6.49 | 6.14 |
| log-0147-c05 | log-0147 | 529.29–590.68 | 20.00% | 71 | 19.45 | 17.42 | 5.85 | 5.64 | 5.69 | 5.45 |
| log-0151-c01 | log-0151 | 95.14–161.28 | 20.00% | 85 | 16.56 | 11.04 | 5.54 | 5.17 | 5.02 | 4.51 |
| log-0152-c01 | log-0152 | 5.38–74.48 | 3.68% | 101 | 11.36 | 9.08 | 3.50 | 3.43 | 3.34 | 3.26 |
| log-0152-c02 | log-0152 | 337.44–397.84 | 20.00% | 67 | 13.28 | 11.71 | 5.49 | 5.06 | 5.48 | 5.05 |
| log-0155-c01 | log-0155 | 5.38–77.93 | 6.43% | 52 | 13.19 | 12.07 | 5.84 | 5.86 | 5.84 | 5.86 |
| log-0155-c02 | log-0155 | 77.93–138.01 | 6.18% | 51 | 13.98 | 12.34 | 7.43 | 7.04 | 7.43 | 7.04 |

Detailed RMSE, centered RMSE, MAE, mean error, and sample counts for magnetic-only, first-fusion, and second-fusion outputs are in `per_chunk_metrics.csv`.
