# Slayer maximally grown 20%-dropout chunk pipeline experiment

These are the seven non-overlapping ranges obtained by joining and maximally growing the original 15 seed chunks while preserving at least 60 seconds of activity and less than 20% raw core-sensor zero-output dropout per range. Full-log preprocessing, mounting calibration, magnetic baseline, and absolute reference are reused. The magnetic model, first travel solver, nuisance correction, and second travel solver run independently per chunk. Dropout masks are not used during training or solving; they are applied only to the masked evaluation columns.

## Aggregate results

| Stage | Units | Evaluation | Chunk-mean RMSE | Chunk-mean centered RMSE | Log-balanced RMSE | Pooled RMSE | Pooled centered RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| accel | m/s^2 | unmasked | 14.30 | 14.26 | 14.01 | 14.19 | 14.16 |
| accel | m/s^2 | masked | 11.79 | 11.66 | 11.60 | 11.89 | 11.79 |
| mag | mm | unmasked | 16.87 | 7.52 | 15.32 | 23.99 | 9.19 |
| mag | mm | masked | 16.20 | 6.72 | 14.61 | 21.98 | 8.10 |
| fusion1 | mm | unmasked | 15.53 | 5.85 | 13.57 | 23.43 | 6.10 |
| fusion1 | mm | masked | 15.19 | 5.59 | 13.24 | 21.49 | 5.73 |
| fusion2 | mm | unmasked | 15.48 | 5.55 | 13.49 | 23.55 | 5.70 |
| fusion2 | mm | masked | 15.16 | 5.25 | 13.18 | 21.64 | 5.28 |

## Main observations

- Dropout-only evaluation masking reduces pooled acceleration RMSE from 14.19 to 11.89 m/s².
- First-fusion pooled chunk-centered RMSE changes from 6.10 mm unmasked to 5.73 mm masked.
- Final-fusion pooled chunk-centered RMSE changes from 5.70 mm unmasked to 5.28 mm masked.
- Final fusion improves on first fusion in aggregate in both evaluations: 6.10 to 5.70 mm unmasked and 5.73 to 5.28 mm masked.

## Per-chunk results

Travel columns are centered RMSE in millimetres. Acceleration columns are RMSE in m/s² over samples where the encoder-derived acceleration magnitude exceeds 0.5 m/s². Travel evaluation uses active samples with positive encoder travel. Masked columns additionally exclude exact core-sensor zero-output samples; no safety halo is applied. Values displayed as 20.00% are strictly below the 20% selection threshold before rounding.

| Chunk | Log | Range (s) | Dropout | Train chunks | Accel all | Accel masked | F1 all | F1 masked | F2 all | F2 masked |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| log-0145-c01 | log-0145 | 5.38–395.67 | 0.00% | 394 | 9.84 | 9.84 | 4.53 | 4.53 | 4.11 | 4.11 |
| log-0147-c01 | log-0147 | 12.13–221.13 | 20.00% | 82 | 15.17 | 11.77 | 6.32 | 6.36 | 6.27 | 6.30 |
| log-0147-c02 | log-0147 | 237.46–621.74 | 20.00% | 492 | 17.49 | 14.77 | 8.06 | 7.55 | 7.32 | 6.68 |
| log-0151-c01 | log-0151 | 71.25–172.30 | 20.00% | 106 | 16.55 | 11.36 | 5.86 | 5.31 | 5.16 | 4.42 |
| log-0152-c01 | log-0152 | 5.38–167.92 | 20.00% | 188 | 13.98 | 11.22 | 4.66 | 4.40 | 4.52 | 4.26 |
| log-0152-c02 | log-0152 | 319.46–422.20 | 20.00% | 96 | 13.48 | 11.32 | 5.36 | 4.97 | 5.36 | 4.97 |
| log-0155-c01 | log-0155 | 5.38–151.29 | 6.18% | 113 | 13.62 | 12.23 | 6.14 | 6.02 | 6.13 | 6.02 |

Detailed RMSE, centered RMSE, MAE, mean error, and sample counts for magnetic-only, first-fusion, and second-fusion outputs are in `per_chunk_metrics.csv`.
