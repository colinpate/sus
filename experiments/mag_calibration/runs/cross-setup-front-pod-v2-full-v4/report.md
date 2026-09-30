# cross-setup-front-pod-v2-full-v4

Completed 4332 of 4332 train/evaluation pairs; 0 failed.

Each source calibration is trained once on its complete source log. Diagonal pairs are the per-log baseline; off-diagonal pairs test transfer to another log. Aggregates first summarize target logs within each source calibration, then weight source logs equally.

| Trainer | Pair type | Source logs | Aligned RMSE (mm) | Anchored RMSE (mm) |
|---|---|---:|---:|---:|
| self-supervised | diagonal | 38 | 6.691 | 10.321 |
| self-supervised | transfer | 38 | 10.295 | 15.951 |
| oracle-power | diagonal | 38 | 4.934 | 4.934 |
| oracle-power | transfer | 38 | 10.289 | 23.710 |
| oracle-binned-median | diagonal | 38 | 4.309 | 4.332 |
| oracle-binned-median | transfer | 38 | 8.045 | 22.939 |

The diagonal must be retained: it controls for each log's intrinsic difficulty. Use `comparison_summary.csv` for the direct shared-calibration versus target per-log self-supervised comparison.

`boring_mask` is still used for evaluation, and oracle trainers use reference travel only from their source training log.
