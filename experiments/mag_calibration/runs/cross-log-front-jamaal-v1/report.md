# cross-log-front-jamaal-v1

Completed 108 of 108 train/evaluation pairs; 0 failed.

Each source calibration is trained once on its complete source log. Diagonal pairs are the per-log baseline; off-diagonal pairs test transfer to another log. Aggregates first summarize target logs within each source calibration, then weight source logs equally.

| Trainer | Pair type | Source logs | Aligned RMSE (mm) | Anchored RMSE (mm) |
|---|---|---:|---:|---:|
| self-supervised | diagonal | 6 | 10.327 | 15.971 |
| self-supervised | transfer | 6 | 9.063 | 14.803 |
| oracle-power | diagonal | 6 | 7.690 | 7.690 |
| oracle-power | transfer | 6 | 8.755 | 9.537 |
| oracle-binned-median | diagonal | 6 | 7.425 | 7.495 |
| oracle-binned-median | transfer | 6 | 8.201 | 9.375 |

The diagonal must be retained: it controls for each log's intrinsic difficulty. Use `comparison_summary.csv` for the direct shared-calibration versus target per-log self-supervised comparison.

`boring_mask` is still used for evaluation, and oracle trainers use reference travel only from their source training log.
