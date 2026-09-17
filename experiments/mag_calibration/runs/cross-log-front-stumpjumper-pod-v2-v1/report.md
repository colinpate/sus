# cross-log-front-stumpjumper-pod-v2-v1

Completed 363 of 363 train/evaluation pairs; 0 failed.

Each source calibration is trained once on its complete source log. Diagonal pairs are the per-log baseline; off-diagonal pairs test transfer to another log. Aggregates first summarize target logs within each source calibration, then weight source logs equally.

| Trainer | Pair type | Source logs | Aligned RMSE (mm) | Anchored RMSE (mm) |
|---|---|---:|---:|---:|
| self-supervised | diagonal | 11 | 4.577 | 8.163 |
| self-supervised | transfer | 11 | 4.798 | 7.781 |
| oracle-power | diagonal | 11 | 2.997 | 2.997 |
| oracle-power | transfer | 11 | 3.429 | 3.847 |
| oracle-binned-median | diagonal | 11 | 2.882 | 2.887 |
| oracle-binned-median | transfer | 11 | 3.244 | 3.755 |

The diagonal must be retained: it controls for each log's intrinsic difficulty. Use `comparison_summary.csv` for the direct shared-calibration versus target per-log self-supervised comparison.

`boring_mask` is still used for evaluation, and oracle trainers use reference travel only from their source training log.
