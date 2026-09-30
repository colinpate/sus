# cross-setup-front-pod-v2-full-v3

Completed 4332 of 4332 train/evaluation pairs; 0 failed.

Each source calibration is trained once on its complete source log. Diagonal pairs are the per-log baseline; off-diagonal pairs test transfer to another log. Aggregates first summarize target logs within each source calibration, then weight source logs equally.

| Trainer | Pair type | Source logs | Aligned RMSE (mm) | Anchored RMSE (mm) |
|---|---|---:|---:|---:|
| self-supervised | diagonal | 38 | 6.687 | 10.620 |
| self-supervised | transfer | 38 | 10.809 | 16.085 |
| oracle-power | diagonal | 38 | 4.847 | 4.847 |
| oracle-power | transfer | 38 | 10.297 | 23.549 |
| oracle-binned-median | diagonal | 38 | 4.259 | 4.273 |
| oracle-binned-median | transfer | 38 | 7.816 | 22.886 |

The diagonal must be retained: it controls for each log's intrinsic difficulty. Use `comparison_summary.csv` for the direct shared-calibration versus target per-log self-supervised comparison.

`boring_mask` is still used for evaluation, and oracle trainers use reference travel only from their source training log.
