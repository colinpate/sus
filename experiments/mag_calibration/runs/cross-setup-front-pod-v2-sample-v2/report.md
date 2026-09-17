# cross-setup-front-pod-v2-sample-v2

Completed 1200 of 1200 train/evaluation pairs; 0 failed.

Each source calibration is trained once on its complete source log. Diagonal pairs are the per-log baseline; off-diagonal pairs test transfer to another log. Aggregates first summarize target logs within each source calibration, then weight source logs equally.

| Trainer | Pair type | Source logs | Aligned RMSE (mm) | Anchored RMSE (mm) |
|---|---|---:|---:|---:|
| self-supervised | diagonal | 20 | 7.750 | 11.976 |
| self-supervised | transfer | 20 | 11.429 | 14.585 |
| oracle-power | diagonal | 20 | 4.456 | 4.456 |
| oracle-power | transfer | 20 | 10.881 | 25.431 |
| oracle-binned-median | diagonal | 20 | 3.732 | 3.746 |
| oracle-binned-median | transfer | 20 | 10.321 | 24.686 |

The diagonal must be retained: it controls for each log's intrinsic difficulty. Use `comparison_summary.csv` for the direct shared-calibration versus target per-log self-supervised comparison.

`boring_mask` is still used for evaluation, and oracle trainers use reference travel only from their source training log.
