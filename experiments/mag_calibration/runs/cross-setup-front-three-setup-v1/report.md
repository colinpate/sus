# cross-setup-front-three-setup-v1

Completed 1728 of 1728 train/evaluation pairs; 0 failed.

Each source calibration is trained once on its complete source log. Diagonal pairs are the per-log baseline; off-diagonal pairs test transfer to another log. Aggregates first summarize target logs within each source calibration, then weight source logs equally.

| Trainer | Pair type | Source logs | Aligned RMSE (mm) | Anchored RMSE (mm) |
|---|---|---:|---:|---:|
| self-supervised | diagonal | 24 | 5.132 | 11.126 |
| self-supervised | transfer | 24 | 5.392 | 12.060 |
| oracle-power | diagonal | 24 | 3.763 | 3.763 |
| oracle-power | transfer | 24 | 4.531 | 6.243 |
| oracle-binned-median | diagonal | 24 | 3.692 | 3.711 |
| oracle-binned-median | transfer | 24 | 4.420 | 6.226 |

The diagonal must be retained: it controls for each log's intrinsic difficulty. Use `comparison_summary.csv` for the direct shared-calibration versus target per-log self-supervised comparison.

`boring_mask` is still used for evaluation, and oracle trainers use reference travel only from their source training log.
