# Rear Experiment Archive

These scripts reproduce completed rear investigations. Keep them when their report artifacts
are still useful, but do not treat them as default entrypoints for current work.

| Tool | Report | Takeaway |
| --- | --- | --- |
| `analyze_rear_accel.py` | `reports/rear_accel_exploration/` | Early rear acceleration estimator exploration. |
| `analyze_rear_chunk_curve_methods.py` | `reports/rear_chunk_curve_methods_149_153/` | Chunk-only curvature methods did not beat the current learned model decisively. |
| `analyze_rear_gyro_proxy_methods.py` | `reports/rear_gyro_proxy_methods_149_153/` | Gyro carries motion information but did not beat the accel proxy overall. |
| `analyze_rear_mag_error_patterns.py` | `reports/rear_mag_error_patterns*/` | Static mag error patterns and quartile analysis. |
| `analyze_rear_slope_guided_methods.py` | `reports/rear_slope_guided_methods_149_153/` | Slope-guided objectives mostly behaved as no-ops or worsened RMSE. |
| `rear_mag_chunk_experiments.py` | Superseded by `../analyze_rear_chunking_tradeoffs.py` | Early chunk-parameter sweep retained for reference. |

Run these only when you need to reproduce or extend a specific archived finding.
