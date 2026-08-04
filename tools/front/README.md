# Front-Pipeline Diagnostics

These scripts target the original/front pipeline in `backend/pipeline.py`.

| Tool | Purpose |
| --- | --- |
| `analyze_accel_mismatch.py` | Diagnose projected acceleration against travel-derived acceleration and LIS2 saturation. |
| `analyze_angle.py` | Sweep older trigonometric angle-to-travel geometry and zero-angle settings. |
| `analyze_mag_lag.py` | Sweep lag between magnetometer-derived signals and travel. |
| `analyze_solver_regression.py` | Replay and inspect solver behavior versus mag-model predictions. |
| `refine_mag_proj.py` | Evaluate non-GT magnetometer projection-vector estimators. |

These scripts assume front cache keys such as `accel/lpfhp/proj`, `mag/proj/corr/lpf`,
and `mag_baseline`. For rear logs, start with `tools/stats_aggregator.py` or `tools/rear/`.
