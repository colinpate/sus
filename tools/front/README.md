# Front-Pipeline Diagnostics

These scripts target the original/front pipeline in `backend/pipeline.py`.

| Tool | Purpose |
| --- | --- |
| `analyze_accel_mismatch.py` | Diagnose projected acceleration against travel-derived acceleration and LIS2 saturation. |
| `analyze_angle.py` | Sweep trigonometric angle-to-travel geometry and zero-angle settings using pipeline-aligned cached signals and per-log configuration. |
| `analyze_mag_lag.py` | Sweep lag between magnetometer-derived signals and travel. |
| `analyze_solver_regression.py` | Replay and inspect solver behavior versus mag-model predictions. |
| `refine_mag_proj.py` | Evaluate non-GT magnetometer projection-vector estimators. |

The magnetic nuisance-field investigation is grouped under
`mag_nuisance/`. Its `README.md` distinguishes reusable solvers, current
experiments, encoder-supervised diagnostics, and archived prototypes.

These scripts assume front cache keys such as `accel/lpfhp/proj`, `mag/proj/corr/lpf`,
and `mag_baseline`. For rear logs, start with `tools/stats_aggregator.py` or `tools/rear/`.

`analyze_angle.py` reads the filtered, lag-adjusted angle and acceleration projection from
the pipeline cache. Its baseline defaults to each log's configured hypotenuse, adjacent
length, angle sign, and zero reference. Explicit CLI values are reported as overrides, and
the aggregate output distinguishes independently optimized per-log results from a single
candidate shared by every requested log.
