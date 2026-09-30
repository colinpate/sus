# random-window-front-v4

Generated: 2026-09-04T21:10:25+00:00

Completed 3840 of 3840 scheduled fits; 0 failed.

Windows are deterministic, randomly centered, and nested across durations within each log/repeat. Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.

## Training Window

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 10 | 24 | 2.664 | [2.434, 3.979] | 100.0% | 0.0% |
| oracle-power | 20 | 24 | 2.977 | [2.599, 4.620] | 100.0% | 0.0% |
| oracle-power | 40 | 24 | 3.136 | [2.761, 4.926] | 100.0% | 0.0% |
| oracle-power | 60 | 24 | 3.248 | [2.907, 5.121] | 100.0% | 0.0% |
| self-supervised | 10 | 24 | 4.172 | [3.511, 5.254] | 100.0% | 0.0% |
| self-supervised | 20 | 24 | 4.202 | [3.568, 5.229] | 100.0% | 0.0% |
| self-supervised | 40 | 24 | 4.535 | [3.605, 5.837] | 100.0% | 0.0% |
| self-supervised | 60 | 24 | 4.918 | [3.780, 6.001] | 100.0% | 0.0% |

## Full Log

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 10 | 24 | 4.414 | [3.718, 7.162] | 100.0% | 0.0% |
| oracle-power | 20 | 24 | 4.358 | [3.500, 6.756] | 100.0% | 0.0% |
| oracle-power | 40 | 24 | 4.254 | [3.481, 6.940] | 100.0% | 0.0% |
| oracle-power | 60 | 24 | 4.325 | [3.490, 6.858] | 100.0% | 0.0% |
| self-supervised | 10 | 24 | 5.383 | [4.491, 7.504] | 100.0% | 0.0% |
| self-supervised | 20 | 24 | 5.147 | [4.473, 7.358] | 100.0% | 0.0% |
| self-supervised | 40 | 24 | 5.021 | [4.442, 7.320] | 100.0% | 0.0% |
| self-supervised | 60 | 24 | 5.429 | [4.404, 7.385] | 100.0% | 0.0% |

## Full Log Excluding Training

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 10 | 24 | 4.477 | [3.759, 7.035] | 100.0% | 0.0% |
| oracle-power | 20 | 24 | 4.480 | [3.554, 6.749] | 100.0% | 0.0% |
| oracle-power | 40 | 24 | 4.557 | [3.587, 6.989] | 100.0% | 0.0% |
| oracle-power | 60 | 24 | 4.640 | [3.541, 6.847] | 100.0% | 0.0% |
| self-supervised | 10 | 24 | 5.463 | [4.514, 7.568] | 100.0% | 0.0% |
| self-supervised | 20 | 24 | 5.229 | [4.582, 7.334] | 100.0% | 0.0% |
| self-supervised | 40 | 24 | 5.274 | [4.452, 7.458] | 100.0% | 0.0% |
| self-supervised | 60 | 24 | 5.524 | [4.417, 7.387] | 100.0% | 0.0% |

## Interpretation notes

- `training_window` measures local reconstruction on the same self-supervised data block.
- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.
- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.
- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.
- Aggregates first take a median across repeats within each log, then weight logs equally.
