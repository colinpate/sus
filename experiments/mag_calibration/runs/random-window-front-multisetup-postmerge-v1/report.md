# random-window-front-multisetup-postmerge-v1

Generated: 2026-09-21T21:11:52+00:00

Completed 600 of 600 scheduled fits; 3 failed.

Windows are deterministic, randomly centered, and nested across durations within each log/repeat. Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.

## Training Window

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 5 | 15 | 3.220 | [2.317, 3.638] | 100.0% | 0.0% |
| oracle-power | 10 | 15 | 3.823 | [2.775, 4.844] | 100.0% | 0.0% |
| oracle-power | 20 | 15 | 4.637 | [3.274, 5.666] | 100.0% | 0.0% |
| oracle-power | 40 | 15 | 4.853 | [3.712, 5.767] | 100.0% | 0.0% |
| oracle-power | 60 | 15 | 5.028 | [3.956, 5.877] | 100.0% | 0.0% |
| self-supervised | 5 | 15 | 5.325 | [4.100, 6.324] | 100.0% | 3.3% |
| self-supervised | 10 | 15 | 6.124 | [4.621, 7.215] | 100.0% | 1.7% |
| self-supervised | 20 | 15 | 5.974 | [5.241, 7.126] | 100.0% | 0.0% |
| self-supervised | 40 | 15 | 6.256 | [5.612, 7.054] | 100.0% | 0.0% |
| self-supervised | 60 | 15 | 6.839 | [5.682, 7.388] | 100.0% | 0.0% |

## Full Log

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 5 | 15 | 8.101 | [5.855, 10.189] | 100.0% | 0.0% |
| oracle-power | 10 | 15 | 7.969 | [6.288, 9.464] | 100.0% | 0.0% |
| oracle-power | 20 | 15 | 6.769 | [5.123, 8.535] | 100.0% | 0.0% |
| oracle-power | 40 | 15 | 5.831 | [4.974, 8.181] | 100.0% | 0.0% |
| oracle-power | 60 | 15 | 5.805 | [4.266, 8.261] | 100.0% | 0.0% |
| self-supervised | 5 | 15 | 8.703 | [6.990, 10.001] | 100.0% | 3.3% |
| self-supervised | 10 | 15 | 7.593 | [6.496, 9.188] | 100.0% | 1.7% |
| self-supervised | 20 | 15 | 7.091 | [5.350, 8.544] | 100.0% | 0.0% |
| self-supervised | 40 | 15 | 7.257 | [6.140, 7.828] | 100.0% | 0.0% |
| self-supervised | 60 | 15 | 7.691 | [5.823, 8.305] | 100.0% | 0.0% |

## Full Log Excluding Training

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 5 | 15 | 8.154 | [5.896, 11.898] | 100.0% | 0.0% |
| oracle-power | 10 | 15 | 8.014 | [6.419, 9.444] | 100.0% | 0.0% |
| oracle-power | 20 | 15 | 6.855 | [5.125, 7.642] | 100.0% | 0.0% |
| oracle-power | 40 | 15 | 6.032 | [4.888, 8.543] | 100.0% | 0.0% |
| oracle-power | 60 | 15 | 6.011 | [4.652, 7.902] | 100.0% | 0.0% |
| self-supervised | 5 | 15 | 8.685 | [7.330, 10.017] | 100.0% | 3.3% |
| self-supervised | 10 | 15 | 7.656 | [6.473, 9.312] | 100.0% | 1.7% |
| self-supervised | 20 | 15 | 7.037 | [5.390, 8.289] | 100.0% | 0.0% |
| self-supervised | 40 | 15 | 7.295 | [6.158, 8.212] | 100.0% | 0.0% |
| self-supervised | 60 | 15 | 7.661 | [6.046, 8.454] | 100.0% | 0.0% |

## Interpretation notes

- `training_window` measures local reconstruction on the same self-supervised data block.
- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.
- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.
- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.
- Aggregates first take a median across repeats within each log, then weight logs equally.
