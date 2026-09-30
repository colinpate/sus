# random-window-front-v1-shorter

Generated: 2026-09-04T20:18:17+00:00

Completed 1100 of 1100 scheduled fits; 46 failed.

Windows are deterministic, randomly centered, and nested across durations within each log/repeat. Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.

## Training Window

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 1 | 11 | 1.405 | [0.916, 1.713] | 100.0% | 0.0% |
| oracle-power | 2 | 11 | 1.536 | [1.141, 1.914] | 100.0% | 0.0% |
| oracle-power | 3 | 11 | 1.642 | [1.325, 2.076] | 100.0% | 0.0% |
| oracle-power | 4 | 11 | 1.591 | [1.454, 2.169] | 100.0% | 0.0% |
| oracle-power | 5 | 11 | 1.658 | [1.473, 2.138] | 100.0% | 0.0% |
| self-supervised | 1 | 11 | 4.175 | [3.056, 4.623] | 100.0% | 18.2% |
| self-supervised | 2 | 11 | 3.457 | [2.705, 4.124] | 100.0% | 8.2% |
| self-supervised | 3 | 11 | 3.898 | [2.615, 4.379] | 100.0% | 7.3% |
| self-supervised | 4 | 11 | 3.645 | [2.941, 4.049] | 100.0% | 4.5% |
| self-supervised | 5 | 11 | 3.607 | [2.927, 4.218] | 100.0% | 3.6% |

## Full Log

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 1 | 11 | 4.386 | [3.872, 6.094] | 100.0% | 0.0% |
| oracle-power | 2 | 11 | 3.955 | [2.963, 4.464] | 100.0% | 0.0% |
| oracle-power | 3 | 11 | 3.871 | [2.683, 4.076] | 100.0% | 0.0% |
| oracle-power | 4 | 11 | 3.681 | [2.849, 3.957] | 100.0% | 0.0% |
| oracle-power | 5 | 11 | 3.760 | [2.857, 3.952] | 100.0% | 0.0% |
| self-supervised | 1 | 11 | 7.669 | [5.829, 8.718] | 100.0% | 18.2% |
| self-supervised | 2 | 11 | 6.030 | [4.788, 7.167] | 100.0% | 8.2% |
| self-supervised | 3 | 11 | 5.565 | [4.065, 5.782] | 100.0% | 7.3% |
| self-supervised | 4 | 11 | 5.035 | [3.883, 5.470] | 100.0% | 4.5% |
| self-supervised | 5 | 11 | 4.971 | [4.315, 5.392] | 100.0% | 3.6% |

## Full Log Excluding Training

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 1 | 11 | 4.391 | [3.875, 6.103] | 100.0% | 0.0% |
| oracle-power | 2 | 11 | 3.960 | [2.967, 4.473] | 100.0% | 0.0% |
| oracle-power | 3 | 11 | 3.883 | [2.685, 4.092] | 100.0% | 0.0% |
| oracle-power | 4 | 11 | 3.696 | [2.857, 3.970] | 100.0% | 0.0% |
| oracle-power | 5 | 11 | 3.778 | [2.863, 3.971] | 100.0% | 0.0% |
| self-supervised | 1 | 11 | 7.689 | [5.821, 8.729] | 100.0% | 18.2% |
| self-supervised | 2 | 11 | 6.037 | [4.790, 7.180] | 100.0% | 8.2% |
| self-supervised | 3 | 11 | 5.582 | [4.059, 5.789] | 100.0% | 7.3% |
| self-supervised | 4 | 11 | 5.029 | [3.889, 5.482] | 100.0% | 4.5% |
| self-supervised | 5 | 11 | 4.986 | [4.319, 5.403] | 100.0% | 3.6% |

## Interpretation notes

- `training_window` measures local reconstruction on the same self-supervised data block.
- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.
- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.
- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.
- Aggregates first take a median across repeats within each log, then weight logs equally.
