# random-window-front-v4-jamaal

Generated: 2026-09-04T21:45:00+00:00

Completed 1200 of 1200 scheduled fits; 0 failed.

Windows are deterministic, randomly centered, and nested across durations within each log/repeat. Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.

## Training Window

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 10 | 6 | 5.181 | [4.274, 6.120] | 100.0% | 0.0% |
| oracle-power | 20 | 6 | 6.016 | [4.585, 6.628] | 100.0% | 0.0% |
| oracle-power | 40 | 6 | 6.376 | [5.012, 7.186] | 100.0% | 0.0% |
| oracle-power | 60 | 6 | 6.613 | [5.106, 7.574] | 100.0% | 0.0% |
| oracle-power | 80 | 6 | 6.524 | [5.296, 7.930] | 100.0% | 0.0% |
| self-supervised | 10 | 6 | 6.036 | [5.525, 7.159] | 100.0% | 0.0% |
| self-supervised | 20 | 6 | 6.625 | [5.528, 8.013] | 100.0% | 0.0% |
| self-supervised | 40 | 6 | 7.697 | [5.880, 8.840] | 100.0% | 0.0% |
| self-supervised | 60 | 6 | 7.754 | [6.244, 8.984] | 100.0% | 0.0% |
| self-supervised | 80 | 6 | 7.679 | [6.432, 9.776] | 100.0% | 0.0% |

## Full Log

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 10 | 6 | 9.045 | [7.431, 10.073] | 100.0% | 0.0% |
| oracle-power | 20 | 6 | 8.268 | [7.062, 9.620] | 100.0% | 0.0% |
| oracle-power | 40 | 6 | 7.913 | [7.122, 9.253] | 100.0% | 0.0% |
| oracle-power | 60 | 6 | 7.873 | [7.024, 9.107] | 100.0% | 0.0% |
| oracle-power | 80 | 6 | 7.868 | [6.933, 9.017] | 100.0% | 0.0% |
| self-supervised | 10 | 6 | 8.454 | [7.486, 9.825] | 100.0% | 0.0% |
| self-supervised | 20 | 6 | 8.507 | [7.543, 10.288] | 100.0% | 0.0% |
| self-supervised | 40 | 6 | 8.736 | [8.174, 11.200] | 100.0% | 0.0% |
| self-supervised | 60 | 6 | 8.394 | [7.929, 11.230] | 100.0% | 0.0% |
| self-supervised | 80 | 6 | 8.867 | [7.826, 11.013] | 100.0% | 0.0% |

## Full Log Excluding Training

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 10 | 6 | 9.041 | [7.488, 10.142] | 100.0% | 0.0% |
| oracle-power | 20 | 6 | 8.363 | [7.068, 9.672] | 100.0% | 0.0% |
| oracle-power | 40 | 6 | 8.036 | [7.259, 9.345] | 100.0% | 0.0% |
| oracle-power | 60 | 6 | 8.043 | [7.073, 9.261] | 100.0% | 0.0% |
| oracle-power | 80 | 6 | 7.993 | [6.953, 9.195] | 100.0% | 0.0% |
| self-supervised | 10 | 6 | 8.473 | [7.513, 9.843] | 100.0% | 0.0% |
| self-supervised | 20 | 6 | 8.579 | [7.463, 9.740] | 100.0% | 0.0% |
| self-supervised | 40 | 6 | 8.813 | [8.025, 11.341] | 100.0% | 0.0% |
| self-supervised | 60 | 6 | 8.670 | [7.893, 11.368] | 100.0% | 0.0% |
| self-supervised | 80 | 6 | 9.083 | [7.701, 11.248] | 100.0% | 0.0% |

## Interpretation notes

- `training_window` measures local reconstruction on the same self-supervised data block.
- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.
- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.
- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.
- Aggregates first take a median across repeats within each log, then weight logs equally.
