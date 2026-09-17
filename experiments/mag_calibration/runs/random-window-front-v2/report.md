# random-window-front-v2

Generated: 2026-09-04T20:27:42+00:00

Completed 1200 of 1200 scheduled fits; 26 failed.

Windows are deterministic, randomly centered, and nested across durations within each log/repeat. Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.

## Training Window

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 1 | 24 | 1.932 | [1.489, 2.477] | 100.0% | 0.0% |
| oracle-power | 2 | 24 | 2.406 | [1.806, 2.863] | 100.0% | 0.0% |
| oracle-power | 5 | 24 | 2.660 | [2.175, 3.562] | 100.0% | 0.0% |
| oracle-power | 10 | 24 | 2.745 | [2.394, 4.142] | 100.0% | 0.0% |
| oracle-power | 40 | 24 | 3.169 | [2.689, 5.241] | 100.0% | 0.0% |
| self-supervised | 1 | 24 | 4.698 | [4.235, 5.779] | 100.0% | 16.7% |
| self-supervised | 2 | 24 | 3.975 | [3.305, 4.545] | 100.0% | 4.2% |
| self-supervised | 5 | 24 | 4.501 | [4.031, 5.296] | 100.0% | 0.8% |
| self-supervised | 10 | 24 | 3.738 | [3.444, 5.058] | 100.0% | 0.0% |
| self-supervised | 40 | 24 | 4.415 | [3.815, 5.846] | 100.0% | 0.0% |

## Full Log

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 1 | 24 | 6.746 | [5.543, 9.718] | 100.0% | 0.0% |
| oracle-power | 2 | 24 | 5.318 | [4.303, 7.401] | 100.0% | 0.0% |
| oracle-power | 5 | 24 | 4.544 | [3.741, 7.325] | 100.0% | 0.0% |
| oracle-power | 10 | 24 | 4.339 | [3.705, 7.140] | 100.0% | 0.0% |
| oracle-power | 40 | 24 | 4.293 | [3.566, 6.951] | 100.0% | 0.0% |
| self-supervised | 1 | 24 | 8.594 | [7.392, 9.628] | 100.0% | 16.7% |
| self-supervised | 2 | 24 | 6.804 | [5.737, 8.380] | 100.0% | 4.2% |
| self-supervised | 5 | 24 | 6.048 | [5.060, 7.714] | 100.0% | 0.8% |
| self-supervised | 10 | 24 | 5.147 | [4.330, 7.236] | 100.0% | 0.0% |
| self-supervised | 40 | 24 | 5.533 | [4.489, 7.296] | 100.0% | 0.0% |

## Interpretation notes

- `training_window` measures local reconstruction on the same self-supervised data block.
- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.
- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.
- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.
- Aggregates first take a median across repeats within each log, then weight logs equally.
