# random-window-front-stumpy-postmerge-v1

Generated: 2026-09-21T21:12:16+00:00

Completed 528 of 528 scheduled fits; 0 failed.

Windows are deterministic, randomly centered, and nested across durations within each log/repeat. Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.

## Training Window

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 5 | 11 | 1.716 | [1.319, 1.819] | 100.0% | 0.0% |
| oracle-power | 10 | 11 | 2.026 | [1.568, 2.352] | 100.0% | 0.0% |
| oracle-power | 20 | 11 | 2.503 | [2.099, 2.735] | 100.0% | 0.0% |
| oracle-power | 40 | 11 | 2.435 | [2.169, 2.731] | 100.0% | 0.0% |
| oracle-power | 60 | 11 | 2.707 | [2.244, 3.132] | 100.0% | 0.0% |
| oracle-power | 120 | 11 | 2.818 | [2.321, 3.458] | 100.0% | 0.0% |
| self-supervised | 5 | 11 | 3.543 | [3.233, 4.099] | 100.0% | 0.0% |
| self-supervised | 10 | 11 | 3.142 | [2.934, 4.050] | 100.0% | 0.0% |
| self-supervised | 20 | 11 | 3.298 | [2.783, 4.329] | 100.0% | 0.0% |
| self-supervised | 40 | 11 | 3.501 | [3.187, 3.889] | 100.0% | 0.0% |
| self-supervised | 60 | 11 | 3.774 | [3.263, 4.167] | 100.0% | 0.0% |
| self-supervised | 120 | 11 | 4.214 | [3.464, 4.392] | 100.0% | 0.0% |

## Full Log

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 5 | 11 | 3.532 | [2.873, 4.616] | 100.0% | 0.0% |
| oracle-power | 10 | 11 | 3.483 | [2.795, 3.971] | 100.0% | 0.0% |
| oracle-power | 20 | 11 | 3.441 | [2.699, 3.840] | 100.0% | 0.0% |
| oracle-power | 40 | 11 | 3.441 | [2.684, 3.751] | 100.0% | 0.0% |
| oracle-power | 60 | 11 | 3.435 | [2.616, 3.655] | 100.0% | 0.0% |
| oracle-power | 120 | 11 | 3.305 | [2.601, 3.677] | 100.0% | 0.0% |
| self-supervised | 5 | 11 | 4.616 | [4.131, 5.660] | 100.0% | 0.0% |
| self-supervised | 10 | 11 | 4.294 | [3.438, 4.681] | 100.0% | 0.0% |
| self-supervised | 20 | 11 | 4.171 | [3.273, 5.348] | 100.0% | 0.0% |
| self-supervised | 40 | 11 | 3.780 | [3.385, 4.510] | 100.0% | 0.0% |
| self-supervised | 60 | 11 | 3.925 | [3.266, 4.392] | 100.0% | 0.0% |
| self-supervised | 120 | 11 | 4.225 | [3.371, 4.408] | 100.0% | 0.0% |

## Full Log Excluding Training

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-power | 5 | 11 | 3.530 | [2.884, 4.647] | 100.0% | 0.0% |
| oracle-power | 10 | 11 | 3.522 | [2.810, 4.005] | 100.0% | 0.0% |
| oracle-power | 20 | 11 | 3.554 | [2.736, 3.906] | 100.0% | 0.0% |
| oracle-power | 40 | 11 | 3.639 | [2.694, 3.845] | 100.0% | 0.0% |
| oracle-power | 60 | 11 | 3.560 | [2.663, 3.788] | 100.0% | 0.0% |
| oracle-power | 120 | 11 | 3.409 | [2.537, 3.874] | 100.0% | 0.0% |
| self-supervised | 5 | 11 | 4.623 | [4.131, 5.687] | 100.0% | 0.0% |
| self-supervised | 10 | 11 | 4.263 | [3.443, 4.639] | 100.0% | 0.0% |
| self-supervised | 20 | 11 | 4.192 | [3.277, 5.410] | 100.0% | 0.0% |
| self-supervised | 40 | 11 | 3.781 | [3.368, 4.382] | 100.0% | 0.0% |
| self-supervised | 60 | 11 | 3.786 | [3.237, 4.472] | 100.0% | 0.0% |
| self-supervised | 120 | 11 | 4.151 | [3.537, 4.385] | 100.0% | 0.0% |

## Interpretation notes

- `training_window` measures local reconstruction on the same self-supervised data block.
- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.
- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.
- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.
- Aggregates first take a median across repeats within each log, then weight logs equally.
