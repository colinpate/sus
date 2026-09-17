# random-window-front-v1

Generated: 2026-09-04T20:18:44+00:00

Completed 3960 of 3960 scheduled fits; 2 failed.

Windows are deterministic, randomly centered, and nested across durations within each log/repeat. Each trainer receives the identical window. Active time and scoring currently use `boring_mask`.

## Training Window

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-isotonic | 5 | 11 | 1.558 | [1.376, 1.749] | 100.0% | 0.0% |
| oracle-isotonic | 10 | 11 | 1.803 | [1.585, 2.049] | 100.0% | 0.0% |
| oracle-isotonic | 20 | 11 | 2.145 | [1.927, 2.444] | 100.0% | 0.0% |
| oracle-isotonic | 40 | 11 | 2.286 | [2.090, 2.580] | 100.0% | 0.0% |
| oracle-isotonic | 60 | 11 | 2.299 | [2.168, 2.693] | 100.0% | 0.0% |
| oracle-isotonic | 120 | 11 | 2.462 | [2.244, 2.783] | 100.0% | 0.0% |
| oracle-power | 5 | 11 | 2.029 | [1.595, 2.137] | 100.0% | 0.0% |
| oracle-power | 10 | 11 | 2.046 | [1.767, 2.423] | 100.0% | 0.0% |
| oracle-power | 20 | 11 | 2.485 | [2.128, 2.869] | 100.0% | 0.0% |
| oracle-power | 40 | 11 | 2.551 | [2.243, 2.966] | 100.0% | 0.0% |
| oracle-power | 60 | 11 | 2.534 | [2.273, 2.992] | 100.0% | 0.0% |
| oracle-power | 120 | 11 | 2.596 | [2.401, 2.965] | 100.0% | 0.0% |
| self-supervised | 5 | 11 | 3.412 | [3.089, 3.807] | 100.0% | 0.5% |
| self-supervised | 10 | 11 | 3.539 | [3.160, 3.708] | 100.0% | 0.5% |
| self-supervised | 20 | 11 | 3.543 | [3.381, 4.375] | 100.0% | 0.0% |
| self-supervised | 40 | 11 | 3.684 | [3.355, 4.104] | 100.0% | 0.0% |
| self-supervised | 60 | 11 | 3.675 | [3.523, 4.217] | 100.0% | 0.0% |
| self-supervised | 120 | 11 | 4.073 | [3.653, 4.742] | 100.0% | 0.0% |

## Full Log

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-isotonic | 5 | 11 | 4.040 | [3.155, 4.745] | 100.0% | 0.0% |
| oracle-isotonic | 10 | 11 | 3.542 | [3.008, 3.673] | 100.0% | 0.0% |
| oracle-isotonic | 20 | 11 | 3.229 | [2.720, 3.584] | 100.0% | 0.0% |
| oracle-isotonic | 40 | 11 | 3.094 | [2.618, 3.492] | 100.0% | 0.0% |
| oracle-isotonic | 60 | 11 | 3.078 | [2.482, 3.414] | 100.0% | 0.0% |
| oracle-isotonic | 120 | 11 | 2.923 | [2.428, 3.423] | 100.0% | 0.0% |
| oracle-power | 5 | 11 | 3.318 | [2.643, 4.118] | 100.0% | 0.0% |
| oracle-power | 10 | 11 | 3.344 | [2.570, 3.923] | 100.0% | 0.0% |
| oracle-power | 20 | 11 | 3.318 | [2.570, 3.666] | 100.0% | 0.0% |
| oracle-power | 40 | 11 | 3.313 | [2.551, 3.637] | 100.0% | 0.0% |
| oracle-power | 60 | 11 | 3.235 | [2.539, 3.608] | 100.0% | 0.0% |
| oracle-power | 120 | 11 | 3.007 | [2.546, 3.609] | 100.0% | 0.0% |
| self-supervised | 5 | 11 | 4.745 | [4.336, 4.940] | 100.0% | 0.5% |
| self-supervised | 10 | 11 | 4.179 | [3.919, 4.441] | 100.0% | 0.5% |
| self-supervised | 20 | 11 | 3.988 | [3.793, 5.058] | 100.0% | 0.0% |
| self-supervised | 40 | 11 | 3.796 | [3.705, 4.443] | 100.0% | 0.0% |
| self-supervised | 60 | 11 | 3.982 | [3.767, 4.661] | 100.0% | 0.0% |
| self-supervised | 120 | 11 | 4.028 | [3.726, 4.961] | 100.0% | 0.0% |

## Full Log Excluding Training

| Trainer | Duration (s) | Logs | Aligned RMSE median (mm) | 95% log-bootstrap CI | Complete | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| oracle-isotonic | 5 | 11 | 4.069 | [3.173, 4.776] | 100.0% | 0.0% |
| oracle-isotonic | 10 | 11 | 3.566 | [3.043, 3.747] | 100.0% | 0.0% |
| oracle-isotonic | 20 | 11 | 3.288 | [2.775, 3.749] | 100.0% | 0.0% |
| oracle-isotonic | 40 | 11 | 3.191 | [2.713, 3.826] | 100.0% | 0.0% |
| oracle-isotonic | 60 | 11 | 3.222 | [2.556, 3.738] | 100.0% | 0.0% |
| oracle-isotonic | 120 | 11 | 3.226 | [2.645, 3.813] | 100.0% | 0.0% |
| oracle-power | 5 | 11 | 3.333 | [2.651, 4.139] | 100.0% | 0.0% |
| oracle-power | 10 | 11 | 3.363 | [2.585, 3.956] | 100.0% | 0.0% |
| oracle-power | 20 | 11 | 3.361 | [2.609, 3.706] | 100.0% | 0.0% |
| oracle-power | 40 | 11 | 3.364 | [2.615, 3.756] | 100.0% | 0.0% |
| oracle-power | 60 | 11 | 3.313 | [2.665, 3.696] | 100.0% | 0.0% |
| oracle-power | 120 | 11 | 3.288 | [2.707, 3.720] | 100.0% | 0.0% |
| self-supervised | 5 | 11 | 4.746 | [4.347, 4.980] | 100.0% | 0.5% |
| self-supervised | 10 | 11 | 4.140 | [3.921, 4.455] | 100.0% | 0.5% |
| self-supervised | 20 | 11 | 3.953 | [3.786, 5.113] | 100.0% | 0.0% |
| self-supervised | 40 | 11 | 3.836 | [3.776, 4.511] | 100.0% | 0.0% |
| self-supervised | 60 | 11 | 3.996 | [3.739, 4.795] | 100.0% | 0.0% |
| self-supervised | 120 | 11 | 4.051 | [3.821, 5.036] | 100.0% | 0.0% |

## Interpretation notes

- `training_window` measures local reconstruction on the same self-supervised data block.
- `full_log` measures how much data is needed for a calibration that represents the recording as a whole.
- `full_log_excluding_training` removes direct sample overlap while staying within the same recording.
- `oracle-power` uses reference travel but the production power-curve family; `oracle-isotonic` is a more flexible ceiling.
- Aggregates first take a median across repeats within each log, then weight logs equally.
