# Post-mag-correction fusion `mag_off_floor` sweep

## Design

The final front fusion solve was replayed from each current pipeline cache using `travel/mag_nuisance/corrected` as its magnetic travel observation. The swept floors were `0.03`, `0.1`, `0.3`, `1`, `3`, `10`, `1e+06`; these correspond to anchor-off magnetic residual multipliers of 2.9%, 9.1%, 23.1%, 50.0%, 75.0%, 90.9%, 100.0%. The current value is `0.1`.

Tuning used 31 logs from `harry`, `jamaal`, `stumpjumper-front-pod-v2`. `stumpjumper-front-pod-v1` (7 logs) was not evaluated until after selection. The primary objective was the mean 0–30 mm centered RMSE after first averaging logs within each tuning cohort and then weighting the three cohorts equally. Centering used one offset per full log before the low-travel subset was scored.

## Tuning result

| mag_off_floor | off-gate weight | 0–30 mm RMSE | overall RMSE | converged |
| ---: | ---: | ---: | ---: | ---: |
| 0.03 | 2.9% | 10.663 mm | 6.336 mm | 90.3% |
| 0.1 | 9.1% | 8.822 mm | 5.879 mm | 100.0% |
| 0.3 | 23.1% | 7.882 mm | 5.701 mm | 100.0% |
| 1 | 50.0% | 7.451 mm | 5.648 mm | 100.0% |
| 3 | 75.0% | 7.348 mm | 5.650 mm | 100.0% |
| 10 | 90.9% | 7.326 mm | 5.657 mm | 100.0% |
| 1e+06 | 100.0% | 7.321 mm | 5.662 mm | 100.0% |

Selected: **`mag_off_floor=1e+06`**. Relative to `0.1` on tuning, its 0–30 mm RMSE changed by -1.501 mm (-17.0%) and overall centered RMSE changed by -0.217 mm (-3.7%).

The aggregate is heterogeneous across setups:

| tuning cohort | 0–30 mm change | logs improved | overall change | logs improved | uncentered change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `harry` | -3.684 mm | 14/14 | -0.835 mm | 14/14 | +0.116 mm |
| `jamaal` | -0.824 mm | 3/6 | +0.046 mm | 2/6 | +0.385 mm |
| `stumpjumper-front-pod-v2` | +0.003 mm | 4/11 | +0.137 mm | 4/11 | +0.372 mm |

## Held-out result

On `stumpjumper-front-pod-v1`, the selected value changed 0–30 mm centered RMSE by -0.341 mm (-4.8%) and overall centered RMSE by -0.066 mm (-1.7%) relative to `0.1`.

| value | 0–30 mm RMSE | overall RMSE | uncentered RMSE |
| ---: | ---: | ---: | ---: |
| 0.1 (current) | 7.058 mm | 3.960 mm | 11.201 mm |
| 1e+06 (selected) | 6.717 mm | 3.894 mm | 11.704 mm |

The held-out mean improvement was mixed at log level: 4/7 logs improved at 0–30 mm and 3/7 improved overall. Uncentered RMSE changed by +0.503 mm on average and improved on 0/7 logs.

## Checks and interpretation

Replaying the current `0.1` setting matched the cached final solver output to a worst-case absolute difference of 0 mm.
Incomplete convergence on tuning: 0.03: 90.3%.

The formal low-travel objective validates the hypothesis directionally, but it does not support changing the global default yet. Most of the tuning gain came from Harry; the fully-on endpoint slightly worsened overall centered error on Jamaal and Stumpjumper-v2, and it worsened uncentered error on every held-out log. A moderate floor of `0.3` was the only swept value that improved both mean low-travel and mean overall centered RMSE in every tuning cohort, so it is the better candidate for a separately preregistered validation run when another untouched setup is available.

This is a held-out cohort check rather than an uncertainty estimate: logs within a cohort share hardware and riding conditions, so the per-log sample count should not be read as independent replication.

Full per-log values are in `per_log_metrics.csv`; cohort and cohort-balanced summaries are in `aggregate_metrics.csv`; the frozen selection is in `selection.json`.
