# Full-rate magnetic nuisance correction

All methods use identical full-rate samples, the pipeline boring mask,
finite-value checks, and the standard angle-corruption exclusion. The
aggregate values are arithmetic means of per-log metrics.

| Centering | Method | Overall RMSE | Delta | Bin RMSE | Delta | RMSE wins | Bin wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | `pipeline` | 9.353 | +0.000 | 9.186 | +0.000 | 0/15 | 0/15 |
| raw | `delta_lifted` | 8.700 | -0.653 | 8.804 | -0.382 | 15/15 | 13/15 |
| raw | `corrected_mag_observation` | 8.971 | -0.382 | 9.148 | -0.038 | 11/15 | 8/15 |
| raw | `fusion2` | 8.430 | -0.924 | 8.526 | -0.660 | 14/15 | 15/15 |
| centered | `pipeline` | 5.023 | +0.000 | 4.891 | +0.000 | 0/15 | 0/15 |
| centered | `delta_lifted` | 4.259 | -0.764 | 4.344 | -0.547 | 14/15 | 13/15 |
| centered | `corrected_mag_observation` | 4.368 | -0.655 | 4.453 | -0.437 | 14/15 | 9/15 |
| centered | `fusion2` | 4.160 | -0.863 | 4.290 | -0.600 | 15/15 | 15/15 |

## Per-log raw errors

| Log | Base RMSE | Delta-lift RMSE | Fusion2 RMSE | Base bin | Delta-lift bin | Fusion2 bin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `log-0046` | 15.028 | 14.368 | 14.038 | 15.378 | 14.943 | 14.474 |
| `log-0048` | 6.541 | 5.981 | 5.589 | 7.891 | 7.739 | 7.215 |
| `log-0049` | 6.780 | 6.697 | 6.208 | 7.048 | 7.063 | 6.597 |
| `log-0054` | 7.196 | 6.414 | 6.039 | 7.510 | 6.920 | 6.485 |
| `log-0055` | 8.413 | 8.081 | 8.460 | 7.931 | 7.534 | 7.892 |
| `log-0056` | 7.223 | 6.216 | 5.962 | 7.802 | 7.046 | 6.831 |
| `log-0058` | 3.566 | 3.415 | 3.242 | 3.942 | 3.800 | 3.669 |
| `log-0063` | 3.987 | 3.871 | 3.709 | 4.395 | 4.300 | 4.145 |
| `log-0071_183` | 8.363 | 8.201 | 7.968 | 8.095 | 8.038 | 7.718 |
| `log-0072_184` | 3.637 | 3.471 | 3.507 | 3.852 | 3.656 | 3.653 |
| `log-0073_185` | 10.335 | 10.294 | 9.388 | 9.361 | 9.666 | 8.774 |
| `log-0078-valid` | 15.098 | 13.345 | 13.029 | 14.366 | 13.486 | 13.227 |
| `log-0079` | 11.994 | 11.092 | 10.335 | 11.439 | 11.135 | 10.482 |
| `log-0080-valid` | 16.374 | 14.604 | 14.937 | 15.238 | 13.621 | 14.103 |
| `log-0081` | 15.768 | 14.451 | 14.032 | 13.541 | 13.107 | 12.625 |

## Per-log centered errors

| Log | Base RMSE | Delta-lift RMSE | Fusion2 RMSE | Base bin | Delta-lift bin | Fusion2 bin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `log-0046` | 3.340 | 3.136 | 3.283 | 4.316 | 3.855 | 4.212 |
| `log-0048` | 3.364 | 3.088 | 2.982 | 3.930 | 3.988 | 3.799 |
| `log-0049` | 3.558 | 3.148 | 2.911 | 3.586 | 3.157 | 2.950 |
| `log-0054` | 3.444 | 2.867 | 2.665 | 3.680 | 3.132 | 2.936 |
| `log-0055` | 4.331 | 4.386 | 4.082 | 4.238 | 4.302 | 4.046 |
| `log-0056` | 3.673 | 3.259 | 3.342 | 3.926 | 3.611 | 3.776 |
| `log-0058` | 3.311 | 3.195 | 3.136 | 3.587 | 3.401 | 3.454 |
| `log-0063` | 3.473 | 3.227 | 3.159 | 3.581 | 3.302 | 3.263 |
| `log-0071_183` | 2.742 | 2.623 | 2.639 | 2.917 | 2.834 | 2.802 |
| `log-0072_184` | 3.626 | 3.435 | 3.405 | 3.888 | 3.715 | 3.702 |
| `log-0073_185` | 5.709 | 5.332 | 4.945 | 5.483 | 5.343 | 4.925 |
| `log-0078-valid` | 7.385 | 4.766 | 4.889 | 6.195 | 4.588 | 4.747 |
| `log-0079` | 7.627 | 6.013 | 5.978 | 6.667 | 5.698 | 5.697 |
| `log-0080-valid` | 9.492 | 7.201 | 6.880 | 9.209 | 7.377 | 7.162 |
| `log-0081` | 10.271 | 8.207 | 8.106 | 8.155 | 6.850 | 6.886 |

`per_log.csv` contains individual travel-bin errors and sample counts,
including the corrected magnetometer observation before refusion.

## High-frequency preservation

After a 5 Hz high-pass, the delta lift changes the baseline travel by
0.310 mm RMS on average. The second fusion pass changes it by 0.263 mm RMS.
The mean high-frequency error versus the encoder is 1.601 mm for the baseline, 1.593 mm after delta lifting, and 1.598 mm after refusion.
`frequency_dynamics.csv` contains the per-log values.
