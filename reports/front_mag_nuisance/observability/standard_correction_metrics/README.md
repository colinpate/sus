# Standard metrics for stage-one magnetic correction

Both methods are evaluated on identical 10 Hz samples using the regular
boring mask, finite-value checks, and angle-corruption exclusion. Standard
`bin_rmse` is the equal-weight RMS over eligible 0--30, 30--60, 60--90,
90--120, and 120--150 mm bins; a bin needs at least 100 sampled points.

| Centering | Method | Overall RMSE | Bin RMSE | RMSE wins | Bin wins |
| --- | --- | ---: | ---: | ---: | ---: |
| raw | `body_world_corrected` | 8.671 | 8.842 | 15/15 | 13/15 |
| raw | `pipeline` | 9.366 | 9.350 | 0/15 | 0/15 |
| centered | `body_world_corrected` | 4.219 | 4.433 | 14/15 | 14/15 |
| centered | `pipeline` | 5.054 | 5.182 | 0/15 | 0/15 |

## Per-log raw errors

| Log | Pipeline RMSE | Corrected RMSE | Delta | Pipeline bin RMSE | Corrected bin RMSE | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `log-0056` | 7.248 | 6.206 | -1.042 | 7.603 | 6.528 | -1.075 |
| `log-0063` | 3.994 | 3.886 | -0.108 | 4.257 | 4.141 | -0.116 |
| `log-0046` | 15.019 | 14.341 | -0.677 | 15.377 | 15.051 | -0.326 |
| `log-0048` | 6.521 | 5.938 | -0.583 | 7.450 | 6.985 | -0.465 |
| `log-0049` | 6.742 | 6.668 | -0.074 | 7.131 | 7.185 | +0.054 |
| `log-0054` | 7.220 | 6.438 | -0.783 | 7.934 | 7.331 | -0.603 |
| `log-0055` | 8.453 | 8.131 | -0.323 | 8.136 | 7.552 | -0.584 |
| `log-0058` | 3.567 | 3.320 | -0.247 | 4.073 | 3.828 | -0.245 |
| `log-0071_183` | 8.416 | 8.221 | -0.196 | 8.070 | 7.935 | -0.135 |
| `log-0072_184` | 3.644 | 3.417 | -0.227 | 4.185 | 3.854 | -0.331 |
| `log-0073_185` | 10.356 | 10.312 | -0.045 | 10.363 | 10.843 | +0.480 |
| `log-0078-valid` | 15.226 | 13.410 | -1.816 | 15.052 | 13.947 | -1.105 |
| `log-0079` | 11.909 | 10.854 | -1.055 | 11.484 | 10.817 | -0.667 |
| `log-0080-valid` | 16.387 | 14.524 | -1.863 | 15.211 | 13.466 | -1.745 |
| `log-0081` | 15.789 | 14.403 | -1.386 | 13.930 | 13.159 | -0.770 |

## Per-log centered errors

| Log | Pipeline RMSE | Corrected RMSE | Delta | Pipeline bin RMSE | Corrected bin RMSE | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `log-0056` | 3.665 | 3.183 | -0.482 | 4.134 | 3.407 | -0.727 |
| `log-0063` | 3.479 | 3.238 | -0.242 | 3.757 | 3.446 | -0.311 |
| `log-0046` | 3.353 | 3.077 | -0.276 | 4.314 | 3.716 | -0.598 |
| `log-0048` | 3.365 | 3.053 | -0.312 | 3.895 | 3.472 | -0.423 |
| `log-0049` | 3.565 | 3.178 | -0.387 | 4.000 | 3.534 | -0.466 |
| `log-0054` | 3.450 | 2.871 | -0.578 | 4.040 | 3.508 | -0.532 |
| `log-0055` | 4.333 | 4.401 | +0.068 | 4.718 | 4.790 | +0.071 |
| `log-0058` | 3.317 | 3.098 | -0.219 | 3.816 | 3.468 | -0.348 |
| `log-0071_183` | 2.776 | 2.553 | -0.223 | 2.917 | 2.632 | -0.284 |
| `log-0072_184` | 3.636 | 3.387 | -0.250 | 4.212 | 3.898 | -0.314 |
| `log-0073_185` | 5.775 | 5.398 | -0.376 | 5.960 | 5.812 | -0.148 |
| `log-0078-valid` | 7.558 | 4.828 | -2.730 | 6.690 | 4.725 | -1.965 |
| `log-0079` | 7.588 | 5.689 | -1.899 | 7.120 | 5.617 | -1.503 |
| `log-0080-valid` | 9.654 | 7.215 | -2.439 | 9.406 | 7.476 | -1.930 |
| `log-0081` | 10.299 | 8.121 | -2.179 | 8.748 | 6.997 | -1.751 |

`per_log.csv` contains the five individual travel-bin RMSE values and
sample counts for every log. Aggregate values above are means of the
per-log metrics, matching the way the experiment's weak RMSE is summarized.
