# Rear Mag Chunking Tradeoff Analysis

Logs:

- `log148_rear`
- `log149_rear`
- `log150_rear`
- `log151_rear`
- `log152_rear`
- `log153_rear`

Metrics are centered on `boring_mask` to match `tools/stats_aggregator.py`.
`bin_rmse` is the equal-weight average over the five 0-150 mm travel bins; `worst_bin_rmse` is the largest of those bins.

## Main Findings

- The best balanced result is `mag_gated_blend_p5_p50`: train one `paired_zv` curve and one `centered_zv` curve with a `0.28` power prior, use the centered curve at the low-mag/high-travel end, and fade to the paired curve by the median mag.
- Simply mixing both chunk sets in one least-squares fit does not combine the advantages. `hybrid_uniform` improves sample RMSE versus `paired_default`, but its worst-bin RMSE stays much closer to paired than centered.
- `centered_zv` is the best single-model family for equal-bin and high-travel error. A lower power prior (`0.28`) improves it a bit more.
- `paired_zv` remains strong in the dense low/mid travel range, but it leaves large 90-150 mm tail errors. Mag-bin chunk weighting and pair selection changes did not materially fix that.
- A strict paired-chunk correlation filter (`min_abs_b_x_corr=0.7`) gets the lowest mean worst-bin RMSE among single fits, but sample RMSE is too high to be a good default.

## Ranking Summary

- Best composite score: `mag_gated_blend_p5_p50`.
- Best mean worst-bin RMSE: `paired_corr_0p7`.
- Best mean sample RMSE: `mag_gated_blend_p5_p50`.

## Aggregate Metrics

| Variant | Chunking | Weighting | RMSE | Bin RMSE | Worst Bin | Max Worst | Chunks | Power | |Scale| | Score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mag_gated_blend_p5_p50` | `paired_zv+centered_zv` | `mag_gate_p5_p50` | 4.415 | 7.967 | 14.139 | 18.767 | 10508 | nan | nan | 11.934 |
| `centered_power_prior_0p28` | `centered_zv` | `uniform` | 4.707 | 9.083 | 16.316 | 19.206 | 8759 | 0.263 | 113.3 | 13.328 |
| `centered_magbin_weighted` | `centered_zv` | `magbin` | 4.755 | 9.212 | 16.572 | 19.437 | 8759 | 0.312 | 59.7 | 13.504 |
| `centered_default` | `centered_zv` | `uniform` | 4.755 | 9.213 | 16.573 | 19.440 | 8759 | 0.312 | 59.7 | 13.504 |
| `centered_power_prior_0p38` | `centered_zv` | `uniform` | 4.796 | 9.327 | 16.798 | 19.646 | 8759 | 0.354 | 34.9 | 13.660 |
| `paired_corr_0p7` | `paired_zv` | `uniform` | 6.214 | 8.202 | 13.589 | 16.387 | 1086 | 0.358 | 32.6 | 13.712 |
| `hybrid_magbin_weighted` | `hybrid` | `magbin` | 4.524 | 10.437 | 19.399 | 21.290 | 10508 | 0.390 | 22.7 | 14.592 |
| `hybrid_uniform` | `hybrid` | `uniform` | 4.524 | 10.437 | 19.400 | 21.292 | 10508 | 0.390 | 22.7 | 14.592 |
| `paired_min_db_250` | `paired_zv` | `uniform` | 4.900 | 10.843 | 19.740 | 24.906 | 2423 | 0.418 | 15.1 | 15.256 |
| `paired_dx_180` | `paired_zv` | `uniform` | 5.105 | 10.800 | 19.367 | 24.645 | 1776 | 0.447 | 11.8 | 15.347 |
| `hybrid_equal_source_magbin` | `hybrid` | `equal_source_magbin` | 4.599 | 12.062 | 22.609 | 25.129 | 10508 | 0.540 | 5.5 | 16.282 |
| `hybrid_equal_source` | `hybrid` | `equal_source` | 4.599 | 12.063 | 22.610 | 25.131 | 10508 | 0.540 | 5.5 | 16.282 |
| `paired_power_prior_0p28` | `paired_zv` | `uniform` | 4.551 | 12.485 | 23.360 | 26.906 | 1749 | 0.366 | 26.2 | 16.633 |
| `paired_magbin_weighted` | `paired_zv` | `magbin` | 4.587 | 12.602 | 23.587 | 27.109 | 1749 | 0.417 | 14.2 | 16.785 |
| `paired_default` | `paired_zv` | `uniform` | 4.587 | 12.610 | 23.602 | 27.113 | 1749 | 0.417 | 14.2 | 16.793 |
| `paired_pair_max_abs_db` | `paired_zv` | `uniform` | 4.621 | 12.663 | 23.679 | 27.206 | 1694 | 0.432 | 12.0 | 16.872 |
| `paired_power_prior_0p38` | `paired_zv` | `uniform` | 4.620 | 12.718 | 23.814 | 27.294 | 1749 | 0.462 | 8.4 | 16.933 |
| `paired_pair_max_db_dt` | `paired_zv` | `uniform` | 4.672 | 13.463 | 25.200 | 29.040 | 1728 | 0.422 | 13.2 | 17.703 |
| `paired_min_db_750` | `paired_zv` | `uniform` | 4.924 | 14.563 | 27.157 | 31.311 | 1206 | 0.429 | 11.3 | 18.995 |
| `centered_rad_12` | `centered_zv` | `uniform` | 9.668 | 10.552 | 16.705 | 20.150 | 7825 | 0.274 | 109.2 | 19.120 |
| `centered_rad_30` | `centered_zv` | `uniform` | 6.297 | 20.155 | 36.925 | 40.784 | 8478 | 0.465 | 7.8 | 25.606 |
| `centered_corr_0p7` | `centered_zv` | `uniform` | 16.675 | 23.056 | 29.047 | 35.216 | 3649 | 0.255 | 186.1 | 35.465 |

## Travel-Bin Means

| Variant | 0-30 | 30-60 | 60-90 | 90-120 | 120-150 |
|---|---:|---:|---:|---:|---:|
| `mag_gated_blend_p5_p50` | 4.838 | 3.726 | 5.139 | 7.389 | 13.494 |
| `centered_power_prior_0p28` | 6.479 | 3.847 | 3.989 | 8.204 | 16.316 |
| `centered_magbin_weighted` | 6.597 | 3.865 | 3.984 | 8.313 | 16.572 |
| `centered_default` | 6.596 | 3.865 | 3.984 | 8.314 | 16.573 |
| `centered_power_prior_0p38` | 6.700 | 3.882 | 3.979 | 8.411 | 16.798 |
| `paired_corr_0p7` | 9.967 | 4.540 | 4.947 | 6.344 | 11.277 |
| `hybrid_magbin_weighted` | 5.642 | 3.718 | 3.946 | 10.126 | 19.399 |
| `hybrid_uniform` | 5.641 | 3.718 | 3.946 | 10.126 | 19.400 |
| `paired_min_db_250` | 6.021 | 3.837 | 4.319 | 10.454 | 19.740 |
| `paired_dx_180` | 6.445 | 3.923 | 4.444 | 10.229 | 19.367 |
| `hybrid_equal_source_magbin` | 5.079 | 3.658 | 4.321 | 12.278 | 22.609 |
| `hybrid_equal_source` | 5.078 | 3.658 | 4.321 | 12.279 | 22.610 |
| `paired_power_prior_0p28` | 4.463 | 3.555 | 4.730 | 13.053 | 23.360 |
| `paired_magbin_weighted` | 4.546 | 3.573 | 4.727 | 13.161 | 23.587 |
| `paired_default` | 4.542 | 3.572 | 4.730 | 13.172 | 23.602 |
| `paired_pair_max_abs_db` | 4.571 | 3.585 | 4.757 | 13.218 | 23.679 |
| `paired_power_prior_0p38` | 4.613 | 3.587 | 4.729 | 13.277 | 23.814 |
| `paired_pair_max_db_dt` | 4.397 | 3.536 | 5.098 | 14.356 | 25.200 |
| `paired_min_db_750` | 4.545 | 3.536 | 5.688 | 15.820 | 27.157 |
| `centered_rad_12` | 16.705 | 5.991 | 9.107 | 10.223 | 6.934 |
| `centered_rad_30` | 6.259 | 3.522 | 8.647 | 23.212 | 36.925 |
| `centered_corr_0p7` | 28.673 | 8.966 | 17.503 | 26.678 | 27.053 |

## Selected Per-Log Metrics

| Log | Variant | RMSE | Bin RMSE | Worst Bin | b0 | b1 | b2 | b3 | b4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `log148_rear` | `mag_gated_blend_p5_p50` | 3.258 | 5.983 | 10.732 | 3.262 | 2.664 | 4.574 | 5.012 | 10.732 |
| `log148_rear` | `paired_default` | 4.172 | 14.615 | 27.113 | 2.791 | 2.740 | 5.222 | 17.037 | 27.113 |
| `log148_rear` | `centered_default` | 4.854 | 8.927 | 15.759 | 7.426 | 3.425 | 3.059 | 8.593 | 15.759 |
| `log148_rear` | `hybrid_equal_source` | 4.056 | 13.300 | 24.814 | 3.633 | 2.902 | 4.265 | 15.128 | 24.814 |
| `log149_rear` | `mag_gated_blend_p5_p50` | 6.247 | 10.316 | 18.767 | 7.334 | 5.613 | 5.624 | 7.934 | 18.767 |
| `log149_rear` | `paired_default` | 6.202 | 12.423 | 23.319 | 7.084 | 5.553 | 5.402 | 10.848 | 23.319 |
| `log149_rear` | `centered_default` | 6.490 | 10.702 | 19.440 | 7.833 | 5.766 | 5.463 | 8.384 | 19.440 |
| `log149_rear` | `hybrid_equal_source` | 6.647 | 12.640 | 23.574 | 8.094 | 5.773 | 5.315 | 10.771 | 23.574 |
| `log150_rear` | `mag_gated_blend_p5_p50` | 5.258 | 9.528 | 16.691 | 7.238 | 4.724 | 5.008 | 8.694 | 16.691 |
| `log150_rear` | `paired_default` | 5.261 | 11.303 | 20.537 | 7.101 | 4.666 | 5.012 | 10.941 | 20.537 |
| `log150_rear` | `centered_default` | 5.364 | 9.937 | 17.456 | 7.577 | 4.790 | 4.920 | 9.192 | 17.456 |
| `log150_rear` | `hybrid_equal_source` | 5.401 | 11.551 | 20.961 | 7.695 | 4.718 | 4.961 | 11.033 | 20.961 |
| `log151_rear` | `mag_gated_blend_p5_p50` | 3.952 | 9.605 | 17.809 | 4.467 | 3.288 | 3.465 | 10.068 | 17.809 |
| `log151_rear` | `paired_default` | 4.062 | 10.652 | 19.868 | 4.379 | 3.333 | 3.603 | 11.372 | 19.868 |
| `log151_rear` | `centered_default` | 4.035 | 9.778 | 18.122 | 4.635 | 3.345 | 3.454 | 10.246 | 18.122 |
| `log151_rear` | `hybrid_equal_source` | 4.121 | 10.609 | 19.770 | 4.594 | 3.368 | 3.561 | 11.261 | 19.770 |
| `log152_rear` | `mag_gated_blend_p5_p50` | 4.661 | 5.743 | 8.218 | 4.122 | 3.409 | 8.218 | 7.065 | 4.345 |
| `log152_rear` | `paired_default` | 3.349 | 12.563 | 24.083 | 2.638 | 2.367 | 4.146 | 13.395 | 24.083 |
| `log152_rear` | `centered_default` | 4.061 | 6.557 | 10.867 | 7.356 | 3.019 | 3.738 | 4.438 | 10.867 |
| `log152_rear` | `hybrid_equal_source` | 3.178 | 11.099 | 21.412 | 3.310 | 2.417 | 3.431 | 11.350 | 21.412 |
| `log153_rear` | `mag_gated_blend_p5_p50` | 3.115 | 6.627 | 12.618 | 2.606 | 2.658 | 3.945 | 5.561 | 12.618 |
| `log153_rear` | `paired_default` | 4.478 | 14.101 | 26.693 | 3.259 | 2.773 | 4.992 | 15.441 | 26.693 |
| `log153_rear` | `centered_default` | 3.723 | 9.376 | 17.795 | 4.751 | 2.846 | 3.269 | 9.029 | 17.795 |
| `log153_rear` | `hybrid_equal_source` | 4.189 | 13.176 | 25.131 | 3.143 | 2.772 | 4.396 | 14.129 | 25.131 |
