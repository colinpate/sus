# Slayer filtered-log corrected parent-reference experiment

This rerun preserves the earlier manual strategy—one absolute magnetic reference from each complete parent log reused by its filtered child chunks—but recalculates that reference after nuisance correction. The detector uses corrected magnitude, a 0.20-second bump window, the existing magnetic-band selector, `bump_mag_min=1000 mG`, and the Slayer 20-point/40-mm safety gates. Both solvers and nuisance correction are rerun from the same zero-offset starting point for every policy.

## Final adjusted-travel results

| Reference policy | Evaluation | Fallbacks | Child-mean RMSE | Parent-balanced RMSE | Pooled RMSE | Pooled child-centered RMSE | Pooled mean error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing registry parent reference | unmasked | 1/7 | 14.79 | 13.51 | 23.27 | 5.73 | 6.38 |
| Existing registry parent reference | dropout masked | 1/7 | 14.41 | 13.14 | 21.43 | 5.29 | 4.57 |
| Automatic per child | unmasked | 6/7 | 11.51 | 10.79 | 13.05 | 5.61 | 0.70 |
| Automatic per child | dropout masked | 6/7 | 11.21 | 10.52 | 11.88 | 5.14 | -0.06 |
| Recomputed corrected parent reference | unmasked | 1/7 | 15.06 | 13.88 | 20.78 | 5.79 | 6.30 |
| Recomputed corrected parent reference | dropout masked | 1/7 | 14.75 | 13.58 | 19.05 | 5.34 | 4.98 |

## Conclusions

- Recomputing parent references in corrected coordinates lowers pooled RMSE from 23.27 to 20.78 mm, but child-mean RMSE slightly worsens from 14.79 to 15.06 mm. It is not a reliable replacement for the old manual values.
- Automatic corrected-coordinate calibration per child is substantially better: 11.51 mm child-mean, 10.79 mm parent-balanced, and 13.05 mm pooled RMSE.
- Only `log-0145` has a parent reference supported by at least 20 selected points. The other four parents fall back to their own corrected-magnitude p8 zero; `log-0151` is the nearest miss with 17 points.
- The two `log-0147` children require very different local offsets. That branch instability is why one shared parent reference cannot solve both chunks.
- Production action: remove the seven stale fixed-reference overrides and let each filtered child use the guarded post-correction policy. Keep the fixed raw baselines and shared accelerometer rotation.

## Recomputed parent references

| Parent | Source | Reference x | Corrected mag | Chunks | Selected points | Reference error vs GT |
|---|---:|---:|---:|---:|---:|---:|
| log-0145 | detected | 65.52 mm | 3254.20 mG | 16 | 48 | 4.08 mm |
| log-0147 | p8_fallback | 0.00 mm | 1269.26 mG | 1 | 0 | — mm |
| log-0151 | p8_fallback | 0.00 mm | 1297.10 mG | 4 | 17 | — mm |
| log-0152 | p8_fallback | 0.00 mm | 1456.32 mG | 2 | 8 | — mm |
| log-0155 | p8_fallback | 0.00 mm | 1343.53 mG | 1 | 2 | — mm |

## Per-child final solver (unmasked)

| Child | Parent | Existing RMSE / mean error | Automatic RMSE / mean error | Recomputed-parent RMSE / mean error |
|---|---|---:|---:|---:|
| log-0145-filtered-c01 | log-0145 | 7.48 / -6.17 | 4.33 / +0.28 | 4.33 / +0.26 |
| log-0147-filtered-c01 | log-0147 | 14.67 / -13.32 | 14.50 / -13.14 | 15.01 / -13.68 |
| log-0147-filtered-c02 | log-0147 | 42.22 / +41.59 | 19.96 / +18.76 | 36.54 / +35.82 |
| log-0151-filtered-c01 | log-0151 | 9.28 / +7.36 | 8.25 / -6.10 | 14.33 / +13.05 |
| log-0152-filtered-c01 | log-0152 | 5.22 / -2.75 | 8.29 / -6.90 | 9.93 / -8.73 |
| log-0152-filtered-c02 | log-0152 | 9.96 / -8.25 | 10.51 / -8.89 | 10.51 / -8.89 |
| log-0155-filtered-c01 | log-0155 | 14.74 / -13.37 | 14.75 / -13.38 | 14.75 / -13.38 |

The prior grown-chunk artifact reported 23.55 mm pooled absolute RMSE and 5.70 mm pooled chunk-centered RMSE. Its magnetic model/reference application occurred before nuisance correction, so the current-policy reruns above are the meaningful absolute-error comparison; centered results should remain close because all three reference policies primarily change a scalar offset.

Detailed results are in `per_child_metrics.csv`, `aggregate_metrics.csv`, and `parent_references.csv`.
