# Post-merge front calibration-duration study

## Bottom line

The current pipeline preserves the practical calibration-time result: useful accuracy is available by 10–20 active seconds, and the typical full-recording error is near its best around 40 seconds. The exact curve is setup-dependent, so 40 seconds should remain a default rather than a universal optimum. Longer accumulation is most useful as a robustness option when the calibration-quality signal remains poor.

The earlier methodological explanation also survives. Same-window error can rise while full-log error improves because longer windows span more travel states. The supervised power oracle shows the same expanding-window behavior, while fixed-support and downstream results do not support the interpretation that more data simply makes the learner worse.

## Design

- Current merged front pipeline and production magnetic bad-mask behavior.
- Eleven Stumpjumper/pod-v2 logs at 5, 10, 20, 40, 60, and 120 active seconds, with four deterministic nested centers.
- Five independent recordings each from Jamaal's TR11, Harry's TR11, and Slayer at 5–60 seconds.
- Full downstream solves on all Stumpjumper conditions except 60 seconds, plus two paired centers at 10, 40, and 60 seconds for every non-Stumpjumper log.
- Logs are the independent analysis units; repeats are collapsed within log before setup medians are taken.
- The selection/evaluation activity mask remains the reference-derived `boring_mask`.

## Calibration-stage results

Full-log aligned RMSE for the self-supervised magnetic curve:

| Setup | 5 s | 10 s | 20 s | 40 s | 60 s | 120 s |
|---|---:|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 | 4.62 | 4.29 | 4.17 | 3.78 | 3.93 | 4.23 |
| TR11 / pod v2 (Jamaal) | 10.00 | 9.32 | 9.22 | 8.94 | 8.86 | — |
| TR11 2025 / pod v2 (Harry) | 8.48 | 7.42 | 6.96 | 7.31 | 7.80 | — |
| Slayer / pod v2 | 6.99 | 6.14 | 5.13 | 6.14 | 5.82 | — |

Short-window failures under the production bad mask were uncommon but informative:

| Setup | 5 s failure | 10 s failure | 20 s failure |
|---|---:|---:|---:|
| Stumpjumper / pod v2 | 0.0% | 0.0% | 0.0% |
| TR11 / pod v2 (Jamaal) | 5.0% | 0.0% | 0.0% |
| TR11 2025 / pod v2 (Harry) | 5.0% | 5.0% | 0.0% |
| Slayer / pod v2 | 0.0% | 0.0% | 0.0% |

## Final solver results

Full-log aligned RMSE for the final solved output:

| Setup | 5 s | 10 s | 20 s | 40 s | 60 s | 120 s |
|---|---:|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 | 3.98 | 3.20 | 3.36 | 2.93 | — | 2.65 |
| TR11 / pod v2 (Jamaal) | — | 7.99 | — | 6.79 | 6.34 | — |
| TR11 2025 / pod v2 (Harry) | — | 5.57 | — | 5.36 | 5.47 | — |
| Slayer / pod v2 | — | 7.62 | — | 5.07 | 4.82 | — |

For Stumpjumper, the final solve changes from 3.98 mm at 5 seconds to 2.93 mm at 40 seconds and 2.65 mm at 120 seconds. The setup-balanced extension tests whether the same plateau appears on the other geometries rather than inferring that from pooled data.

The apparent same-window reversal is still an evaluation-support effect. On Stumpjumper, raw magnetic RMSE on each window's own samples rises from 3.54 mm at 5 seconds to 4.21 mm at 120 seconds. When all calibrations are instead scored on the identical central 5-second core, raw magnetic RMSE changes from 3.54 to 2.92 mm, and final solved RMSE changes from 3.02 to 2.01 mm. The remaining non-monotonicity and the oracle's own-window rise still indicate a real one-dimensional curve compromise in addition to the support artifact.

## Sensitivity to the pipeline revision

The pre-merge and current studies use the same logs, duration labels, repeat identities, and deterministic random fractions. The activity mask changed with the pipeline, however, so their active-time coordinates do not always resolve to identical physical samples. The comparison below is therefore a cohort-level sensitivity analysis, not a pure paired code ablation.

| Duration | Pre-merge mag | Current mag | Change | Pre-merge solved | Current solved | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 5.10 | 4.62 | -0.49 | 4.52 | 3.98 | -0.54 |
| 10 s | 4.39 | 4.29 | -0.09 | 4.17 | 3.20 | -0.97 |
| 20 s | 4.73 | 4.17 | -0.56 | 3.68 | 3.36 | -0.32 |
| 40 s | 4.39 | 3.78 | -0.61 | 3.29 | 2.93 | -0.36 |
| 120 s | 4.10 | 4.23 | 0.13 | 3.24 | 2.65 | -0.59 |

## Implications

1. Keep 10 active seconds as an earliest attempt, but gate release on accepted motion chunks and magnetic/travel-proxy coverage rather than time alone.
2. Keep approximately 40 active seconds as the normal default. Continue collecting or retry on another block when the quality signal is weak.
3. Do not select duration from own-window RMSE; use full-log/held-out behavior, fixed-support diagnostics, and final solved error.
4. Report setup-stratified curves in the paper. A single pooled duration curve hides meaningful differences in both error floor and optimum.
5. Treat the existing pre-merge mechanism study as supporting analysis, while using this report's current-pipeline tables for numerical recommendations.

## Artifacts

- `raw_setup_summary.csv`: log-balanced calibration-stage metrics by setup.
- `solver_setup_summary.csv`: log-balanced stagewise solver metrics by setup.
- `raw_failure_summary.csv`: planned-window failure rates, including no-chunk failures.
- `raw_premerge_comparison.csv` and `solver_premerge_comparison.csv`: cohort-level sensitivity tables.
- `postmerge_duration_by_setup.png`: paper-oriented setup-stratified learning curves.
