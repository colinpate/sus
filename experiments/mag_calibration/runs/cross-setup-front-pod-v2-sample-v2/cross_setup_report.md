# cross-setup-front-pod-v2-sample-v2: cross-setup magnetometer calibration transfer

## Bottom line

The expanded experiment finds a positive self-supervised transfer penalty in 11 of 12 directed cross-setup cells; 5 have a crossed-bootstrap interval entirely above zero. The penalty ranges from -1.31 to +45.44 mm, demonstrating that transfer is strongly directional and that pooled cross-setup averages are not sufficient.

## Design

The experiment evaluates all 20 source logs against all 20 target logs for 3 trainers: self-supervised, oracle-power, oracle-binned-median. It covers 4 setup cohorts and 20 independent recording units. Each calibration is trained once on its complete source log; the matrix retains the target log's own calibration and within-setup transfers as controls. All 1,200 evaluations completed successfully. The cohort is a frozen setup-stratified random sample of 5 independent recordings per setup using seed 20260916. The screening run was used to verify cache freshness and failures, not to select by accuracy.

The primary comparison is each transfer's aligned error minus the target log's own calibration error from the same trainer. Positive values favor independent target/per-recording calibration. Aligned normalized RMSE controls for different travel distributions. Confidence intervals use a crossed bootstrap that independently resamples source and target recording units. Derived chunks from the same parent recording are collapsed within unit pairs and resampled as one unit.

## Target-specific baselines

| Trainer | Setup | Logs | Independent units | Aligned RMSE | Normalized RMSE | Anchored RMSE |
|---|---|---:|---:|---:|---:|---:|
| self-supervised | Stumpjumper / pod v2 | 5 | 5 | 3.85 mm | 0.175 | 4.45 mm |
| self-supervised | TR11 / pod v2 (Jamaal) | 5 | 5 | 10.80 mm | 0.377 | 12.93 mm |
| self-supervised | TR11 2025 / pod v2 (Harry) | 5 | 5 | 7.45 mm | 0.242 | 12.87 mm |
| self-supervised | Slayer / pod v2 | 5 | 5 | 8.87 mm | 0.488 | 13.40 mm |
| oracle-power | Stumpjumper / pod v2 | 5 | 5 | 2.74 mm | 0.143 | 2.74 mm |
| oracle-power | TR11 / pod v2 (Jamaal) | 5 | 5 | 8.02 mm | 0.302 | 8.02 mm |
| oracle-power | TR11 2025 / pod v2 (Harry) | 5 | 5 | 5.48 mm | 0.173 | 5.48 mm |
| oracle-power | Slayer / pod v2 | 5 | 5 | 3.88 mm | 0.228 | 3.88 mm |
| oracle-binned-median | Stumpjumper / pod v2 | 5 | 5 | 2.71 mm | 0.134 | 2.71 mm |
| oracle-binned-median | TR11 / pod v2 (Jamaal) | 5 | 5 | 7.75 mm | 0.292 | 7.83 mm |
| oracle-binned-median | TR11 2025 / pod v2 (Harry) | 5 | 5 | 5.17 mm | 0.163 | 5.21 mm |
| oracle-binned-median | Slayer / pod v2 | 5 | 5 | 3.34 mm | 0.191 | 3.35 mm |

## Self-supervised transfer results

| Source → target | Type | Source/target units | Transfer RMSE | Penalty vs target calibration (95% CI) | Δ normalized RMSE | Unit pairs worse | Anchored RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 → Stumpjumper / pod v2 | same setup | 5/5 | 3.96 mm | +0.01 [-0.86, +0.49] mm | +0.000 | 50% | 4.13 mm |
| Stumpjumper / pod v2 → TR11 / pod v2 (Jamaal) | cross setup | 5/5 | 20.24 mm | +10.24 [+2.37, +12.27] mm | +0.359 | 100% | 42.42 mm |
| Stumpjumper / pod v2 → TR11 2025 / pod v2 (Harry) | cross setup | 5/5 | 20.17 mm | +12.44 [+11.31, +16.11] mm | +0.405 | 100% | 22.16 mm |
| Stumpjumper / pod v2 → Slayer / pod v2 | cross setup | 5/5 | 10.23 mm | +2.56 [-2.95, +8.84] mm | +0.130 | 80% | 27.50 mm |
| TR11 / pod v2 (Jamaal) → Stumpjumper / pod v2 | cross setup | 5/5 | 49.53 mm | +45.44 [+37.95, +59.05] mm | +2.183 | 100% | 100.98 mm |
| TR11 / pod v2 (Jamaal) → TR11 / pod v2 (Jamaal) | same setup | 5/5 | 10.56 mm | +0.06 [-1.44, +1.05] mm | +0.004 | 35% | 12.47 mm |
| TR11 / pod v2 (Jamaal) → TR11 2025 / pod v2 (Harry) | cross setup | 5/5 | 8.35 mm | +0.77 [-0.40, +6.04] mm | +0.021 | 68% | 15.54 mm |
| TR11 / pod v2 (Jamaal) → Slayer / pod v2 | cross setup | 5/5 | 10.56 mm | +2.19 [-2.25, +7.78] mm | +0.147 | 64% | 10.60 mm |
| TR11 2025 / pod v2 (Harry) → Stumpjumper / pod v2 | cross setup | 5/5 | 38.70 mm | +34.65 [+22.64, +43.10] mm | +1.658 | 100% | 62.71 mm |
| TR11 2025 / pod v2 (Harry) → TR11 / pod v2 (Jamaal) | cross setup | 5/5 | 9.41 mm | -1.31 [-2.91, -0.15] mm | -0.051 | 8% | 12.27 mm |
| TR11 2025 / pod v2 (Harry) → TR11 2025 / pod v2 (Harry) | same setup | 5/5 | 7.04 mm | -0.34 [-1.20, +3.09] mm | -0.011 | 50% | 12.36 mm |
| TR11 2025 / pod v2 (Harry) → Slayer / pod v2 | cross setup | 5/5 | 7.27 mm | +0.98 [-5.03, +3.15] mm | +0.056 | 44% | 8.24 mm |
| Slayer / pod v2 → Stumpjumper / pod v2 | cross setup | 5/5 | 26.84 mm | +22.68 [+9.17, +47.24] mm | +1.088 | 100% | 30.79 mm |
| Slayer / pod v2 → TR11 / pod v2 (Jamaal) | cross setup | 5/5 | 15.10 mm | +4.42 [-3.35, +9.25] mm | +0.153 | 64% | 17.53 mm |
| Slayer / pod v2 → TR11 2025 / pod v2 (Harry) | cross setup | 5/5 | 10.29 mm | +3.17 [-0.29, +10.99] mm | +0.111 | 80% | 15.46 mm |
| Slayer / pod v2 → Slayer / pod v2 | same setup | 5/5 | 7.59 mm | +0.49 [-5.67, +6.39] mm | +0.017 | 45% | 13.15 mm |

## Main findings

1. **Most setup changes penalize a frozen calibration.** 11 of 12 directed self-supervised cross-setup cells have a positive median penalty, compared with a same-setup range of -0.34 to +0.49 mm.
2. **The strongest failure is TR11 / pod v2 (Jamaal) → Stumpjumper / pod v2.** Its median penalty is +45.44 mm [+37.95, +59.05], and 100% of independent source-target unit pairs are worse than target-specific calibration.
3. **The easiest transfer direction is TR11 2025 / pod v2 (Harry) → TR11 / pod v2 (Jamaal).** Its median penalty is -1.31 mm [-2.91, -0.15]. This direction should be interpreted separately rather than used to justify universal transfer.
4. **Oracle comparisons distinguish physical transfer mismatch from learner noise.** The oracle trainers directly observe reference travel on the source log. Agreement with the self-supervised direction therefore supports a setup-specific mapping; disagreement identifies directions where self-supervised estimation variance affects the comparison.
5. **Absolute and normalized metrics remain necessary.** Normalized error checks that results are not just caused by different travel ranges, while anchored error exposes offset and mounting-reference transfer in addition to curve shape.

### Oracle cross-setup summary

| Trainer | Positive median penalties | CIs above zero | Penalty range |
|---|---:|---:|---:|
| self-supervised | 11/12 | 5/12 | -1.31 to +45.44 mm |
| oracle-power | 12/12 | 12/12 | +1.10 to +44.96 mm |
| oracle-binned-median | 12/12 | 12/12 | +1.24 to +27.23 mm |

## Magnetic and travel support

| Setup | Logs | Independent units | Median magnetic p5–p95 | Median magnetic span | Median travel span |
|---|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 | 5 | 5 | 554–5922 | 5399 | 74.6 mm |
| TR11 / pod v2 (Jamaal) | 5 | 5 | 1012–2106 | 1095 | 90.0 mm |
| TR11 2025 / pod v2 (Harry) | 5 | 5 | 1077–4301 | 3242 | 109.4 mm |
| Slayer / pod v2 | 5 | 5 | 1338–3363 | 2013 | 61.3 mm |

Large support differences explain some directional asymmetry: a curve trained on a narrow magnetic interval may extrapolate or clip when transferred to a wider target interval. Support overlap is not sufficient by itself, because different magnet placement and fork geometry can still produce a different curve within overlapping ranges.

## Paper implications

- Report the directed setup matrix rather than one pooled transfer statistic; source and target roles are not interchangeable.
- Use the target-specific diagonal and within-setup transfer as controls, so target difficulty and ordinary recording variation are separated from setup mismatch.
- Treat the two supervised oracles as a physical/curve-family control rather than as a deployable method.
- Keep aligned and normalized error primary, with anchored error as the production-oriented secondary measure.

## Pipeline implications

- Do not silently reuse a calibration across an unknown setup. Require setup identity or self-calibrate on the current recording.
- Store the calibration's observed magnetic support and reject unsupported extrapolation.
- Validate any warm-start or population prior with current-recording IMU constraints before accepting it.
- Use this matrix to develop a compatibility score from support overlap, curve parameters, and held-out self-supervised residuals.

## Limitations and next step

This experiment evaluates the magnetic mapping before downstream fusion and uses full-log source calibration. It uses the reference-derived `boring_mask` for evaluation; oracle curves use reference travel only from their source log. The confidence intervals are descriptive because these recordings were not collected as a prospectively held-out cohort. The next highest-value experiment is a smaller downstream-solver transfer study using representative source calibrations from each setup.
