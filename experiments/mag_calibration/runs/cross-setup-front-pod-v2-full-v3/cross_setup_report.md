# cross-setup-front-pod-v2-full-v3: cross-setup magnetometer calibration transfer

## Bottom line

The expanded experiment finds a positive self-supervised transfer penalty in 11 of 12 directed cross-setup cells; 6 have a crossed-bootstrap interval entirely above zero. The penalty ranges from -0.91 to +45.87 mm, demonstrating that transfer is strongly directional and that pooled cross-setup averages are not sufficient.

## Design

The experiment evaluates all 38 source logs against all 38 target logs for 3 trainers: self-supervised, oracle-power, oracle-binned-median. It covers 4 setup cohorts and 36 independent recording units. Each calibration is trained once on its complete source log; the matrix retains the target log's own calibration and within-setup transfers as controls. All 4,332 evaluations completed successfully.

The primary comparison is each transfer's aligned error minus the target log's own calibration error from the same trainer. Positive values favor independent target/per-recording calibration. Aligned normalized RMSE controls for different travel distributions. Confidence intervals use a crossed bootstrap that independently resamples source and target recording units. Derived chunks from the same parent recording are collapsed within unit pairs and resampled as one unit.

## Target-specific baselines

| Trainer | Setup | Logs | Independent units | Aligned RMSE | Normalized RMSE | Anchored RMSE |
|---|---|---:|---:|---:|---:|---:|
| self-supervised | Stumpjumper / pod v2 | 11 | 11 | 3.81 mm | 0.175 | 4.63 mm |
| self-supervised | TR11 / pod v2 (Jamaal) | 6 | 6 | 10.40 mm | 0.375 | 11.46 mm |
| self-supervised | TR11 2025 / pod v2 (Harry) | 14 | 14 | 7.26 mm | 0.233 | 12.38 mm |
| self-supervised | Slayer / pod v2 | 7 | 5 | 8.87 mm | 0.488 | 13.40 mm |
| oracle-power | Stumpjumper / pod v2 | 11 | 11 | 2.74 mm | 0.129 | 2.74 mm |
| oracle-power | TR11 / pod v2 (Jamaal) | 6 | 6 | 7.65 mm | 0.278 | 7.65 mm |
| oracle-power | TR11 2025 / pod v2 (Harry) | 14 | 14 | 5.39 mm | 0.174 | 5.39 mm |
| oracle-power | Slayer / pod v2 | 7 | 5 | 4.02 mm | 0.250 | 4.02 mm |
| oracle-binned-median | Stumpjumper / pod v2 | 11 | 11 | 2.71 mm | 0.125 | 2.71 mm |
| oracle-binned-median | TR11 / pod v2 (Jamaal) | 6 | 6 | 7.39 mm | 0.270 | 7.47 mm |
| oracle-binned-median | TR11 2025 / pod v2 (Harry) | 14 | 14 | 5.08 mm | 0.165 | 5.09 mm |
| oracle-binned-median | Slayer / pod v2 | 7 | 5 | 3.53 mm | 0.204 | 3.55 mm |

## Self-supervised transfer results

| Source → target | Type | Source/target units | Transfer RMSE | Penalty vs target calibration (95% CI) | Δ normalized RMSE | Unit pairs worse | Anchored RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 → Stumpjumper / pod v2 | same setup | 11/11 | 4.16 mm | +0.16 [-0.38, +0.67] mm | +0.008 | 55% | 4.65 mm |
| Stumpjumper / pod v2 → TR11 / pod v2 (Jamaal) | cross setup | 11/6 | 19.78 mm | +10.25 [+2.57, +12.41] mm | +0.371 | 100% | 43.05 mm |
| Stumpjumper / pod v2 → TR11 2025 / pod v2 (Harry) | cross setup | 11/14 | 20.77 mm | +12.46 [+10.64, +15.12] mm | +0.413 | 100% | 25.41 mm |
| Stumpjumper / pod v2 → Slayer / pod v2 | cross setup | 11/5 | 9.94 mm | +2.45 [+0.38, +8.88] mm | +0.124 | 100% | 27.30 mm |
| TR11 / pod v2 (Jamaal) → Stumpjumper / pod v2 | cross setup | 6/11 | 49.68 mm | +45.87 [+35.40, +55.80] mm | +2.141 | 100% | 98.98 mm |
| TR11 / pod v2 (Jamaal) → TR11 / pod v2 (Jamaal) | same setup | 6/6 | 9.59 mm | -0.21 [-1.83, +0.84] mm | -0.007 | 40% | 11.03 mm |
| TR11 / pod v2 (Jamaal) → TR11 2025 / pod v2 (Harry) | cross setup | 6/14 | 8.88 mm | +1.00 [-0.44, +2.49] mm | +0.028 | 65% | 17.41 mm |
| TR11 / pod v2 (Jamaal) → Slayer / pod v2 | cross setup | 6/5 | 10.71 mm | +3.68 [-0.94, +7.33] mm | +0.213 | 77% | 11.49 mm |
| TR11 2025 / pod v2 (Harry) → Stumpjumper / pod v2 | cross setup | 14/11 | 35.56 mm | +31.74 [+28.38, +35.63] mm | +1.494 | 100% | 59.64 mm |
| TR11 2025 / pod v2 (Harry) → TR11 / pod v2 (Jamaal) | cross setup | 14/6 | 9.01 mm | -0.91 [-2.24, +0.27] mm | -0.038 | 30% | 11.80 mm |
| TR11 2025 / pod v2 (Harry) → TR11 2025 / pod v2 (Harry) | same setup | 14/14 | 7.25 mm | -0.03 [-0.94, +1.18] mm | -0.001 | 51% | 13.27 mm |
| TR11 2025 / pod v2 (Harry) → Slayer / pod v2 | cross setup | 14/5 | 7.62 mm | +1.33 [-3.31, +2.60] mm | +0.061 | 66% | 9.58 mm |
| Slayer / pod v2 → Stumpjumper / pod v2 | cross setup | 5/11 | 26.52 mm | +22.67 [+9.12, +27.06] mm | +1.088 | 100% | 31.27 mm |
| Slayer / pod v2 → TR11 / pod v2 (Jamaal) | cross setup | 5/6 | 13.43 mm | +4.07 [-2.98, +8.14] mm | +0.144 | 67% | 17.18 mm |
| Slayer / pod v2 → TR11 2025 / pod v2 (Harry) | cross setup | 5/14 | 10.38 mm | +2.60 [-0.33, +8.19] mm | +0.085 | 74% | 14.46 mm |
| Slayer / pod v2 → Slayer / pod v2 | same setup | 5/5 | 8.11 mm | +0.49 [-3.57, +2.72] mm | +0.017 | 45% | 13.26 mm |

## Main findings

1. **Most setup changes penalize a frozen calibration.** 11 of 12 directed self-supervised cross-setup cells have a positive median penalty, compared with a same-setup range of -0.21 to +0.49 mm.
2. **The strongest failure is TR11 / pod v2 (Jamaal) → Stumpjumper / pod v2.** Its median penalty is +45.87 mm [+35.40, +55.80], and 100% of independent source-target unit pairs are worse than target-specific calibration.
3. **The easiest transfer direction is TR11 2025 / pod v2 (Harry) → TR11 / pod v2 (Jamaal).** Its median penalty is -0.91 mm [-2.24, +0.27]. This direction should be interpreted separately rather than used to justify universal transfer.
4. **Oracle comparisons distinguish physical transfer mismatch from learner noise.** The oracle trainers directly observe reference travel on the source log. Agreement with the self-supervised direction therefore supports a setup-specific mapping; disagreement identifies directions where self-supervised estimation variance affects the comparison.
5. **Absolute and normalized metrics remain necessary.** Normalized error checks that results are not just caused by different travel ranges, while anchored error exposes offset and mounting-reference transfer in addition to curve shape.

### Oracle cross-setup summary

| Trainer | Positive median penalties | CIs above zero | Penalty range |
|---|---:|---:|---:|
| self-supervised | 11/12 | 6/12 | -0.91 to +45.87 mm |
| oracle-power | 12/12 | 12/12 | +1.09 to +47.66 mm |
| oracle-binned-median | 12/12 | 12/12 | +1.33 to +27.37 mm |

## Magnetic and travel support

| Setup | Logs | Independent units | Median magnetic p5–p95 | Median magnetic span | Median travel span |
|---|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 | 11 | 11 | 530–5922 | 5399 | 74.6 mm |
| TR11 / pod v2 (Jamaal) | 6 | 6 | 1021–2174 | 1138 | 88.3 mm |
| TR11 2025 / pod v2 (Harry) | 14 | 14 | 1093–4300 | 3224 | 110.8 mm |
| Slayer / pod v2 | 7 | 5 | 1325–3298 | 1973 | 61.3 mm |

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
