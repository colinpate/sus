# cross-setup-front-pod-v2-full-v4: cross-setup magnetometer calibration transfer

## Bottom line

The expanded experiment finds a positive self-supervised transfer penalty in 11 of 12 directed cross-setup cells; 6 have a crossed-bootstrap interval entirely above zero. The penalty ranges from -1.50 to +47.00 mm, demonstrating that transfer is strongly directional and that pooled cross-setup averages are not sufficient.

## Design

The experiment evaluates all 38 source logs against all 38 target logs for 3 trainers: self-supervised, oracle-power, oracle-binned-median. It covers 4 setup cohorts and 36 independent recording units. Each calibration is trained once on its complete source log; the matrix retains the target log's own calibration and within-setup transfers as controls. All 4,332 evaluations completed successfully.

The primary comparison is each transfer's aligned error minus the target log's own calibration error from the same trainer. Positive values favor independent target/per-recording calibration. Aligned normalized RMSE controls for different travel distributions. Confidence intervals use a crossed bootstrap that independently resamples source and target recording units. Derived chunks from the same parent recording are collapsed within unit pairs and resampled as one unit.

## Target-specific baselines

| Trainer | Setup | Logs | Independent units | Aligned RMSE | Normalized RMSE | Anchored RMSE |
|---|---|---:|---:|---:|---:|---:|
| self-supervised | Stumpjumper / pod v2 | 11 | 11 | 4.07 mm | 0.191 | 5.10 mm |
| self-supervised | TR11 / pod v2 (Jamaal) | 6 | 6 | 11.35 mm | 0.389 | 11.88 mm |
| self-supervised | TR11 2025 / pod v2 (Harry) | 14 | 14 | 7.57 mm | 0.230 | 12.39 mm |
| self-supervised | Slayer / pod v2 | 7 | 5 | 6.89 mm | 0.378 | 9.37 mm |
| oracle-power | Stumpjumper / pod v2 | 11 | 11 | 3.05 mm | 0.135 | 3.05 mm |
| oracle-power | TR11 / pod v2 (Jamaal) | 6 | 6 | 7.87 mm | 0.274 | 7.87 mm |
| oracle-power | TR11 2025 / pod v2 (Harry) | 14 | 14 | 5.28 mm | 0.167 | 5.28 mm |
| oracle-power | Slayer / pod v2 | 7 | 5 | 4.12 mm | 0.256 | 4.12 mm |
| oracle-binned-median | Stumpjumper / pod v2 | 11 | 11 | 2.91 mm | 0.132 | 2.92 mm |
| oracle-binned-median | TR11 / pod v2 (Jamaal) | 6 | 6 | 7.57 mm | 0.264 | 7.64 mm |
| oracle-binned-median | TR11 2025 / pod v2 (Harry) | 14 | 14 | 4.93 mm | 0.159 | 4.95 mm |
| oracle-binned-median | Slayer / pod v2 | 7 | 5 | 3.67 mm | 0.213 | 3.68 mm |

## Self-supervised transfer results

| Source → target | Type | Source/target units | Transfer RMSE | Penalty vs target calibration (95% CI) | Δ normalized RMSE | Unit pairs worse | Anchored RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 → Stumpjumper / pod v2 | same setup | 11/11 | 4.25 mm | +0.23 [-0.51, +0.64] mm | +0.012 | 50% | 4.84 mm |
| Stumpjumper / pod v2 → TR11 / pod v2 (Jamaal) | cross setup | 11/6 | 20.19 mm | +10.72 [+2.21, +11.98] mm | +0.364 | 100% | 43.75 mm |
| Stumpjumper / pod v2 → TR11 2025 / pod v2 (Harry) | cross setup | 11/14 | 21.39 mm | +12.84 [+10.79, +15.88] mm | +0.423 | 100% | 26.37 mm |
| Stumpjumper / pod v2 → Slayer / pod v2 | cross setup | 11/5 | 10.00 mm | +3.76 [+0.64, +8.92] mm | +0.230 | 100% | 27.08 mm |
| TR11 / pod v2 (Jamaal) → Stumpjumper / pod v2 | cross setup | 6/11 | 50.99 mm | +47.00 [+35.02, +57.53] mm | +2.133 | 100% | 93.45 mm |
| TR11 / pod v2 (Jamaal) → TR11 / pod v2 (Jamaal) | same setup | 6/6 | 10.19 mm | -0.51 [-2.56, +0.80] mm | -0.018 | 43% | 11.13 mm |
| TR11 / pod v2 (Jamaal) → TR11 2025 / pod v2 (Harry) | cross setup | 6/14 | 9.39 mm | +1.78 [-0.30, +2.77] mm | +0.049 | 67% | 17.34 mm |
| TR11 / pod v2 (Jamaal) → Slayer / pod v2 | cross setup | 6/5 | 10.52 mm | +3.98 [-1.01, +9.59] mm | +0.216 | 80% | 11.29 mm |
| TR11 2025 / pod v2 (Harry) → Stumpjumper / pod v2 | cross setup | 14/11 | 35.99 mm | +31.34 [+28.10, +35.73] mm | +1.477 | 100% | 55.78 mm |
| TR11 2025 / pod v2 (Harry) → TR11 / pod v2 (Jamaal) | cross setup | 14/6 | 8.97 mm | -1.50 [-2.96, +0.06] mm | -0.060 | 26% | 11.85 mm |
| TR11 2025 / pod v2 (Harry) → TR11 2025 / pod v2 (Harry) | same setup | 14/14 | 7.54 mm | +0.02 [-0.84, +1.26] mm | +0.001 | 53% | 13.39 mm |
| TR11 2025 / pod v2 (Harry) → Slayer / pod v2 | cross setup | 14/5 | 7.50 mm | +1.13 [-3.28, +4.59] mm | +0.062 | 79% | 9.45 mm |
| Slayer / pod v2 → Stumpjumper / pod v2 | cross setup | 5/11 | 20.51 mm | +16.41 [+8.84, +22.38] mm | +0.759 | 100% | 21.25 mm |
| Slayer / pod v2 → TR11 / pod v2 (Jamaal) | cross setup | 5/6 | 10.36 mm | +0.49 [-3.03, +6.29] mm | +0.017 | 60% | 13.00 mm |
| Slayer / pod v2 → TR11 2025 / pod v2 (Harry) | cross setup | 5/14 | 7.99 mm | +0.63 [-1.02, +3.61] mm | +0.017 | 57% | 12.68 mm |
| Slayer / pod v2 → Slayer / pod v2 | same setup | 5/5 | 6.56 mm | +0.09 [-1.96, +1.80] mm | +0.004 | 50% | 9.73 mm |

## Main findings

1. **Most setup changes penalize a frozen calibration.** 11 of 12 directed self-supervised cross-setup cells have a positive median penalty, compared with a same-setup range of -0.51 to +0.23 mm.
2. **The strongest failure is TR11 / pod v2 (Jamaal) → Stumpjumper / pod v2.** Its median penalty is +47.00 mm [+35.02, +57.53], and 100% of independent source-target unit pairs are worse than target-specific calibration.
3. **The easiest transfer direction is TR11 2025 / pod v2 (Harry) → TR11 / pod v2 (Jamaal).** Its median penalty is -1.50 mm [-2.96, +0.06]. This direction should be interpreted separately rather than used to justify universal transfer.
4. **Oracle comparisons distinguish physical transfer mismatch from learner noise.** The oracle trainers directly observe reference travel on the source log. Agreement with the self-supervised direction therefore supports a setup-specific mapping; disagreement identifies directions where self-supervised estimation variance affects the comparison.
5. **Absolute and normalized metrics remain necessary.** Normalized error checks that results are not just caused by different travel ranges, while anchored error exposes offset and mounting-reference transfer in addition to curve shape.

### Oracle cross-setup summary

| Trainer | Positive median penalties | CIs above zero | Penalty range |
|---|---:|---:|---:|
| self-supervised | 11/12 | 6/12 | -1.50 to +47.00 mm |
| oracle-power | 12/12 | 12/12 | +1.40 to +42.09 mm |
| oracle-binned-median | 12/12 | 12/12 | +1.64 to +27.08 mm |

## Magnetic and travel support

| Setup | Logs | Independent units | Median magnetic p5–p95 | Median magnetic span | Median travel span |
|---|---:|---:|---:|---:|---:|
| Stumpjumper / pod v2 | 11 | 11 | 483–5189 | 4758 | 76.1 mm |
| TR11 / pod v2 (Jamaal) | 6 | 6 | 1004–2131 | 1099 | 88.0 mm |
| TR11 2025 / pod v2 (Harry) | 14 | 14 | 1079–4171 | 3119 | 108.6 mm |
| Slayer / pod v2 | 7 | 5 | 1327–3210 | 1882 | 59.8 mm |

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
