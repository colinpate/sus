# Cross-setup magnetometer calibration transfer

## Bottom line

Full-log calibrations transfer well between pod v1 and pod v2 on the same Stumpjumper, but do not transfer between the Stumpjumper and TR11. The bike/setup change produces a far larger penalty than either log-to-log variation within a setup or the pod-generation change on the same bike. This directly supports per-setup calibration and strengthens the motivation for automatic per-recording self-calibration.

## Design

The experiment evaluates all 24 source logs against all 24 target logs for 3 trainers: self-supervised, oracle-power, oracle-binned-median. Each calibration is trained once on its complete source log. The matrix contains the target log's own calibration, same-setup transfers, and all six directed cross-setup transfers. All 1,728 evaluations completed successfully.

The primary comparison is each transfer's aligned error minus the target log's own calibration error. Positive values mean that independent target/per-recording calibration is better. Aligned normalized RMSE is retained to ensure that conclusions are not caused only by the setups' different travel ranges. Confidence intervals use a crossed bootstrap that independently resamples source and target logs; they are descriptive because the available logs are not a prospectively held-out cohort.

## Per-log baselines

| Trainer | Setup | Logs | Aligned RMSE | Normalized RMSE | Anchored RMSE |
|---|---|---:|---:|---:|---:|
| self-supervised | Stumpjumper / pod v1 | 7 | 5.77 mm | 0.251 | 12.06 mm |
| self-supervised | Stumpjumper / pod v2 | 11 | 4.58 mm | 0.205 | 8.16 mm |
| self-supervised | TR11 / pod v2 | 6 | 10.33 mm | 0.370 | 15.97 mm |
| oracle-power | Stumpjumper / pod v1 | 7 | 5.42 mm | 0.239 | 5.42 mm |
| oracle-power | Stumpjumper / pod v2 | 11 | 3.00 mm | 0.138 | 3.00 mm |
| oracle-power | TR11 / pod v2 | 6 | 7.69 mm | 0.276 | 7.69 mm |
| oracle-binned-median | Stumpjumper / pod v1 | 7 | 5.49 mm | 0.239 | 5.50 mm |
| oracle-binned-median | Stumpjumper / pod v2 | 11 | 2.88 mm | 0.130 | 2.89 mm |
| oracle-binned-median | TR11 / pod v2 | 6 | 7.42 mm | 0.267 | 7.50 mm |

## Self-supervised transfer results

| Source → target | Type | Transfer RMSE | Penalty vs target calibration (95% CI) | Δ normalized RMSE | Transfer worse | Anchored RMSE |
|---|---|---:|---:|---:|---:|---:|
| Stumpjumper / pod v1 → Stumpjumper / pod v1 | same setup | 5.82 mm | +0.31 [-0.23, +0.66] mm | +0.014 | 71% | 11.83 mm |
| Stumpjumper / pod v1 → Stumpjumper / pod v2 | cross setup | 4.46 mm | -0.05 [-0.90, +0.52] mm | -0.002 | 36% | 8.25 mm |
| Stumpjumper / pod v1 → TR11 / pod v2 | cross setup | 19.47 mm | +9.75 [+2.70, +12.69] mm | +0.350 | 98% | 40.34 mm |
| Stumpjumper / pod v2 → Stumpjumper / pod v1 | cross setup | 6.16 mm | +0.30 [+0.04, +1.01] mm | +0.013 | 86% | 13.08 mm |
| Stumpjumper / pod v2 → Stumpjumper / pod v2 | same setup | 4.80 mm | +0.14 [-0.42, +0.65] mm | +0.006 | 54% | 7.78 mm |
| Stumpjumper / pod v2 → TR11 / pod v2 | cross setup | 19.87 mm | +10.17 [+3.04, +13.10] mm | +0.365 | 98% | 40.96 mm |
| TR11 / pod v2 → Stumpjumper / pod v1 | cross setup | 40.18 mm | +35.67 [+33.92, +42.62] mm | +1.567 | 100% | 61.30 mm |
| TR11 / pod v2 → Stumpjumper / pod v2 | cross setup | 41.18 mm | +36.27 [+34.09, +45.58] mm | +1.616 | 100% | 66.84 mm |
| TR11 / pod v2 → TR11 / pod v2 | same setup | 9.06 mm | -0.18 [-1.83, +1.51] mm | -0.007 | 40% | 14.80 mm |

## Main findings

1. **Changing sensor generation on the same bike has a small transfer cost.** Pod-v1 Stumpjumper curves transferred to pod-v2 Stumpjumper with a -0.05 mm median penalty; the reverse direction was +0.30 mm. These are comparable to same-setup log-transfer penalties and tiny relative to cross-bike effects. Supervised oracle penalties are positive in both directions, showing a real but modest hardware/mounting difference that self-supervised fit variance can obscure.
2. **Stumpjumper calibrations fail on the TR11.** Self-supervised transfer penalties are +9.75 mm from pod-v1 Stumpjumper and +10.17 mm from pod-v2 Stumpjumper. Transfer is worse than the TR11 target calibration for 98% of source-target pairs from both Stumpjumper setups.
3. **TR11 calibrations fail even more severely on the Stumpjumper.** The penalties are +35.67 and +36.27 mm, and every evaluated pair is worse than the target's own calibration.
4. **The conclusion survives normalization and oracle substitution.** Cross-bike normalized-RMSE penalties remain large, and both supervised oracle families show the same qualitative separation. The result is therefore not explained merely by different fork travel, target difficulty, or self-supervised optimizer noise.
5. **Absolute anchoring amplifies cross-bike failure.** Self-supervised cross-bike anchored errors are much larger than aligned errors, so transferring a frozen curve cannot be rescued by the present target-side anchor policy.

## Why transfer is directionally asymmetric

The active-data magnetic ranges differ substantially:

| Setup | Median magnetic p5–p95 | Median magnetic span | Median travel span |
|---|---:|---:|---:|
| Stumpjumper / pod v1 | 626–6799 | 6139 | 79.4 mm |
| Stumpjumper / pod v2 | 512–5567 | 5063 | 77.7 mm |
| TR11 / pod v2 | 1015–2127 | 1154 | 89.1 mm |

The TR11 occupies a narrow, low magnetic interval compared with either Stumpjumper setup. A TR11-trained power curve must extrapolate far outside its observed magnetic support on Stumpjumper targets, explaining why that direction is especially destructive. Stumpjumper-to-TR11 transfer stays nearer the source's low-magnitude region but still applies the wrong curve shape/scale. This asymmetry is evidence for a setup-specific mapping, not evidence that one transfer direction is acceptable.

## Paper implications

- The experiment directly supports the claim that a one-time calibration does not generalize across bike/sensor geometry, particularly across bikes.
- It strengthens the value proposition for calibration without stored bike-specific priors: each target log's independently learned mapping is dramatically better than importing a curve from the other bike.
- The pod-v1↔pod-v2 result is a useful nuance: the method need not claim that every remount or sensor revision creates a wholly unrelated curve. The dominant tested change is bike/fork/magnet geometry.
- The paper should show directed transfer, because the failure is strongly asymmetric. A pooled 'cross-setup' number would conceal the extrapolation mechanism.
- The primary paper table should include aligned and normalized error; anchored error can be a secondary end-to-end measure.

## Pipeline implications

- Never silently reuse a calibration across an unknown bike/setup. Require setup identity or perform self-calibration on the current recording.
- Store the calibration's observed magnetic support. If a target recording lies outside it, reject the imported curve rather than extrapolating.
- A pod-generation change on the same bike may permit a warm start, but it should still be validated by self-supervised constraints before acceptance.
- A generic population prior could initialize optimization, but the final mapping must adapt to the current bike/setup.
- Add a curve-compatibility score based on magnetic-support overlap and short-window IMU residuals; this experiment provides positive and negative pairs for selecting a threshold.

## Limitations and next step

This experiment evaluates the magnetic mapping before downstream fusion and uses full-log source calibration. It also uses the reference-derived `boring_mask` for evaluation, while oracle curves use target-independent reference data from their source log only. The next highest-value experiment is a smaller downstream-solver transfer study using representative source calibrations from each setup. That will measure how much cross-setup curve failure survives IMU fusion and magnetic-nuisance correction.
