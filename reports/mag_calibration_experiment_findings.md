# Magnetic self-calibration experiments: findings and next steps

**Status:** 20 September 2026
**Scope:** Magnetometer-to-suspension-travel calibration experiments completed to date, with emphasis on the front pipeline.

## Executive summary

The experiments support the central claim that a magnetic suspension sensor can learn a useful magnetometer-to-travel mapping from ordinary ride data without a manual, bike-specific position calibration. On the best-controlled front cohort—11 Stumpjumper logs using the pod-v2 sensor—approximately 20–40 seconds of active data is enough to reach the main accuracy plateau. When the learned curve is passed through the complete fusion pipeline, median full-log RMSE falls from 4.52 mm with 5 seconds of training to 3.29 mm with 40 seconds and 3.24 mm with 120 seconds. All 11 logs improve in the paired 5-to-120-second comparison.

The results also refine the story. The apparent rise in same-window error with longer training is mostly caused by evaluating longer windows over a wider travel distribution and by allowing short windows their own favorable offset. It is not evidence that more calibration data simply makes the learner worse. A real residual effect remains, however: longer windows improve the high-travel portion of the curve while degrading the raw 0–30 mm region. Supervised oracles show a similar compromise, suggesting that magnetic magnitude is not perfectly single-valued with travel because of hysteresis, time variation, magnetic perturbation, or another missing state variable.

Within a fixed bike/sensor setup, a calibration learned once and transferred to another log performs surprisingly close to a per-log self-supervised calibration. The present results therefore show only a small or absent advantage for per-log fitting over one-time same-setup fitting. Supervised oracles do transfer measurably worse than their per-log versions, indicating that real log-to-log variation exists, but current self-supervised estimation noise partly hides that benefit. Cross-setup transfer is strongly directional and ranges from mild to catastrophic. In the updated complete 38-log pod-v2 study, Stumpjumper-to-TR11 penalties are 11–13 mm and the reverse penalties are 31–47 mm, while the two TR11 installations differ by about 1–3 mm in the supervised oracles. All 12 directed oracle comparisons favor target-specific calibration, but closely related fork/installations can share much more of their mapping than unrelated geometries.

For the pipeline, the evidence favors a 40-active-second default with an information/readiness gate and the ability to continue toward 120 seconds or refit when confidence is low. The largest immediate improvement opportunity is absolute anchoring, followed by calibration-quality detection, balanced travel-state coverage, and modeling the missing state behind the low-/high-travel compromise.

## Evidence base and terminology

The strongest results come from three complementary experiment families:

1. **Random nested-window sweeps** fit self-supervised and supervised-oracle curves on fixed active-data durations and score them on the training window, the full log, and the full log excluding training.
2. **Downstream solver sweeps** inject the saved self-supervised curves, run both fusion solves over the complete log, and score every intermediate and final stage afterward.
3. **Cross-log transfer matrices** fit once on each source log and apply that calibration to every other log from the same setup. Diagonal entries are per-log fits; off-diagonal entries are transferred fits.
4. **Cross-setup transfer** includes a complete 24-log matrix across the original three setups, a reproducibly sampled pod-v2 matrix, and the final complete 38-log pod-v2 matrix across Stumpjumper, two TR11 installations, and Slayer. All use the self-supervised learner and two supervised-oracle trainers.

Unless stated otherwise, errors below are RMSE after one constant alignment. This measures curve shape and dynamic travel reconstruction separately from the absolute zero-travel anchor. “Anchored RMSE” retains the current deployable, non-ground-truth anchor policy. Aggregate results collapse repeats within each log before weighting logs equally.

The current experiments use `boring_mask` to define active samples. Because that mask is derived from reference travel, it is valid for these controlled analyses but not yet a deployable part of the self-calibration method.

## Main findings

### 1. Short ride-data windows produce useful self-calibration

On the 11-log Stumpjumper/pod-v2 cohort, raw self-supervised full-log RMSE improves from 4.75 mm with 5 seconds of active training data to 4.18 mm at 10 seconds, 3.99 mm at 20 seconds, and 3.80 mm at 40 seconds. Performance then plateaus at 3.98 mm at 60 seconds and 4.03 mm at 120 seconds. Scoring the full log with the training samples removed gives nearly identical values, so the result is not explained by evaluating on the same samples used during self-supervised training. [Source S1](../experiments/mag_calibration/runs/random-window-front-v1/report.md)

Very short windows are much less reliable. In the 1–5-second sweep, self-supervised fit failures decrease from 18.2% at 1 second to 3.6% at 5 seconds, while full-log RMSE decreases from 7.67 to 4.97 mm. The median accepted motion-chunk count rises from 3.0 to 15.5 over the same range. This shows that usable motion information—not clock time alone—determines readiness. [Source S2](../experiments/mag_calibration/runs/random-window-front-v1-shorter/report.md), [supporting analysis](../experiments/mag_calibration/analysis/front-window-length/report.md)

**Interpretation:** Ten active seconds is a defensible minimum for the current front learner; 20–40 seconds is a better target for whole-recording accuracy. One or two seconds is not dependable.

### 2. Longer-window same-block error is mostly a measurement artifact

In the original front sweep, own-window aligned RMSE rises from 3.41 mm at 5 seconds to 4.07 mm at 120 seconds. Over the same comparison, however, reference-travel range expands from 97.2 to 142.1 mm and reference standard deviation expands from 18.93 to 21.52 mm. Across 219 paired nested windows, absolute RMSE rises in 72% of comparisons, but normalized RMSE has a median change of only +0.004 and rises in only 53%.

Short windows also benefit more from local recentering. At 5 seconds, RMSE is 3.41 mm with a window-specific offset and 3.89 mm when the full-log offset is held fixed; at 120 seconds the values are 4.07 and 4.08 mm. When every trained curve is evaluated on the exact same central 5-second samples, error improves from 3.41 mm at 5 seconds to 3.20–3.19 mm at 10–20 seconds before plateauing. [Source S3](../experiments/mag_calibration/analysis/front-window-length/report.md)

**Interpretation:** Same-window absolute RMSE should not be used alone to choose calibration duration. Fixed-support, normalized, fixed-offset, held-out, and final-solver metrics are more informative.

### 3. The full fusion pipeline benefits from longer calibration and removes most low-travel degradation

The repeated downstream experiment contains 220 successful conditions: 11 logs, four random centers per log, and 5, 10, 20, 40, and 120 seconds of active calibration data. [Source S4](../experiments/mag_calibration/runs/solver-window-front-phase2/report.md)

| Training duration | Raw mag-model full-log RMSE | Final solved full-log RMSE |
|---:|---:|---:|
| 5 s | 5.10 mm | 4.52 mm |
| 10 s | 4.39 mm | 4.17 mm |
| 20 s | 4.73 mm | 3.68 mm |
| 40 s | 4.39 mm | 3.29 mm |
| 120 s | 4.10 mm | 3.24 mm |

The paired final-solved change from 5 to 120 seconds is −1.31 mm; all 11 logs improve, with an exploratory Wilcoxon p-value of 0.001. The typical paired improvement is only 0.17 mm from 20 to 40 seconds and 0.04 mm from 40 to 120 seconds, placing the central-error elbow around 20–40 active seconds.

Longer windows still improve consistency. Median within-log repeat IQR falls from 0.40 mm at 20 seconds to 0.25 mm at 40 seconds and 0.21 mm at 120 seconds. The 90th-percentile final error across conditions is 5.30, 5.83, and 4.16 mm at 20, 40, and 120 seconds, respectively. Thus 120 seconds primarily reduces the risk of an unlucky calibration window rather than substantially lowering typical error.

The raw 0–30 mm fixed-offset error increases by 2.64 mm from 5 to 120 seconds on the expanding training windows and by 2.39 mm on the identical central core. After the complete solver, the corresponding changes are only +0.43 and +0.61 mm. Fusion therefore attenuates most, but not all, of the raw low-travel compromise. [Source S4](../experiments/mag_calibration/runs/solver-window-front-phase2/report.md)

### 4. A one-dimensional magnetic curve has a real state-dependent error floor

In the controlled Stumpjumper sweep, both self-supervised fits and supervised oracles accumulate more same-window error as the window spans more behavior. The isotonic oracle rises from 1.56 mm at 5 seconds to 2.46 mm at 120 seconds on its own training window; the supervised power oracle rises from 2.03 to 2.60 mm. Because these fits directly observe reference travel, the rise cannot be attributed solely to self-supervised optimization.

The travel-bin pattern is also systematic. With a fixed full-log offset, self-supervised RMSE changes from 5.38 to 8.04 mm in the 0–30 mm bin and from 6.00 to 2.83 mm in the 90–120 mm bin between 5 and 120 seconds. The predicted/reference standard-deviation ratio moves from 0.905 at 5 seconds to 1.020 at 40 seconds and 1.061 at 120 seconds: short fits under-span travel, whereas long fits slightly over-span it. [Source S1](../experiments/mag_calibration/runs/random-window-front-v1/report.md), [Source S3](../experiments/mag_calibration/analysis/front-window-length/report.md)

**Interpretation:** Magnetic magnitude alone is probably not a perfectly single-valued travel coordinate over a whole ride. Loading direction, hysteresis, sensor orientation, temperature, magnetic nuisance, or temporal drift may be acting as a missing state.

### 5. Absolute anchoring is now a larger error source than curve shape

At 40 seconds in the Stumpjumper/pod-v2 sweep, full-log self-supervised error is 3.80 mm with optimal constant alignment but 7.82 mm with the current deployable anchor. At 120 seconds the corresponding values are 4.03 and 8.11 mm. Once enough motion exists to estimate curve shape, zero/reference placement contributes more error than the remaining shape error. [Source S3](../experiments/mag_calibration/analysis/front-window-length/report.md)

**Interpretation:** Improvements to the anchoring policy may yield a larger production gain than extending curve training beyond roughly 40 seconds.

### 6. Per-log calibration has only a mild demonstrated advantage over same-setup calibration

For Stumpjumper/pod-v2 self-supervised curves, transferring a full-log calibration to another same-setup log changes aligned RMSE by a paired median of only +0.093 mm relative to the target log’s own self-supervised calibration; the per-log calibration is better in 53.6% of 110 off-diagonal pairs. For the six TR11/pod-v2 logs, the paired median is −0.108 mm and the per-log calibration is better in only 40% of 30 pairs. These effects are effectively neutral at the present learner accuracy. [Source S5](../experiments/mag_calibration/runs/cross-log-front-stumpjumper-pod-v2-v1/comparison_summary.csv), [Source S6](../experiments/mag_calibration/runs/cross-log-front-jamaal-v1/comparison_summary.csv)

The oracle results reveal a smaller underlying transfer penalty. Stumpjumper oracle transfers are 0.27–0.33 mm worse than their target-log oracle baselines in median paired comparisons, with the per-log oracle better in 95–96% of pairs. TR11 oracle transfers are 0.82–0.91 mm worse, with the per-log oracle better in 97–100% of pairs. [Source S5](../experiments/mag_calibration/runs/cross-log-front-stumpjumper-pod-v2-v1/comparison_summary.csv), [Source S6](../experiments/mag_calibration/runs/cross-log-front-jamaal-v1/comparison_summary.csv)

**Interpretation:** Real log-specific variation exists, but self-supervised estimation variance currently masks much of its benefit. The paper can claim that per-recording calibration removes the need to store setup-specific priors; it should not yet claim a large accuracy advantage over a well-made one-time calibration for an unchanged setup.

### 7. Cross-setup transfer ranges from mild to catastrophic and is strongly directional

Across the all-LSM6DSO32 front analysis, self-supervised full-log RMSE at 10/40 seconds is 4.26/3.83 mm for Stumpjumper/pod-v2, 5.94/6.02 mm for Stumpjumper/pod-v1, and 8.45/8.65 mm for TR11/pod-v2. The same algorithm receives no explicit geometry prior, but its achieved accuracy and optimal duration vary substantially by setup. [Source S3](../experiments/mag_calibration/analysis/front-window-length/report.md), [Source S7](../experiments/mag_calibration/runs/random-window-front-v4/report.md), [Source S8](../experiments/mag_calibration/runs/random-window-front-v4-jamaal/report.md)

The direct 24×24 matrix confirms that these are incompatible physical mappings. Relative to each target log's own self-supervised calibration, Stumpjumper/pod-v1 and Stumpjumper/pod-v2 curves add +9.75 and +10.17 mm aligned RMSE when transferred to the TR11; 98% of those source-target pairs are worse. In the reverse direction, TR11 curves add +35.67 mm on Stumpjumper/pod-v1 and +36.27 mm on Stumpjumper/pod-v2, and every pair is worse. Normalized RMSE and both supervised oracle families reproduce the separation. [Source S9](../experiments/mag_calibration/runs/cross-setup-front-three-setup-v1/cross_setup_report.md)

The strong directional asymmetry is explained in part by signal support. The median active magnetic p5–p95 span is 6,139 units for Stumpjumper/pod-v1, 5,063 for Stumpjumper/pod-v2, and only 1,154 for TR11/pod-v2. A TR11-trained curve must therefore extrapolate far outside its observed domain on Stumpjumper targets. By contrast, changing from pod v1 to pod v2 on the same Stumpjumper changes aligned RMSE by only −0.05 mm in one direction and +0.30 mm in the other. This supports per-setup calibration while showing that the method need not claim every sensor revision creates a wholly unrelated mapping. [Source S9](../experiments/mag_calibration/runs/cross-setup-front-three-setup-v1/cross_setup_report.md)

The final pod-v2-only experiment includes all 38 usable `front-default` logs from Stumpjumper, original TR11/Jamaal, TR11-2025/Harry, and Slayer. The seven Slayer chunks are collapsed to five parent-recording units, giving 36 independent analysis units. All 4,332 evaluations succeeded. Eleven of 12 directed self-supervised cross-setup cells have a positive median penalty, with six crossed-bootstrap intervals entirely above zero. Both supervised oracles have positive penalties with intervals above zero in all 12 directions, separating physical mapping differences from self-supervised estimation noise. [Source S11](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v4/cross_setup_report.md)

The complete matrix reveals useful structure rather than one universal “cross-bike” effect. Transfers involving the Stumpjumper remain severe: Stumpjumper curves add +10.72 and +12.84 mm on the two TR11 targets, while original-TR11, TR11-2025, and Slayer curves add +47.00, +31.34, and +16.41 mm on Stumpjumper targets. In contrast, the two TR11 installations are relatively compatible: their supervised-oracle penalties are +1.40 to +3.34 mm. The self-supervised Harry→Jamaal transfer is −1.50 mm relative to the target's own noisy fit, but both oracles show a positive +1.40 to +1.64 mm physical transfer penalty. [Source S11](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v4/cross_setup_report.md)

Rerunning the identical cohort after the 200 Hz/front-pipeline merge and production bad-mask change preserves the headline result: the counts remain 11/12 positive self-supervised directions and 6/12 intervals above zero. Across the 12 directed cells, the median absolute change in self-supervised penalty is 0.69 mm; the largest change is the Slayer→Stumpjumper direction, which falls by 6.26 mm but remains strongly positive at +16.41 mm. The binned-median oracle is especially stable, with a median absolute cell change of 0.27 mm. [Source S11](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v4/cross_setup_report.md), [Source S12](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v3/cross_setup_report.md)

The five-per-setup sample was an effective pilot under the earlier pipeline revision. Relative to the corresponding complete v3 matrix, the median absolute change across the 12 self-supervised directed penalties is 0.35 mm; ten directions change by less than 1.5 mm and every qualitative conclusion is retained. This numerical sampling comparison should remain paired with v3 rather than being mixed directly with the updated v4 pipeline. [Source S10](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-sample-v2/cross_setup_report.md), [Source S12](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v3/cross_setup_report.md)

**Interpretation:** A calibration cannot be assumed portable merely because the sensor generation matches, but setup mismatch is continuous rather than binary. Fork family, magnet placement, and magnetic support appear to determine compatibility. Per-recording calibration remains the reliable default; a stored prior may still be useful as a validated warm start for closely related installations.

### 8. Front evidence is mature; rear evidence is not

No completed rear experiment is included in the retained evidence set. An early 28-condition smoke run was too incomplete to support analysis and has been removed rather than risk accidental citation. Rear claims should wait for a newly scoped, completed study.

## How these findings help the paper

### Claims supported now

- **Self-calibration from ride data is feasible.** Useful front curves emerge from tens of seconds of active data without a manual position sweep or stored bike-geometry mapping.
- **The method generalizes operationally across recordings.** It can independently fit every tested front recording using the same algorithm and no setup-specific coefficients.
- **IMU/magnetometer fusion adds measurable value.** The final solver is more accurate and substantially less sensitive to the raw low-travel compromise than the magnetic mapping alone.
- **A short calibration is sufficient for practical use.** The main final-error plateau occurs around 20–40 active seconds on the best-controlled setup.
- **The experiments expose a meaningful physical/modeling limitation.** Even supervised one-dimensional oracles show state-dependent residual, supporting the hypothesis that the magnetic signal contains hysteresis or another missing state rather than only optimizer error.
- **Target-specific validation is necessary across the tested setups.** Frozen curves can fail catastrophically across dissimilar geometries, while independent per-recording calibration avoids that risk without a stored geometry prior.

### Claims that should be qualified

- Use **“self-calibrating from a recording”** or **“ride-data calibration”** rather than implying that the current experiments continuously adapt causally in real time.
- Do not claim that per-log fitting is dramatically more accurate than a one-time same-setup calibration. The measured self-supervised advantage is small or absent.
- Do not yet claim a production-ready fully automatic method while active-window selection depends on `boring_mask`.
- Describe cross-setup evidence as a magnetic-mapping result until the transferred and independently calibrated curves have also been compared after downstream fusion.
- Do not generalize the mature front findings to the rear pipeline until the rear sweep is completed.

### Recommended paper figures/tables

1. **Final solved RMSE versus active calibration duration**, with log-level uncertainty and the raw mag-model curve shown for comparison.
2. **Fixed-support/common-core versus own-window error**, demonstrating why the naive same-window trend is misleading.
3. **Travel-bin error before and after fusion**, highlighting both the low-/high-travel compromise and its downstream attenuation.
4. **Directed cross-setup transfer heatmaps**, showing absolute error and penalty versus target per-log calibration for all three trainer families.
5. **Same-setup cross-log transfer heatmaps**, with diagonal and paired off-diagonal summaries for self-supervised and oracle fits.
6. **Setup-stratified results**, not a single pooled LSM6DSO32 curve.
7. **Oracle error decomposition**, separating curve-family limits, self-supervised estimation error, absolute anchoring, and downstream fusion gain.

## How these findings help improve the pipeline

### Recommended operating policy

- Begin fitting after **at least 10 seconds of valid active motion**.
- Treat **20–40 active seconds as the normal accuracy target**.
- Use **approximately 40 seconds as the default** when latency matters.
- Continue toward **120 seconds, combine multiple windows, or refit** when the curve-quality signal is poor or when robustness is more important than calibration latency.

### Highest-value engineering changes

1. **Replace the time-only trigger with an information gate.** Require enough accepted IMU displacement chunks, magnetic span, and coverage across magnetic/travel-proxy states. One-second windows fail because they lack usable constraints, not merely because they are short.
2. **Add a ground-truth-free calibration confidence score.** Use chunk consistency, held-out self-supervised residuals, coefficient plausibility, predicted travel span, and agreement among several subwindow fits. Low confidence should cause the learner to accumulate more data or retry on another block.
3. **Improve absolute anchoring.** The roughly 4 mm gap between aligned and currently anchored error is now one of the clearest bottlenecks. Candidate anchors should be tested without reference travel and evaluated across remounts and setups.
4. **Balance training coverage.** Prevent high-excitation chunks from progressively dominating long-window fits. Magnetic-bin balancing, capped per-bin weights, or a coverage-aware sampler may preserve the low-travel region while retaining high-travel gains.
5. **Model the missing state.** Test separate compression/rebound curves or include direction, velocity, orientation, temperature, and magnetic-nuisance state. The goal is to reduce the oracle residual as well as the self-supervised residual.
6. **Use repeated-window agreement.** Multiple 20–40-second fits could be compared and combined using a medoid or robust ensemble. This may obtain much of the 120-second robustness without requiring a single long contiguous calibration block.
7. **Make recalibration confidence-driven.** Refit when curve parameters or subwindow predictions drift beyond a threshold, rather than on a fixed timer.

## Next experiments

### Priority experiments for the paper

1. **Deployable activity-mask replacement.** Build an activity mask from `accel_proj`, IMU energy, accepted motion chunks, or a related non-reference signal. Run the same deterministic windows with `boring_mask` and the deployable mask, reporting selection overlap, fit failure, and final-solved error. This is the most important step for making “automatic self-calibration” a defensible end-to-end claim.
2. **Multi-setup final-solver validation.** Repeat the Phase 2 downstream experiment on the original TR11, TR11-2025, and Slayer, preferably at 10, 40, and the longest feasible duration. This will determine whether the 20–40-second result is setup-specific and will connect self-calibration directly to final telemetry accuracy across geometries.
3. **Complete rear validation.** Complete or redesign the incomplete rear sweep, then run a smaller downstream study before scaling. Until this is complete, keep the main paper claim explicitly front-focused or describe rear support as future work.
4. **Cross-setup downstream transfer.** Select representative source curves from each setup, apply them to independently held target logs, and run the complete fusion pipeline. Compare frozen cross-setup curves with each target's independently learned curve at every solver stage. This will show how much of the magnetic-mapping mismatch survives fusion.
5. **Controlled mounting/perturbation study.** Deliberately remount the sensor/magnet, vary spacing or angular alignment, and introduce repeatable external magnetic perturbations. Compare a frozen calibration with per-recording self-calibration and report recovery in both curve and final-solved error. This is needed to support the mounting-variation and perturbation-rejection language in the central claim.
6. **Time-separated within-log transfer.** Train on early, middle, and late fixed-length blocks; evaluate on their own block and the other blocks while matching magnetic/travel support. This distinguishes true temporal drift from changing travel distribution and tests whether recalibration during a ride is valuable.
7. **Paper-grade uncertainty.** Predefine primary metrics and use a hierarchical bootstrap or mixed-effects model with log as the independent unit and windows nested within log. Reserve some logs or rides as a final untouched confirmation set.

### Priority experiments for pipeline improvement

1. **Calibration-quality predictor study.** Use the completed window sweeps to predict held-out or final-solved error from non-reference features available at calibration time: accepted chunk count, chunk-direction balance, magnetic span, coefficient stability, subwindow disagreement, and self-supervised residual distribution. Select thresholds using leave-one-log-out validation.
2. **Coverage-balanced learner ablation.** Compare the current objective with magnetic-bin-balanced sampling/weights. Primary outputs should be full-log final RMSE, 0–30 and 90–120 mm error, and the number of regressions relative to baseline.
3. **Multiwindow robust fitting.** Compare one contiguous 40-second fit, two or four shorter distributed fits totaling 40 seconds, and a 120-second fit. Test coefficient median/medoid selection and joint robust fitting.
4. **Anchor ablation.** Evaluate candidate non-reference anchors independently of curve learning. Report shape-aligned error, anchor bias, anchored RMSE, and failure rate across setups and remounts.
5. **Direction/state-conditioned mapping.** First use the oracle to measure the ceiling from compression/rebound-specific curves; only then implement the best state variable in the self-supervised learner. This prevents spending engineering effort on a state split that does not reduce the supervised floor.
6. **Confidence-driven recalibration simulation.** Replay logs causally, accumulate valid chunks, emit a calibration once confidence is sufficient, and trigger refits only when subwindow agreement deteriorates. Score accuracy, time-to-first-calibration, number of refits, and compute cost.

## Suggested near-term sequence

The most efficient sequence is:

1. Replace `boring_mask` and validate the automatic data-selection path.
2. Build a GT-free quality gate from the already-completed experiments.
3. Run the smaller TR11 and rear downstream studies at a few selected durations.
4. Run cross-setup downstream fusion and controlled-remount comparisons.
5. Investigate coverage balancing and state-conditioned curves using oracle-first ablations.
6. Finish with a held-out paper confirmation set and hierarchical uncertainty estimates.

This order strengthens the paper’s core claim first, while using each experiment to inform the next pipeline change.

## Source index

- **S1 — Front random-window v1:** 11 Stumpjumper/pod-v2 logs, 20 repeats, six durations, self-supervised plus two oracles; 3,960 scheduled fits, two failures. [Report](../experiments/mag_calibration/runs/random-window-front-v1/report.md)
- **S2 — Very-short-window sweep:** 11 Stumpjumper/pod-v2 logs, ten repeats, 1–5 seconds; 1,100 scheduled fits, 46 failures. [Report](../experiments/mag_calibration/runs/random-window-front-v1-shorter/report.md)
- **S3 — Window-length mechanism analysis:** fixed support, normalization, anchoring, travel bins, and setup strata. [Report](../experiments/mag_calibration/analysis/front-window-length/report.md)
- **S4 — Downstream Phase 2:** 11 Stumpjumper/pod-v2 logs, four random centers, five durations; 220/220 successful full-pipeline conditions. [Report](../experiments/mag_calibration/runs/solver-window-front-phase2/report.md), [aggregate CSV](../experiments/mag_calibration/runs/solver-window-front-phase2/aggregate_summary.csv)
- **S5 — Stumpjumper same-setup transfer:** 363/363 source-target-trainer evaluations. [Report](../experiments/mag_calibration/runs/cross-log-front-stumpjumper-pod-v2-v1/report.md), [paired comparison](../experiments/mag_calibration/runs/cross-log-front-stumpjumper-pod-v2-v1/comparison_summary.csv)
- **S6 — TR11 same-setup transfer:** 108/108 source-target-trainer evaluations. [Report](../experiments/mag_calibration/runs/cross-log-front-jamaal-v1/report.md), [paired comparison](../experiments/mag_calibration/runs/cross-log-front-jamaal-v1/comparison_summary.csv)
- **S7 — All-LSM6DSO32 front sweep:** 24 logs across multiple setups, 10–60 seconds; 3,840/3,840 successful fits. [Report](../experiments/mag_calibration/runs/random-window-front-v4/report.md)
- **S8 — TR11/Jamaal window sweep:** six logs, 10–80 seconds; 1,200/1,200 successful fits. [Report](../experiments/mag_calibration/runs/random-window-front-v4-jamaal/report.md)
- **S9 — Three-setup transfer matrix:** 24 front logs across Stumpjumper/pod-v1, Stumpjumper/pod-v2, and TR11/pod-v2; 1,728/1,728 successful source-target-trainer evaluations. [Report](../experiments/mag_calibration/runs/cross-setup-front-three-setup-v1/cross_setup_report.md), [setup summary](../experiments/mag_calibration/runs/cross-setup-front-three-setup-v1/setup_transfer_summary.csv)
- **S10 — Expanded pod-v2 transfer sample:** five independent recordings from each of Stumpjumper, original TR11/Jamaal, TR11-2025/Harry, and Slayer; 1,200/1,200 successful source-target-trainer evaluations. [Report](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-sample-v2/cross_setup_report.md), [setup summary](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-sample-v2/setup_transfer_summary.csv), [frozen sample](../experiments/mag_calibration/specs/cross_setup_front_pod_v2_sample_v2.toml)
- **S11 — Updated complete pod-v2 transfer matrix:** all 38 usable pod-v2 `front-default` logs across Stumpjumper, original TR11/Jamaal, TR11-2025/Harry, and Slayer; 36 independent parent recordings and 4,332/4,332 successful evaluations after the 200 Hz/front-pipeline merge and bad-mask update. [Report](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v4/cross_setup_report.md), [setup summary](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v4/setup_transfer_summary.csv), [frozen specification](../experiments/mag_calibration/specs/cross_setup_front_pod_v2_full_v4.toml)
- **S12 — Pre-merge complete pod-v2 matrix:** the identical 38-log cohort under the prior pipeline revision, retained for sensitivity comparisons. [Report](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v3/cross_setup_report.md), [setup summary](../experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v3/setup_transfer_summary.csv)

## Limitations applying to the current evidence

- `boring_mask` uses reference travel and must be replaced for a deployable end-to-end claim.
- Most mature downstream results come from one bike/sensor cohort.
- Cross-setup transfer has been measured at the magnetic-mapping stage, not yet after the complete fusion pipeline.
- The current statistics are exploratory; several analyses and duration choices were informed by observed data.
- Parent recordings are the appropriate independent unit. Multiple windows or derived chunks from one parent improve precision but do not create independent rides; the expanded Slayer analysis uses one chunk per parent and records the parent IDs explicitly.
- Rear-pipeline evidence is incomplete.
