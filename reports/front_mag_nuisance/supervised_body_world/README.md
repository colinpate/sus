# Continuous Magnetometer-Correction Experiment

> **Follow-up:** Later generalization diagnostics found that projection-based
> initialization and the straight XYZ path confounded magnet-model error with
> nuisance field on the Fox logs. The provisional slope gate below is no longer
> the preferred root fix. See
> `../failure_diagnosis/README.md` for the updated conclusion.

## Conclusion

The continuous body/world field model improves the four low-sensitivity v2
logs, but should not be enabled unconditionally. The safest first integration
is:

1. fit the constrained low-field XYZ line for the setup;
2. enable ambient-field correction only for setups whose fitted XYZ slope norm
   is below a validated threshold (15 mG/mm is the exploratory threshold here);
3. estimate the body/world correction across the full recording at 10 Hz;
4. subtract it only when both predicted and measured field magnitude are below
   1500 mG; and
5. use corrected magnitude as the low-field travel signal, while retaining the
   setup's existing signal outside that region.

Do not remove `mag_proj` globally: projection remained the stronger raw signal
for the v1 logs, while magnitude was stronger for the v2 logs.

## Evaluation protocol

- Current pipeline caches were regenerated before comparison.
- The eight logs were sampled synchronously at 10 Hz.
- Alternating 20-second blocks were used for calibration and held-out scoring.
- Low-field XYZ and scalar travel models used encoder travel only in calibration
  blocks.
- The batch smoother could inspect magnetometer and gyro signals across the
  whole log, but never received held-out encoder travel.
- Metrics use active (`boring_mask`) samples with nonnegative encoder travel.
- `log085` could not complete the current pipeline because its adjusted travel
  prediction left the travel solver with an initial guess outside its bounds;
  `log103` was used as the second v1 comparison instead.

This is a stronger test than fitting and scoring the same samples, but the
15 mG/mm setup gate was observed on this log set and still needs validation on
additional setups.

## Problematic v2 results

RMSE in millimeters on held-out samples:

| Log | Region | Raw projection | Raw magnitude | Corrected magnitude, weak-only | Current pipeline solver |
| --- | --- | ---: | ---: | ---: | ---: |
| 0078 | all | 13.17 | 8.36 | 6.82 | 8.06 |
| 0079 | all | 8.52 | 6.74 | 6.21 | 7.96 |
| 0080 | all | 16.14 | 9.24 | 8.32 | 15.01 |
| 0081 | all | 15.44 | 12.63 | 11.02 | 14.71 |
| 0078 | weak | 10.10 | 7.80 | 5.48 | 8.67 |
| 0079 | weak | 9.44 | 7.47 | 6.28 | 9.37 |
| 0080 | weak | 13.58 | 10.07 | 8.94 | 16.08 |
| 0081 | weak | 15.84 | 12.92 | 11.24 | 15.09 |

Median weak-field RMSE across these four logs was 7.61 mm for weak-only
corrected magnitude, versus 8.93 mm for raw magnitude, 11.84 mm for raw
projection, and 12.23 mm for the current pipeline solver.

Applying the correction to all samples was only slightly better on aggregate
than weak-only application, and carries more regression risk in strong field.
The weak-only result is therefore the recommended production starting point.

## Generalization and enable gate

The low-field XYZ slope norm cleanly separated this particular log set:

| Setup group | Logs | XYZ slope norm (mG/mm) | Result of correction |
| --- | --- | ---: | --- |
| New low-sensitivity v2 | 0078–0081 | 7.88–10.26 | Improved every log |
| Older v2 | 0046, 0072 | 27.34–27.44 | Mostly regressed |
| v1 | 103, 110 | 25.54–33.33 | Regressed |

A 15 mG/mm threshold is a plausible starting gate, not yet a production-tuned
constant. It was selected after seeing these setups.

## Model and weight checks

The agreed 10 Hz starting weights were retained:

- magnetometer measurement sigma: 40 mG per axis;
- body-field random walk: 1.0 mG/sqrt(s);
- world-field/model random walk: 1.5 mG/sqrt(s);
- initial body sigma: 300 mG;
- initial world sigma: 500 mG; and
- weak-field threshold: 1500 mG.

On 0078–0081, median weak-field RMSE was 7.61 mm for the full body+world
model. Ablations were worse: 9.93 mm for body-only and 9.23 mm for
gyro-rotated-world-only. Faster and slower random-walk variants were also worse
on this group. This supports retaining both field components, although their
individual values are not uniquely identifiable; the useful estimated quantity
is their sum.

Four outer travel/correction iterations were sufficient for the aggregate
result. Six iterations made little aggregate difference, though 0081 continued
to improve slightly. A convergence tolerance plus a maximum of roughly eight
iterations would be reasonable in a batch integration.

## Reproduction

From the repository root:

```bash
venv/bin/python tools/front/mag_nuisance/experiment_mag_correction.py \
  --output-dir reports/front_mag_nuisance/supervised_body_world
```

Detailed per-log metrics are in `metrics.csv`, aggregate medians in
`aggregate.csv`, model/state diagnostics in `details.json`, and the PNG files
show each log's predictions, held-out error, low-field region, and correction
magnitude.
