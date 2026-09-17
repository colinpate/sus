# Front mag-calibration window-length analysis

## Bottom line

The rise in same-window absolute RMSE is mostly an evaluation-support and local-centering effect, not evidence that the self-supervised learner simply gets worse with more data. Longer windows contain a wider travel distribution and cannot use a highly local offset correction. On an identical central evaluation core, self-supervised curves generally improve from the shortest windows and then plateau. Full-log accuracy also improves strongly before reaching a setup-dependent optimum.

There is nevertheless evidence of a real long-window compromise: the low-travel and high-travel regions move in opposite directions, and even supervised one-dimensional oracles accumulate more normalized residual as a window spans more behavior. This is consistent with hysteresis, time variation, or another missing state variable making travel not perfectly single-valued in magnetic magnitude.

## 1. Travel support explains much of the apparent rise

| Front v1 self-supervised | 5 s | 120 s |
|---|---:|---:|
| Own-window aligned RMSE | 3.41 mm | 4.07 mm |
| Travel standard deviation | 18.93 mm | 21.52 mm |
| Travel range | 97.2 mm | 142.1 mm |
| RMSE / travel standard deviation | 0.181 | 0.194 |

Across 219 paired nested windows, raw RMSE increased in 72% of 5-to-120-second comparisons (median change +0.68 mm). After normalization by each window's travel standard deviation, the median paired change was only +0.004, and increases occurred in 53%. The supervised oracles also rise on their own larger training windows, confirming that this pattern is not unique to self-supervised optimization.

## 2. Local centering makes short windows look better

At 5 seconds, front-v1 self-supervised RMSE is 3.41 mm with a separate local offset, but 3.89 mm when the full-log offset is held fixed. At 120 seconds the two are 4.07 and 4.08 mm. Thus local centering contributes about half a millimeter of the apparent 5-second advantage and almost none at 120 seconds.

## 3. Holding evaluation data fixed reverses the main interpretation

| Training duration | Own window | Same central 5 s | Full log |
|---:|---:|---:|---:|
| 5 s | 3.41 | 3.41 | 4.74 |
| 10 s | 3.54 | 3.20 | 4.18 |
| 20 s | 3.54 | 3.19 | 3.99 |
| 40 s | 3.68 | 3.49 | 3.80 |
| 60 s | 3.67 | 3.29 | 3.98 |
| 120 s | 4.07 | 3.38 | 4.03 |

On the identical central 5-second samples, increasing training from 5 to 10–20 seconds improves RMSE rather than worsening it. The own-window curve rises because its evaluation set expands. Full-log accuracy reaches its best median around 40 seconds in this cohort, after which it is approximately flat or slightly worse.

## 4. Absolute anchoring is now a larger error source than curve shape

For front v1, the existing non-reference anchoring policy gives 8.17 mm full-log RMSE at 5 seconds, 7.82 mm at 40 seconds, and 8.11 mm at 120 seconds. The corresponding optimally aligned errors are 4.74, 3.80, and 4.03 mm. Once 10–40 seconds of useful motion are available, improving the absolute reference/anchor is likely more valuable than simply accumulating more curve-training data.

## 5. The long-window compromise is travel-region dependent

Using the same full-log alignment offset for every training window, front-v1 self-supervised fixed-bin RMSE changes as follows:

| Travel bin | 5 s training | 120 s training | Direction |
|---|---:|---:|---|
| 0–30 mm | 5.38 | 8.04 | worse |
| 30–60 mm | 3.00 | 3.43 | slightly worse |
| 60–90 mm | 3.19 | 2.00 | better |
| 90–120 mm | 6.00 | 2.83 | better |

Longer training exposes high-travel motion and improves that end of the curve, while the low-travel end degrades. On the full log, the predicted-to-reference travel standard-deviation ratio moves from 0.905 at 5 seconds to 1.020 at 40 seconds and 1.061 at 120 seconds: short fits tend to under-span travel, whereas the longest fits slightly over-span it. The same low-end tendency appears in the supervised oracles, so it is unlikely to be only a self-supervised optimizer bug. A single magnetic-magnitude curve is being asked to compromise across states that are not perfectly consistent.

## 6. Very short windows are information-limited

In the 1–5-second Stumpjumper sweep, self-supervised fit failures fall from 18.2% at 1 second to 3.6% at 5 seconds. The log-balanced median number of accepted motion chunks grows from 3.0 to 15.5. Full-log RMSE falls from 7.67 to 4.97 mm. Time alone is therefore not the best readiness criterion; accepted chunks and magnetic/travel excitation should gate calibration.

## 7. The all-LSM6DSO32 result is setup-dependent

The completed v2 manifest contains 1, 2, 5, 10, and 40 seconds. The TOML was subsequently extended, so 60–120-second v2 points do not exist in this run and are not analyzed here.

| Setup | Logs | Full-log RMSE at 10 s | Full-log RMSE at 40 s |
|---|---:|---:|---:|
| Stumpjumper, pod-v2 | 11 | 4.26 | 3.83 |
| Stumpjumper, pod-v1 | 7 | 5.94 | 6.02 |
| TR11, pod-v2 | 6 | 8.45 | 8.65 |

The pooled v2 uptick at 40 seconds is not universal: the original Stumpjumper/pod-v2 group continues improving, while the older pod-v1 and TR11 groups plateau at substantially higher error. Sensor/geometry strata should therefore be shown separately even though the algorithm receives no setup-specific prior.

## Pipeline implications

1. Do not optimize calibration duration using own-window absolute RMSE alone. Use full-log or held-out-block error, normalized error, fixed-bin curves, and production-anchored error together.
2. Require a minimum information budget rather than only elapsed active time: accepted chunk count, magnetic span, and preferably coverage across relevant travel states.
3. Ten active seconds is a defensible minimum starting point for the current front learner; 20–40 seconds is safer when the objective is whole-recording accuracy. One to two seconds is unreliable.
4. Consider balancing training chunks over magnetic/travel proxy bins. The long-window curve currently trades low-travel accuracy for high-travel accuracy as high-excitation samples enter the fit.
5. Treat within-log recalibration as a remaining hypothesis, not yet a conclusion. A matched-support, time-separated transfer experiment is needed to distinguish true temporal drift from hysteresis and changing travel distributions.

## Method notes

All aggregate curves first take the median across repeats within each log and then the median across logs. Common-core results evaluate every nested calibration on the exact smallest central window and are conditioned on that smallest self-supervised fit succeeding. Fixed-bin diagnostics require at least 20 samples in a bin and report the number of contributing logs in the accompanying CSV. `boring_mask` remains the reference-derived activity mask.
