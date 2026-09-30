# Why trusting corrected mag worsens uncentered RMSE

## Conclusion

The sweeps are exposing an absolute-position bias, not a failure of the corrected magnetic signal's shape.

The corrected-mag observation is smoother and more accurate after per-log centering, but it remains systematically below ground truth in absolute travel. The current fusion solution sits slightly above that observation. Increasing `mag_off_floor` or lowering `mag_x_thresh` therefore pulls the entire fused trajectory downward: centered RMSE improves, while the larger negative mean error makes uncentered RMSE worse.

The most likely upstream source is `GetMagTravelRefPoint`. It double-integrates acceleration within short bump windows to obtain displacement from the beginning of each window, then uses that relative displacement as an absolute travel reference. The window's unknown starting position is not added back. Recreating that production reference calculation against ground truth found a negative reference error on 37 of 38 logs, with a mean error of about -21.9 mm.

## Evidence

### 1. The corrected signal has better shape but a low absolute level

Macro means over the ordinary evaluation mask:

| cohort | corrected-mag mean error | corrected-mag centered RMSE | current fusion mean error | current fusion centered RMSE |
| --- | ---: | ---: | ---: | ---: |
| `harry` | -14.530 mm | 6.955 mm | -13.434 mm | 7.966 mm |
| `jamaal` | -6.287 mm | 6.537 mm | -5.429 mm | 6.374 mm |
| `stumpjumper-front-pod-v2` | -4.635 mm | 3.612 mm | -3.945 mm | 3.298 mm |
| `stumpjumper-front-pod-v1` | -10.997 mm | 4.074 mm | -10.407 mm | 3.960 mm |

The nuisance correction itself is directionally helpful: compared with the adjusted scalar-mag curve, it reduces the negative mean error by roughly 1 mm and improves centered shape error in every cohort. It just does not establish a new absolute zero; its output is an interpolation from the scalar-mag travel curve toward the inferred corrected path.

### 2. Lowering `mag_x_thresh` directly pulls the solver toward that low observation

Comparing the current per-log threshold with 750 mG across all 38 logs:

- The fused mean error moved downward by 0.594 mm on average and on 37 of 38 logs.
- After the solver's nonnegative clipping, corrected-mag travel sat below the current fused solution on all 38 logs, by 0.733 mm on average.
- The per-log correlation between that corrected-mag/fusion gap and the 750 mG mean-position shift was 0.922.
- The shift was larger on logs where lowering the threshold added more full-weight magnetic samples (correlation -0.723).

By cohort, the mean-error movement was:

| cohort | current mean error | 750 mG mean error | movement |
| --- | ---: | ---: | ---: |
| `harry` | -13.434 mm | -14.367 mm | -0.933 mm |
| `jamaal` | -5.429 mm | -6.144 mm | -0.715 mm |
| `stumpjumper-front-pod-v2` | -3.945 mm | -4.177 mm | -0.231 mm |
| `stumpjumper-front-pod-v1` | -10.407 mm | -10.790 mm | -0.383 mm |

The newly trusted 750 mG-to-baseline band is itself lower than the current solution. Depending on cohort, it covers about 10-27% of evaluated samples; within that band, corrected-mag mean error is approximately 1.5-4.2 mm more negative than the current fusion mean error.

### 3. The solver cannot independently recover a constant position offset

The acceleration/dynamics residual uses differences between adjacent positions, so adding a constant to the entire trajectory leaves it unchanged. ZUPT constrains velocity, and the acceleration-bias state changes trajectory shape, but neither determines the absolute travel zero.

That leaves the magnetic residual and the 0-to-170 mm bounds to set the absolute level. Once the low-field magnetic residual is given more weight, the solver is expected to inherit more of the magnetic observation's offset. This also explains why the result can improve centered RMSE at the same time: for any error vector,

`uncentered_RMSE^2 = centered_RMSE^2 + mean_error^2`.

The shape term gets smaller, but the already-negative mean error grows in magnitude.

## Root-cause path

1. The self-supervised scalar-mag model learns travel changes and curve shape, but its additive constant is not identifiable from those constraints.
2. `GetMagTravelRefPoint` supplies that constant from short double-integrated acceleration bumps.
3. Those integrations begin at zero displacement and do not include the actual travel at the start of the bump, so the resulting "absolute" reference tends to be too low.
4. `GetMagToTravelModel.adjust_with_ref_point` applies that reference as a constant offset to the full scalar-mag curve.
5. The nuisance-corrected travel starts from that adjusted scalar curve and blends toward an inferred path, so it improves local shape while largely inheriting the low absolute level.
6. Raising `mag_off_floor` or lowering `mag_x_thresh` transfers more of that low level into the final fusion solution.

## Oracle-offset confirmation

As a diagnostic only, one constant was added per log so the corrected-mag median error matched ground truth. The waveform, gate, and all other solver inputs were left unchanged. Replaying the current threshold and 750 mG across all 38 logs produced these cohort-balanced macro means:

| threshold | centered RMSE | 0-30 mm centered RMSE | uncentered RMSE | mean error |
| --- | ---: | ---: | ---: | ---: |
| current per-log baseline | 5.342 mm | 8.129 mm | 5.396 mm | +0.185 mm |
| 750 mG | 5.146 mm | 6.963 mm | 5.218 mm | -0.330 mm |

The 5.342-to-5.146 mm centered improvement is the effect of lowering the threshold *within* the oracle-offset replay; it is not the effect of removing the offset. Applying the oracle offset by itself changed the current-threshold centered RMSE only from about 5.399 to 5.342 mm. A perfect constant translation would leave centered error exactly unchanged. This small residual difference arises because the final solver is not translation-invariant at the 0 and 170 mm bounds (and, secondarily, because it is solved to finite numerical tolerances).

After removing only the inherited constant offset, lowering the threshold still improved centered shape, but it now also improved cohort-balanced uncentered RMSE by 0.179 mm instead of worsening it. This reverses the aggregate effect seen with the original corrected-mag level and confirms that the offset is the main cause. Two individual cohorts still had small uncentered regressions, so the absolute reference is not the only source of variation, but it explains the common cross-sweep pattern.

This replay uses ground truth and is not deployable or suitable for hyperparameter selection. Full results are in `oracle_offset_per_log.csv` and `oracle_offset_aggregate.csv`; the reproducible runner is `tools/front/mag_offset_calibration/diagnose_post_mag_correction_offset.py`.

## Recommended next step

Do not choose `mag_off_floor` or `mag_x_thresh` from uncentered RMSE until magnetic absolute reference accuracy is separated from magnetic shape confidence.

A production candidate should estimate the magnetic curve's additive constant independently—for example from confidently detected top-out samples or another repeatable mechanical reference—while leaving the nuisance correction responsible only for shape.
