# Front magnetic offset-calibration exploration

## Summary

The sparse-pulse hypothesis is correct. Logs whose accepted calibration chunks actually start near top-out have far better reference estimates. The newer rider cohorts contain fewer accepted chunks, almost none that begin at zero travel, and substantially worse double-integration error.

Tightening the existing chunk selection does not rescue it: starting magnetic level is not a reliable enough proxy for zero travel, and using fewer chunks amplifies integration failures. A low-percentile anchor is much more stable, but using it alone over-corrects some older pod-v2 logs. The best tuning result is a conservative confidence blend between the current offset and a p2 zero anchor.

No production behavior was changed.

## Experimental split

- Tuning: 24 logs from `jamaal`, `stumpjumper-front-pod-v1`, and `stumpjumper-front-pod-v2`.
- Validation: 14 `harry` logs, treated as the newest-cohort holdout.
- Ground truth was used only to score candidates and describe whether accepted chunks truly began near top-out.
- Every candidate offset itself is deployable from existing magnetometer/accelerometer/model outputs.

This is a clean split within this experiment, although these logs have been examined in earlier project work and therefore are not globally pristine.

## Is the missing-pulse hypothesis correct?

Yes. A chunk was labeled zero-start for diagnosis when its first ground-truth bump sample was at or below 3 mm.

| log group | logs | chunks/log | mean per-log zero-start fraction | median chunk-start travel | raw reference error | relative integration error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| older pod cohorts | 18 | 20.6 | about 38% | 7.1 mm | -12.6 mm | -5.0 mm |
| newer Jamaal + Harry cohorts | 20 | 6.2 | about 3% | 10.4 mm | -30.2 mm | -19.9 mm |
| Harry validation only | 14 | 6.0 | 0% | 11.9 mm | -37.7 mm | -26.1 mm |

Across all logs, chunk count correlated -0.438 with absolute reference error: fewer chunks generally meant a worse reference.

The manual-pulse proxy is even clearer:

| chunk-start category | logs | chunks/log | raw reference error | current corrected-mag uncentered RMSE |
| --- | ---: | ---: | ---: | ---: |
| at least half start within 3 mm | 6 | 23.7 | -5.7 mm | 7.20 mm |
| 10-50% start within 3 mm | 13 | 16.8 | -15.0 mm | 11.17 mm |
| fewer than 10% start within 3 mm | 19 | 6.9 | -31.7 mm | 15.82 mm |

The missing starting position is not the whole error. On newer logs, the relative double integrations are also biased low by about 19.9 mm versus 5.0 mm on older pod logs. Controlled off-bike pulses therefore help twice: they start near a known mechanical zero and produce cleaner, repeatable acceleration integrations.

## Existing-method tweaks

The following variations were tested without ground-truth inputs:

- keep only the 25% or 50% of chunks with the lowest starting magnetic level;
- keep chunks starting within 100 or 250 mG of the lowest accepted start;
- use only forward or only reverse chunks;
- estimate each chunk's starting travel from the learned curve and the existing p8 fallback;
- fall back to a percentile when chunk count is below several thresholds.

None of the stricter reference-only variants was competitive. The original no-fallback reference gave corrected-mag uncentered RMSE of 14.82 mm on tuning and 31.22 mm on Harry. Lowest-start and direction-only variants were generally still worse. Low starting magnetic magnitude did not reliably identify an unloaded/top-out start, and discarding chunks increased sensitivity to bad integrations.

The pipeline's existing p8 fallback is useful but too weak and too conditional. It activated on 8/14 Harry logs, versus 1/6 Jamaal, 2/7 pod-v1, and 4/11 pod-v2 logs. Applying the p8 rule unconditionally improved the pre-fusion Harry metric from 17.41 to 15.34 mm, but remained well behind p2 or blended approaches.

## Percentile and OOB alternatives

The direct percentile candidates set a low percentile of the learned raw travel curve to zero, using samples that passed the existing bad-mag mask. The robust OOB candidate selected the smallest offset that put the p1-p99 prediction range inside 0-170 mm; in practice it behaves similarly to a p1 zero anchor.

Pre-fusion corrected-mag uncentered RMSE:

| method | tuning | Harry validation | behavior |
| --- | ---: | ---: | --- |
| current pipeline | 10.949 mm | 17.410 mm | strongly low-biased |
| always use pipeline p8 fallback | 10.806 mm | 15.336 mm | helpful, but not enough |
| p2 to zero | 10.102 mm | 12.366 mm | good on new logs; over-corrects pod-v2 |
| robust p1-p99 OOB | 10.204 mm | 12.672 mm | slightly more over-correction than p2 |
| fixed 50/50 current-p2 blend | 9.204 mm | 12.596 mm | conservative and stable |
| chunk-confidence blend | **8.987 mm** | **11.872 mm** | best tuning and validation overall |

The selected chunk-confidence rule is:

`offset = w * current_offset + (1 - w) * p2_offset`, where `w = min(chunk_count / 10, 0.5)`.

It never gives the current chunk-derived result more than 50% weight. With one to four chunks its weight is 10-40%; with five or more chunks it becomes a fixed 50/50 blend. This rule was selected using tuning data only.

## Final fusion replay

The current fusion gate was held fixed (`mag_off_floor=0.1`, per-log baseline `mag_x_thresh`). Current rows use the exact cached production result; alternative offsets shift the corrected-mag observation before replaying the solver.

| method | tuning uncentered RMSE | tuning mean error | Harry uncentered RMSE | Harry mean error |
| --- | ---: | ---: | ---: | ---: |
| current | 10.361 mm | -6.594 mm | 17.211 mm | -13.434 mm |
| fixed 50/50 blend | 8.875 mm | -2.019 mm | 12.806 mm | -6.412 mm |
| p2 to zero | 10.037 mm | +2.599 mm | 13.038 mm | +0.639 mm |
| robust OOB | 10.176 mm | +3.365 mm | 13.467 mm | +2.578 mm |
| chunk-count-25 fallback | 9.709 mm | +0.745 mm | 13.038 mm | +0.639 mm |
| chunk-confidence blend | **8.711 mm** | **-1.661 mm** | **12.224 mm** | **-4.782 mm** |

The selected confidence blend improved final uncentered RMSE by 1.650 mm (15.9%) on tuning and 4.987 mm (29.0%) on held-out Harry. It improved 7/7 pod-v1, 8/11 pod-v2, 5/6 Jamaal, and 12/14 Harry logs. Overall centered RMSE was essentially unchanged, as expected for an offset-only intervention.

There is one important low-end tradeoff. On Harry, 0-30 mm uncentered RMSE changed from 11.621 to 11.761 mm for the confidence blend, while the fixed 50/50 blend improved it to 10.915 mm. The confidence blend is best for overall absolute travel; the fixed blend is the safer choice if low-travel absolute error is the primary objective.

All 190 alternative solver runs converged. (`current` rows were read from cache.)

## Independent legacy offset-only validation

Sixteen additional usable legacy front logs were evaluated without running the modern gyro/nuisance/fusion pipeline. Their cached scalar magnetic prediction, historical adjusted prediction, projected acceleration, magnetic baseline, and ground-truth travel are sufficient to reconstruct and score the offset exactly. Fourteen use the older `mag/proj/corr/lpf` coordinate and two use the newer corrected norm coordinate.

These logs look much more like deliberate-pulse recordings:

- 22.7 accepted chunks per log on average;
- 42% mean zero-start fraction;
- 3.45 mm median of the per-log median starting travel;
- -6.07 mm mean raw reference error;
- +0.82 mm mean relative-integration error.

That last value is especially important: unlike the newer on-bike logs, the acceleration integration itself is approximately unbiased. The historical applied offset was reconstructed from `travel/mag_model/adj - travel/mag_model`; its maximum non-constant residual was below `9e-15` mm.

| offset method | legacy uncentered RMSE | legacy 0-30 mm uncentered RMSE | mean error | logs improved vs current |
| --- | ---: | ---: | ---: | ---: |
| historical cached offset | **8.175 mm** | 11.519 mm | -0.715 mm | baseline |
| pipeline p8 to zero | 10.074 mm | 11.234 mm | +3.857 mm | 7/16 |
| p2 to zero | 11.629 mm | 11.774 mm | +7.854 mm | 5/16 |
| robust OOB | 13.603 mm | 13.292 mm | +10.389 mm | 3/16 |
| fixed 50/50 current-p2 blend | 8.630 mm | 10.398 mm | +3.570 mm | 7/16 |
| chunk-confidence blend | 8.620 mm | **10.383 mm** | +3.599 mm | 7/16 |

This reverses the newer-log result: on these pulse-rich legacy logs, the historical chunk-based offset is best overall. The blends improve low-travel RMSE but are slightly worse overall, and pure percentile/OOB rules substantially over-shift the curve.

A post-hoc compatibility rule—keep the historical offset with at least 25 chunks, otherwise use the confidence blend—scored 8.034 mm overall and 10.497 mm at 0-30 mm. It preserved the modern Harry result and most modern gains. However, it improved only 4/16 legacy logs and still had individual regressions as large as 4.32 mm, so chunk count alone is not a safe production pulse detector. This threshold was examined after seeing the legacy results and is not an independent validation result.

## Recommendation

1. Do not spend more time tightening the current chunk filters alone. Without evidence that a chunk began at mechanical top-out, absolute starting travel remains unobservable, and the newer on-bike integrations are themselves less reliable.
2. For the simplest replacement, p2-to-zero is materially better than the current method on new logs, but it is too aggressive as a universal rule.
3. Do not make the blend a universal replacement: the independent legacy check shows that it gives away accuracy on recordings with good calibration pulses.
4. Preserve a way to explicitly mark deliberate off-bike calibration pulses. When that evidence exists, keep the historical chunk offset. When it does not, the confidence or fixed blend is preferable. Chunk count and starting magnetic magnitude alone were not reliable enough to infer this automatically.
5. Before changing production, validate this two-mode policy on another genuinely new front cohort. Harry is held out from the original selection, but the low-end tradeoff and post-hoc legacy safeguard argue against committing from one validation cohort.

## Artifacts

- `chunk_diagnostics.csv`: accepted-chunk counts, true starting travel, reference error, and relative integration error.
- `per_log_method_metrics.csv` / `aggregate_method_metrics.csv`: all offset candidates before final fusion.
- `solver_per_log_metrics.csv` / `solver_aggregate_metrics.csv`: frozen finalist replay through the final solver.
- `selection.json`: tuning-only candidate selection.
- `tools/front/mag_offset_calibration/explore_mag_offset_calibration.py`: reproducible experiment runner.
- `legacy_chunk_diagnostics.csv`, `legacy_per_log_metrics.csv`, and `legacy_aggregate_metrics.csv`: independent legacy offset-only validation.
- `tools/front/mag_offset_calibration/validate_mag_offset_legacy.py`: legacy-cache validation runner.
