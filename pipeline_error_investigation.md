# Pipeline Error Investigation Notes

Last updated: 2026-04-11

This note summarizes the investigation into the worst logs from:

```bash
python3 stats_aggregator.py --center-errors --sort-key rmse
```

Primary logs investigated:

- `log056_ccdh`
- `log085`
- `log080`
- `log091`
- `log079`

Main commands used during this work:

```bash
python3 stats_aggregator.py --center-errors --sort-key rmse
python3 stats_aggregator.py --center-errors --deep-dive --sort-key solved_rmse log056_ccdh log085 log080 log091 log079
```

## Important context

- `boring_mask` is effectively the active-motion mask, not the quiet/boring mask.
- So the reported RMSEs are for the more interesting moving parts of the logs.
- The most important code paths for this investigation were:
  - `/Users/colin/Documents/projects/sus/backend/fusion.py`
  - `/Users/colin/Documents/projects/sus/backend/travel_solver.py`
  - `/Users/colin/Documents/projects/sus/stats_aggregator.py`

## High-level findings

### 1. Low-mag / low-travel behavior is the main source of error

Across the bad logs, the biggest errors usually showed up when:

- travel was low
- projected mag was on the low end
- the mag model had weak sensitivity in that region

This was more important than acceleration. Pooled correlations from the deep-dive stats showed:

- error vs `travel`: moderate negative correlation
- error vs `mag`: weaker negative correlation
- error vs `|accel_hp|`: near zero

Interpretation:

- lower travel and lower mag tend to be harder
- high acceleration by itself was not the main global driver

### 2. The old hard low-mag clamp in `pred_x` was a major problem

Original behavior in `GetMagToTravelModel.pred_x`:

- values below `x0` were clamped to `x0`
- this created a dead zone on the low-mag side
- many bad-log samples lived in or near that dead zone

This was especially harmful when the model needed to respond to weaker mag values instead of flattening them.

### 3. `log091` had a real undertraining problem caused by `min_mag`

For `log091`, the raw `mag_baseline` used as `min_mag` was too strict for training.

What that looked like:

- raw baseline was near zero
- the chunk gate used `min(mag_chunk) < min_mag`
- almost all candidate chunks dipped below that threshold somewhere
- only about 30 chunks survived, which was too few for a stable fit

This was not a global pattern across all bad logs. It was mainly a severe issue on `log091`.

### 4. Chunk weighting was not the fix

We tested the idea that the fit might be over-focused on low-mag chunks because there are many of them.

Result:

- enabling balancing weights in the fit usually made things worse
- stronger inverse-density weighting was much worse

Conclusion:

- the main problem was not that low-mag chunks were overrepresented
- the main problem was usually bad low-mag model behavior, or in `log091`, too few usable chunks

### 5. The solver is still a separate source of error on some logs

Even when the mag model improved, the solver sometimes made things worse:

- `log079` is the clearest example
- `log080` also shows solver regression relative to the mag model

Interpretation:

- the mag-model training path and the solver are separate problems
- fixing the mag model does not automatically fix solver behavior

## What we tried

### A. Added deep-dive analysis tooling

We extended `stats_aggregator.py` with a `--deep-dive` mode that reports:

- stage RMSE for `travel/mag_model`, `travel/mag_model/adj`, and `travel/solved`
- conditioned RMSE by low/high travel, low/high mag, bad-mag, and ZV
- solver delta vs mag model
- pooled feature correlations

This was useful and should be kept.

### B. Replaced the hard low-mag clamp in `pred_x`

This change was kept.

Current idea:

- use a smoothed signed-power response instead of a hard one-sided clamp
- keep sensitivity below `x0`
- avoid a dead zone at low mag

This was a meaningful improvement and is one of the main successful changes from this investigation.

### C. Tried making the solver threshold symmetric

We discussed it, but did not keep that direction.

Reason:

- the physical intuition is that the noisy region is on the low end of projected mag
- symmetric gating would likely trust the wrong side more than intended

If solver gating is revisited later, the better direction is probably:

- keep it one-sided
- make it softer, not symmetric

### D. Tried removing or broadly relaxing low-mag filtering for training

This did not work well.

Examples:

- letting in too much low-mag data often destabilized the fit
- looser chunk metrics like `median(mag_chunk)` instead of `min(mag_chunk)` caused major regressions

Takeaway:

- broad relaxation is too blunt
- the filter should stay mostly strict

### E. Tried changing the chunk metric from `min` to `median`

This was a bad regression and was reverted.

Why it failed:

- it admitted many more chunks
- many of those chunks were low quality
- several logs got much worse

Conclusion:

- changing the chunk metric alone is not the right fix

### F. Tested different `min_mag` definitions using still-mag stats

We compared:

- raw baseline: `median(still) + std(still)` via `GetMagBaseline`
- `still_median`
- `still_median + 0.5 * still_std`
- other scales around that

What happened:

- globally replacing `min_mag` with a softer still-stat rule helped some logs
- but it hurt others, especially when the original baseline was already working

Best interpretation:

- a softer `min_mag` is useful as a fallback, not as a universal replacement

### G. Added an adaptive `min_mag` fallback

This change was kept.

Current behavior in `GetMagToTravelModel`:

- start with the original raw `mag_baseline`
- count how many chunks survive
- if too few chunks survive, relax `min_mag` to:

```text
still_median + 0.5 * still_std
```

- only do this when the relaxed threshold is actually lower than the raw baseline

Current trigger:

- relax only if the original threshold yields fewer than `50` chunks

This fixed the real failure mode on `log091` without loosening the gate for the other four logs.

## What worked

### Kept changes

1. `pred_x` softening in `/Users/colin/Documents/projects/sus/backend/fusion.py`
2. Deep-dive reporting in `/Users/colin/Documents/projects/sus/stats_aggregator.py`
3. Adaptive low-chunk fallback for `min_mag` in `/Users/colin/Documents/projects/sus/backend/fusion.py`

### Net result on the five worst logs

Centered `travel/solved` RMSE after the adaptive fallback change:

| log | solved RMSE |
| --- | ---: |
| `log056_ccdh` | 5.61 |
| `log085` | 5.47 |
| `log080` | 5.86 |
| `log091` | 5.39 |
| `log079` | 6.76 |

Most important improvement:

- `log091` improved from about `8.34` to `5.39`

No regressions were introduced on the other four logs relative to the earlier safe patch state.

## What did not work

- symmetric solver thresholding

## Absolute Reference Investigation

### Problem statement

The mag-to-travel model is mostly relative. The absolute offset comes from the reference-point path in `/Users/colin/Documents/projects/sus/backend/fusion.py`, and because `bx` is disabled in the solver, a bad absolute offset propagates directly into:

- negative mag-model predictions
- extra lower-bound OOB pressure in the solver
- large raw mean error even when centered RMSE looks acceptable

### Current absolute-ref methods

Two main methods were investigated:

1. Calibration-reference method in `GetMagTravelRefPoint`
2. Percentile-based zero-mag heuristic from `/Users/colin/Documents/projects/sus/find_mag_trav_ref.py`

Important detail:

- the calibration path does **not** estimate zero-travel mag directly
- it estimates a higher-mag reference point from detected bump chunks, then shifts the mag model to match that point

This works when the calibration chunks are clean, but it fails when:

- there are no real calibration pulses
- false-positive chunks dominate
- the chosen reference point implies too much negative predicted travel

### Safe patch that was kept

Current code in `/Users/colin/Documents/projects/sus/backend/fusion.py` now does a guarded fallback:

- first use the calibration-based offset
- build a production-safe motion mask from `|accel/lpfhp/proj|` and exclude `mag/proj/bad_mask`
- then measure how many motion-mask samples are predicted below `0 mm`
- if that negative fraction is too high, fall back to a percentile-based zero-mag offset

Current fallback:

- trigger when negative motion-mask fraction exceeds `10%`
- use global mag `p8` as the zero-travel mag candidate

Why this version was kept:

- it does not rely on `boring_mask`, which uses GT travel and is not available in production
- it avoids false fallback triggers from long low-motion sections
- it still catches the genuinely bad-offset logs where negative travel shows up during real motion

### Real pipeline validation of the safe patch

Using fresh pipeline runs from the repo root with plots disabled via a local runner monkeypatch:

```bash
venv/bin/python3 backend/pipeline.py log080
```

Equivalent no-plot reruns were used for the five highest solved-RMSE logs from the previous iteration, plus `log080` because it was essentially tied with `log100`:

- `log096`
- `log091`
- `log085`
- `log079`
- `log100`
- `log080`

Observed centered `travel/solved` RMSE relative to the previous all-samples fallback behavior:

| log | previous solved RMSE | current solved RMSE | change |
| --- | ---: |
| `log096` | 6.31 | 6.04 | -0.27 |
| `log091` | 5.49 | 5.49 | 0.00 |
| `log085` | 5.47 | 5.47 | 0.00 |
| `log079` | 5.40 | 5.40 | 0.00 |
| `log100` | 5.19 | 4.91 | -0.28 |
| `log080` | 5.18 | 5.18 | 0.00 |

Fallback behavior on the reruns:

- `log096`: `3.2%` negative on the motion mask, so fallback stayed off
- `log091`: `0.2%`, fallback stayed off
- `log085`: `0.5%`, fallback stayed off
- `log079`: `10.3%`, fallback turned on
- `log100`: `6.2%`, fallback stayed off
- `log080`: `25.2%`, fallback turned on

Main result:

- the new trigger preserves the helpful fallback behavior on `log079` and `log080`
- it stops the harmful fallback override on `log096` and `log100`
- after this patch, `log100` drops out of the top-five solved-RMSE list

### Current top-error logs after the production-safe fallback patch

Current centered `travel/solved` RMSE ranking:

| log | solved RMSE | current read |
| --- | ---: | --- |
| `log096` | 6.04 | still mainly a low-travel / low-mag mag-model problem; solver helps a lot (`9.02 -> 6.04`) |
| `log091` | 5.49 | mostly unchanged by the ref update; still constrained by weak low-end mag-model behavior |
| `log085` | 5.47 | mostly unchanged; low-travel / low-mag remains the main error pocket |
| `log079` | 5.40 | fallback still needed; remaining issue is mostly solver behavior rather than absolute ref |
| `log080` | 5.18 | fallback still needed; remaining issue is still shared between low-end mag model and solver |

Deep-dive summary on those five logs:

- `log096` remains the worst solved log, but it is no longer being made worse by the absolute-ref fallback. The solver still provides the biggest gain of the group.
- `log091` and `log085` did not need the fallback in the first place, and the motion-mask trigger correctly leaves them alone.
- `log079` and `log080` still benefit from the fallback trigger, so the new production-safe mask did not throw away the earlier progress on those logs.

### Main new finding: there are two different absolute-ref failure modes

The fallback sweep showed that the bad logs are not all the same.

#### Type 1: wide / noisy calibration chunks

Example:

- `log056_ccdh`

Characteristics:

- calibration chunk still-mags are widely spread
- this suggests many false positives or mixed operating points
- aggressive low-percentile fallback overshoots positive

What worked best:

- a milder active-motion percentile fallback, around `p6` to `p8`

#### Type 2: tight but mis-anchored calibration chunks

Examples:

- `log079`
- `log080`

Characteristics:

- calibration chunk still-mags are tight and internally consistent
- but the calibration-based offset still leaves far too many predictions below zero
- this suggests the calibration detector found a coherent but wrong operating point

What worked best:

- stronger low-tail fallback
- especially on `log079`, using a lower percentile than `p8` helped a lot

### Useful discriminator: spread of calibration still-mags

For each detected calibration chunk, look at the mean projected mag in the initial still window.

Observed spread of these still-mags:

- `log056_ccdh`: std about `286 mG`
- `log079`: std about `31 mG`
- `log080`: std about `96 mG`
- `log085`: std about `178 mG`
- `log091`: std about `361 mG`

Interpretation:

- low spread means the detector found a consistent cluster
- high spread means the detector is mixing very different operating points

This was one of the most useful signals for deciding whether a stronger percentile fallback was safe.

### Stronger heuristic that looked promising but was not merged

Tested heuristic:

- if calibration-adjusted active-motion predictions have too many negatives:
  - if calibration still-mag spread is high, use active `p6`
  - if spread is low and the calibration offset is negative, use global `p4`
  - if spread is low and calibration offset is positive, use active `p4`

This improved raw mag-model error across the default logs more than the current safe patch, especially:

- `log079`
- `log080`
- `log056_ccdh`

But it also introduced a tradeoff:

- raw absolute error improved
- centered solver RMSE sometimes got a bit worse than the safer `active p8` fallback

Examples from solver replay:

- `log079`
  - safe patch: raw solved RMSE about `12.79`, centered `5.21`
  - stronger fallback: raw solved RMSE about `7.04`, centered `5.47`
- `log080`
  - safe patch: raw solved RMSE about `5.28`, centered `4.97`
  - stronger fallback: raw solved RMSE about `5.10`, centered `5.05`
- `log056_ccdh`
  - safe patch: raw solved RMSE about `5.13`, centered `4.86`
  - stronger fallback: raw solved RMSE about `5.16`, centered `5.11`

Conclusion:

- the stronger heuristic is promising if absolute bias matters most
- the current safe patch is better if centered solver RMSE is the primary objective

### Current recommendation

Keep the current guarded fallback in code for now because it is a strong safe improvement.

If we revisit this again, the next best direction is not “pick one global percentile.” It is:

1. classify the calibration quality first
2. choose the fallback aggressiveness based on that quality
3. likely revisit solver `bx` or lower-OOB handling if absolute bias remains more important than centered RMSE

Short version:

- `p8` is a good safe fallback
- it is not the universal best fallback
- bad absolute-ref logs split into multiple subtypes, and they want different fixes
- removing low-mag filtering broadly
- replacing `min(mag_chunk)` with `median(mag_chunk)`
- turning on chunk-balancing weights in the fit
- using very aggressive inverse-density weighting
- globally softening `min_mag` for all logs

## Per-log summary

### `log056_ccdh`

- Main issue: low-mag model sensitivity
- The soft `pred_x` helped
- Adaptive `min_mag` fallback was not needed
- Solver still improves this log overall

### `log085`

- Raw baseline path was already okay
- Softer global `min_mag` rules tended to hurt
- Adaptive fallback correctly leaves this log alone

### `log080`

- Mag model can improve with some milder `min_mag` relaxation in isolated tests
- But the solver is still a problem and can give back the gain
- This is a likely next target, but not solved yet

### `log091`

- Strongest example of undertraining from an overly strict raw `min_mag`
- Original threshold left only about 30 training chunks
- Adaptive fallback raised that to about 59 chunks
- This was the biggest clear success from the baseline work

### `log079`

- Solver is the standout problem
- Mag model is decent
- Solver tends to worsen the result
- This should probably be investigated separately from mag-model training

## Current best understanding

The errors are not coming from one single issue.

There are at least two distinct failure modes:

1. Mag-model low-end sensitivity problems
   - fixed partly by the softer `pred_x`

2. Undertrained mag model on some logs because `min_mag` is too strict
   - fixed partly by the adaptive low-chunk fallback

And there is still a separate solver issue on some logs, especially:

- `log079`
- `log080`

## Recommended next steps

1. Leave the current adaptive `min_mag` fallback in place.
2. Keep the soft `pred_x` in place.
3. Treat solver issues as a separate follow-up investigation.
4. If revisiting solver gating, prefer a softer one-sided confidence approach instead of a symmetric hard threshold.
5. If exploring more `min_mag` logic later, keep it adaptive and chunk-count-aware rather than globally looser.

## Quick recap for future conversations

If we come back to this later, the short version is:

- the hard low-mag clamp in `pred_x` was real and worth fixing
- `log091` was failing because the raw `min_mag` left too few training chunks
- broad relaxation of chunk filtering was a dead end
- weighting the fit was a dead end
- adaptive `min_mag` fallback worked
- the all-samples absolute-ref fallback was too aggressive once `boring_mask` was removed
- a production-safe motion-mask trigger fixed `log096` and `log100` without losing the gains on `log079` and `log080`
- solver errors remain, especially on `log079` and probably `log080`
