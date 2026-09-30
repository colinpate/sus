# Pod-v2 Generalization Check

> **Follow-up:** The apparent older-v2 regression was traced primarily to the
> experiment's projection-based initial travel and straight XYZ model. A
> magnitude-initialized quadratic XYZ variant improved 10 of 11 Fox logs after
> excluding moved-pod log `0062`. See
> `../failure_diagnosis/README.md`. The tables below remain the
> reproducible results for the original line+projection experiment.

## Cohort and protocol

This experiment uses all 12 front pod-v2 logs listed in
`logs/lists/logs.csv`:

`0062`, `0056`, `0063`, `0046`, `0048`, `0049`, `0054`, `0055`, `0058`,
`0071_183`, `0072_184`, and `0073_185`.

All 12 caches were regenerated with the current front pipeline before running
the fixed-weight body/world model. The evaluation protocol is unchanged from
the 0078–0081 experiment: 10 Hz field states, alternating 20-second
calibration and held-out blocks, 5 mm travel bins, four outer iterations, and
correction applied only in the weak-field region.

The raw magnitude/projection controls and corrected scalar mappings use encoder
travel only in calibration blocks. The smoother can inspect the full log's
magnetometer and gyro signals, but receives no held-out encoder travel.

## Correction-only result

The table compares weak-field RMSE against the better of the raw projection
and raw magnitude controls for each log. Positive delta means correction made
the result worse.

| Log | XYZ slope (mG/mm) | Best raw RMSE (mm) | Corrected magnitude RMSE (mm) | Delta (mm) |
| --- | ---: | ---: | ---: | ---: |
| 0062 | 31.54 | 5.07 | 7.42 | +2.35 |
| 0056 | 27.54 | 5.12 | 5.07 | -0.05 |
| 0063 | 27.07 | 3.89 | 3.71 | -0.18 |
| 0046 | 27.44 | 3.00 | 4.76 | +1.76 |
| 0048 | 27.23 | 4.11 | 4.51 | +0.40 |
| 0049 | 31.66 | 4.75 | 3.70 | -1.05 |
| 0054 | 28.81 | 3.59 | 2.85 | -0.74 |
| 0055 | 27.98 | 4.84 | 4.36 | -0.48 |
| 0058 | 28.08 | 4.15 | 4.61 | +0.47 |
| 0071_183 | 30.29 | 2.90 | 3.67 | +0.77 |
| 0072_184 | 27.34 | 4.01 | 4.58 | +0.57 |
| 0073_185 | 26.78 | 2.55 | 4.67 | +2.12 |

Correction improved five of 12 logs relative to each log's best raw signal and
regressed seven. Median weak-field RMSE increased from 4.06 to 4.55 mm. Median
all-field RMSE increased only slightly, from 3.63 to 3.74 mm, because the
weak-only method leaves nearly all strong-field samples unchanged.

Centered RMSE gives the same conclusion: median weak-field centered RMSE rose
from 3.95 to 4.41 mm, so the regression is not merely a removable constant
bias.

## Interpretation

The ambient correction does not generalize as an unconditional improvement to
these older v2 setups. It is useful specifically on the low-sensitivity
0078–0081 setup.

The proposed setup gate is supported by this cohort:

- 0078–0081 slope norms: 7.88–10.26 mG/mm; correction improved all four.
- Listed older-v2 slope norms: 26.78–31.66 mG/mm; correction was mixed and
  slightly negative in aggregate.
- Exploratory enable threshold: 15 mG/mm.

Every log in this generalization cohort is above the threshold, so a production
implementation that gates only the correction would leave all 12 unchanged.
There is a large untested gap between 10.26 and 26.78 mG/mm; 15 mG/mm should
therefore be treated as a conservative provisional gate, not a tuned boundary.

The corrected-magnitude experiment beat the freshly rerun current pipeline on
11 of 12 logs, with median weak-field RMSE of 4.55 versus 6.78 mm. That
comparison does not isolate ambient correction: the encoder-calibrated raw
magnitude control was already 4.08 mm. Most of the difference from the current
pipeline is attributable to the signal/model family, not to the body/world
correction itself.

## Files

- `metrics.csv`: all per-log metrics.
- `aggregate.csv`: median per-log RMSE by method and region.
- `details.json`: slopes, update fractions, fitted models, and solver states.
- Per-log PNGs: held-out prediction errors and correction magnitudes.
