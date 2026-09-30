# Rear ZV Acceleration Correction Findings

Logs:

- `log148_rear`
- `log149_rear`
- `log150_rear`
- `log151_rear`
- `log152_rear`
- `log153_rear`
- `log154_rear`

## What Changed

I added a full-series acceleration correction step that uses mag zero-velocity points before the rear mag model and solver:

1. Integrate projected rear acceleration once to get raw velocity.
2. Sample raw integrated velocity at mag ZV points.
3. Treat those sampled velocities as drift, because true suspension velocity should be zero at mag extrema.
4. Subtract the drift derivative from the whole projected acceleration signal.

The production pipeline is currently wired to `smoothed_bias` mode with all mag ZV points and a `50 ms` Gaussian smoothing window:

- input: `accel/lphp/proj`
- output: `accel/lphp/proj/zv`
- downstream ZV output: `mag_zv_points/accel_corr`

## Experiment Summary

The best method depends on the objective:

| Variant | Accel RMSE | Accel Corr | Mag Bin RMSE | Worst Bin | Notes |
|---|---:|---:|---:|---:|---|
| raw centered baseline | 22.64 | 0.796 | 9.22 | 18.33 | Current-cache raw centered-ZV behavior |
| raw, `80 ms` ZV spacing | 22.64 | 0.796 | 8.38 | 15.30 | Best filter-only control |
| linear `200 mg / 50 ms` | 21.76 | 0.809 | 8.91 | 14.58 | Best direct accel improvement among selected current-cache variants |
| smoothed-bias all ZVs | 22.08 | 0.805 | 7.34 | 12.57 | Best mag-model bin RMSE |

Filter-only controls help, but they do not get all the way to the smoothed-bias result:

| Variant | Accel RMSE | Mag Bin RMSE |
|---|---:|---:|
| raw acceleration, all ZVs | 22.64 | 9.22 |
| raw acceleration, `80 ms` ZV spacing | 22.64 | 8.38 |
| raw acceleration, `200 mg / 50 ms` ZVs | 22.64 | 8.97 |
| corrected acceleration, smoothed-bias all ZVs | 22.08 | 7.34 |

So there are two useful effects: cleaner ZV selection can help the centered chunks, and full-series ZV acceleration correction helps more for equal-bin accuracy.

## Pipeline Result

After wiring `smoothed_bias` into `backend/pipeline_rear.py` and rerunning logs 148-154:

| Metric | Old | New | Delta |
|---|---:|---:|---:|
| Mean mag-model bin RMSE | 10.13 mm | 7.34 mm | -27.6% |
| Mean solved bin RMSE | 10.02 mm | 6.97 mm | -30.4% |
| Mean high-travel solved RMSE (`travel > 100 mm`) | 14.39 mm | 6.69 mm | -53.5% |
| Mean sample-weighted solved RMSE | 4.28 mm | 6.14 mm | +43.5% |
| Mean low-travel solved RMSE (`travel < 30 mm`) | 4.46 mm | 9.47 mm | +112.5% |

The correction strongly improves equal-bin and high-travel behavior, but it hurts the dense low-travel region. This matches the earlier chunking tradeoff: the previous method was very good where most samples live, while the ZV-corrected acceleration makes the learned curve more useful across the full travel range.

## Interpretation

- Mag ZV points are valid acceleration anchors, but using them as a full-series correction changes the training distribution. The learned model becomes much less dominated by low/mid travel.
- Among the selected current-cache variants, exact ZUPT-style linear correction gives the cleanest direct acceleration waveform, but the smoothed-bias correction gives the best downstream bin RMSE.
- Prominence/separation filtering is useful on its own. Moderate filters (`200 mg / 50 ms`) are a good conservative option, especially if future logs have noisier mag extrema, but the current best equal-bin result comes from smoothed-bias correction over all ZVs.
- The current `smoothed_bias` default is good if equal-bin/high-travel accuracy matters most. It is not good if sample-weighted low-travel RMSE is the primary metric.

## Next Recommendation

The best next step is a hybrid output:

- keep the old centered/raw model or the previous mag-gated blend at low travel,
- use the ZV-corrected acceleration model at higher travel,
- fade between them by mag/travel percentile.

That should preserve the old low-travel strength while keeping the large high-travel gains from the corrected acceleration.

Artifacts:

- Pipeline stats: `experiments/legacy_stats/rear/rear_zv_accel_pipeline_148_154/report.txt`
- Exact selected-variant report: `reports/rear_zv_accel_correction_exact_148_154/report.md`
- Chunk-capped screen/control report: `reports/rear_zv_accel_correction_screen_148_154/report.md`
