# Rear Accel Filtering Findings

Logs: `log148_rear` through `log154_rear`.

## Summary

Changing the rear projected-accel HPF from 2 Hz to 4 Hz is a real downstream win with the current mag-ZV smoothed-bias correction.

| Pipeline | Accel HPF | Mag bin RMSE | Solved bin RMSE | Notes |
|---|---:|---:|---:|---|
| Previous ZV-corrected rear pipeline | 2 Hz | 7.34 mm | 6.97 mm | `experiments/legacy_stats/rear/rear_zv_accel_pipeline_148_154/report.txt` |
| Earlier HPF check | 1 Hz | 6.74 mm | 6.39 mm | `experiments/legacy_stats/rear/rear148_154_oldchunking_zv_acc_smoothed_bias_x0_0_hpf1hz/report.txt` |
| New pipeline rerun | 4 Hz | 6.14 mm | 5.85 mm | `experiments/legacy_stats/rear/rear_accel_filtering_148_154_pipeline_hpf4/report.txt` |

The exact standalone sweep also showed better direct acceleration agreement against differentiated travel:

| Variant | Accel RMSE | Accel corr | Mag bin RMSE | Solver bin RMSE |
|---|---:|---:|---:|---:|
| 2 Hz, order 2 | 22.08 m/s^2 | 0.805 | 6.74 mm | 6.39 mm |
| 3 Hz, order 2 | 21.86 m/s^2 | 0.807 | 6.41 mm | 6.07 mm |
| 4 Hz, order 2 | 21.48 m/s^2 | 0.809 | 6.14 mm | 5.85 mm |

## What Changed

- `backend/pipeline_rear.py` now uses `ACCEL_HP_FREQ = 4` for the rear accel highpass before axis estimation/projection.
- `backend/rear_mag_model.py`'s standalone `project_accel()` helper now matches the 4 Hz rear default.
- Added `tools/rear/analyze_rear_accel_filtering.py` to sweep HPF cutoff, HPF order, lowpass cutoff, and ZV bias smoothing using cached logs.

## Sweep Notes

- HPF cutoff was the main useful lever. 3 Hz improved the pipeline; 4 Hz improved it more.
- The projection axis barely changed across cutoff sweeps (`axis_dot_cached` was about 1.0), so the gain is from the 1D signal content, not a different travel-axis estimate.
- Keeping the 40 Hz pre-lowpass was better than lowering it. The 20 Hz LPF reduced direct accel RMSE but hurt accel correlation and downstream mag-model bin RMSE.
- HPF order did not beat the 4 Hz, order-2 option. 2 Hz/order-1 was decent, 2 Hz/order-4 was worse.
- ZV bias smoothing should stay at 50 ms for now. 25 ms looked better on direct accel RMSE but hurt mag-model RMSE; 100-200 ms also degraded downstream metrics.
- The no-ZV-correction control at 1 Hz was much worse, so the ZV correction remains essential after changing the HPF.

## Remaining Caveat

4 Hz improves the mean and most logs, but `log154_rear` still has a weak high-travel tail: solved 120-150 mm bin RMSE is 15.56 mm. That bin is very sparsely represented in these logs, so I would keep the 4 Hz HPF but continue treating high-travel tail calibration as a separate model-shape/data-coverage problem rather than a filter problem.
