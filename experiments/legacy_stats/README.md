# Legacy direct-aggregator records

These four bundles predate the versioned `tools/stats.py` catalog but remain
the numerical source for retained rear-pipeline reports. They were moved here
from `reports/` so narrative conclusions and machine-readable evidence have
separate homes.

Do not compare these scores directly with newer `experiments/stats/` runs unless
their cohorts, centering, masks, support, and pipeline state have been checked.
New experiments should use `tools/stats.py`; this directory is historical only.

- `rear/rear_zv_accel_pipeline_148_154`: 2 Hz ZV-corrected baseline.
- `rear/rear148_154_oldchunking_zv_acc_smoothed_bias_x0_0_hpf1hz`: earlier 1 Hz check.
- `rear/rear_accel_filtering_148_154_pipeline_hpf4`: selected 4 Hz rerun.
- `rear/rear_extreme_bin_current_stats`: self-consistent source for the retained extreme-bin analysis.
