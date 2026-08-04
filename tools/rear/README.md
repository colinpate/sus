# Rear-Pipeline Tools

These scripts target `backend/pipeline_rear.py` caches and current rear model work.

| Tool | Purpose |
| --- | --- |
| `analyze_rear_chunking_tradeoffs.py` | Compare paired, centered, hybrid, and mag-gated rear mag-model training variants. |
| `analyze_rear_zv_accel_correction.py` | Sweep acceleration correction methods based on mag zero-velocity points. |

The rear pipeline now uses `accel/lphp/proj/zv` and `mag_zv_points/accel_corr` downstream.
Use `tools/stats_aggregator.py rear --center-errors --deep-dive` for the current default rear
log set.

Historical rear experiments live in `archive/`.
