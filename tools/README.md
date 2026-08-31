# Tools

Run tools from the repository root, preferably with the project environment:

```bash
venv/bin/python tools/stats_aggregator.py --help
```

Most tools read cached pipeline output from `backend/run_artifacts/<log>/cache/all.npz`.
Use the pipeline first if a log has no cache.

## Current Utilities

| Tool | Purpose |
| --- | --- |
| `tools/stats_aggregator.py` | Main summary/report tool for pipeline cache metrics. Handles both front and rear cache key shapes. |
| `tools/export_sst_csv.py` | Export solved and ground-truth travel into SST-compatible CSV files. |
| `tools/front/mag_correction_solver.py` | Reusable continuous body/world ambient-field smoother and constrained low-field XYZ model. |
| `tools/front/experiment_mag_correction.py` | Held-out front-log comparison of raw and body/world-corrected magnetometer travel signals. |
| `tools/front/experiment_unsupervised_mag_xyz.py` | Encoder-free XYZ-path and ambient-correction experiment bootstrapped from the accelerometer-trained scalar magnet model. |
| `tools/front/experiment_mag_nuisance_observability.py` | Encoder-free nuisance-field experiment using strong-region propagation, curve-normal residuals, and accelerometer-derived XYZ tangents. |
| `tools/front/joint_mag_accel_solver.py` | Sparse joint latent-travel/body-field/world-field solver with gyro transport, strong-region anchors, and optional relative acceleration factors. |
| `tools/front/experiment_joint_mag_accel.py` | Encoder-free step-3 experiment and ablation for the joint magnetic/acceleration solver; encoder travel is used only for final metrics. |
| `tools/rear/analyze_rear_chunking_tradeoffs.py` | Current rear mag-model chunking/training tradeoff analysis, including the mag-gated blend follow-up. |
| `tools/rear/analyze_rear_zv_accel_correction.py` | Current rear ZV acceleration correction sweep and selected-variant analysis. |
| `tools/linkage/export_horst_linkage_curve.py` | Generate sampled rocker-angle to wheel-travel linkage curves for the rear pipeline. |
| `tools/linkage/horst_linkage_example.py` | Inspect and plot the Horst linkage geometry used by the curve exporter. |

## Legacy Front-Pipeline Diagnostics

`tools/front/` contains scripts that target the original/front pipeline cache keys such as
`accel/lpfhp/proj` and `mag/proj/corr/lpf`. They are still useful when investigating those
logs, but they are not the first place to look for rear-pipeline behavior.

## Rear Experiment Archive

`tools/rear/archive/` contains reproducibility scripts for completed rear investigations.
The reports in `reports/` usually contain the conclusion already, so prefer reading the
report before rerunning the full experiment.

## Adding Or Keeping Tools

- Add an argparse CLI with a working `--help`.
- Avoid hardcoded log names unless they are defaults that the user can override.
- Put current rear work in `tools/rear/`; move completed one-off rear sweeps to `tools/rear/archive/`.
- Put original/front pipeline diagnostics in `tools/front/`.
- Keep reusable exports and shared reporting tools at the top level or in a focused utility folder.
- If a script imports backend modules directly, compute `REPO_ROOT` from `__file__` and add `backend/` to `sys.path`.
- For plotting scripts, set `MPLCONFIGDIR` to a writable temp directory before importing `matplotlib`.
