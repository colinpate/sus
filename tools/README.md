# Tools

Run tools from the repository root, preferably with the project environment:

```bash
venv/bin/python tools/logs.py --help
venv/bin/python tools/stats.py --help
```

Most tools read cached pipeline output from `backend/run_artifacts/<log>/cache/all.npz`.
Use the pipeline first if a log has no cache.

Bulk annotations can select usable logs with repeated metadata filters. For example,
this adds matching logs to a set and gives each one a nested processing override:

```bash
venv/bin/python tools/logs.py annotate \
  --where 'bike_model=Specialized Stumpjumper' \
  --where pod_version=2 \
  --where pipeline=front \
  --set stumpjumper-front-pod-v2 \
  --override steps.angle_to_travel.top_zeroangle=1.52788
```

Add `--all-statuses` to include non-usable matches. Without explicit log IDs,
`annotate` requires at least one `--where` filter so it cannot accidentally update
the entire registry.

## Current Utilities

| Tool | Purpose |
| --- | --- |
| `tools/logs.py` | Import binary logs, reuse metadata presets, annotate/tag logs, validate the registry, and run the configured front/rear pipeline. |
| `tools/stats.py` | Create, catalog, inspect, and compare versioned stats experiments from registry groups. Rejects stale caches and always saves centered plus uncentered metrics. |
| `tools/stats_aggregator.py` | Internal metric calculation engine used by `tools/stats.py`; direct command-line use is disabled. |
| `tools/export_sst_csv.py` | Export solved and ground-truth travel into SST-compatible CSV files. |
| `tools/front/mag_nuisance/` | Body/world magnetic-nuisance solvers, current encoder-blind experiments, supervised diagnostics, and archived prototypes. Start with its `README.md`. |
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
