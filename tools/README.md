# Tools

Run tools from the repository root, preferably with the project environment:

```bash
venv/bin/python tools/logs.py --help
venv/bin/python tools/stats.py --help
```

Most tools read cached pipeline output from `backend/run_artifacts/<log>/cache/all.npz`.
Use the pipeline first if a log has no cache.

## Current Utilities

| Tool | Purpose |
| --- | --- |
| `tools/logs.py` | Import binary logs, reuse metadata presets, annotate/tag logs, validate the registry, and run the configured front/rear pipeline. |
| `tools/stats.py` | Create, catalog, inspect, and compare versioned stats experiments from registry groups. Rejects stale caches and always saves centered plus uncentered metrics. |
| `tools/stats_aggregator.py` | Internal metric calculation engine used by `tools/stats.py`; direct command-line use is disabled. |
| `tools/mag_calibration_experiment.py` | Fit front or rear mag-to-travel curves on selected cached log/time windows, save portable calibrations, and evaluate them on other windows or logs. |
| `tools/mag_calibration_sweep.py` | Schedule, resume, summarize, and plot deterministic random-window learning-curve experiments. |
| `tools/mag_calibration_transfer_sweep.py` | Fit each source calibration once and resumably evaluate the complete cross-log transfer matrix from a TOML spec. |
| `tools/analyze_mag_calibration_cross_setup.py` | Summarize a complete transfer matrix by bike/sensor setup with source-balanced estimates, crossed-bootstrap intervals, signal-support diagnostics, and directed heatmaps. |
| `tools/mag_calibration_solver_sweep.py` | Inject window-trained front calibrations, rerun both full-log fusion solves, and compare every downstream stage across fixed evaluation scopes. |
| `tools/analyze_mag_calibration_window_sweep.py` | Compare front learning-curve runs while controlling evaluation support, centering, travel bins, and hardware/setup strata. |
| `tools/export_sst_csv.py` | Export solved and ground-truth travel into SST-compatible CSV files. |
| `tools/front/mag_nuisance/` | Body/world magnetic-nuisance solvers, current encoder-blind experiments, supervised diagnostics, and archived prototypes. Start with its `README.md`. |
| `tools/rear/analyze_rear_chunking_tradeoffs.py` | Current rear mag-model chunking/training tradeoff analysis, including the mag-gated blend follow-up. |
| `tools/rear/analyze_rear_zv_accel_correction.py` | Current rear ZV acceleration correction sweep and selected-variant analysis. |
| `tools/linkage/export_horst_linkage_curve.py` | Generate sampled rocker-angle to wheel-travel linkage curves for the rear pipeline. |
| `tools/linkage/horst_linkage_example.py` | Inspect and plot the Horst linkage geometry used by the curve exporter. |

## Windowed Mag Calibration Experiments

Train and score on fixed accumulated-active-time blocks:

```bash
venv/bin/python tools/mag_calibration_experiment.py pair \
  --train-log log103 --train-start-s 0 --train-stop-s 60 \
  --eval-log log103 --eval-start-s 60 --eval-stop-s 120 \
  --calibration-out experiments/mag_calibration/log103-first-60s.json \
  --metrics-out experiments/mag_calibration/log103-next-60s.csv
```

Build an inter-log transfer table:

```bash
venv/bin/python tools/mag_calibration_experiment.py matrix \
  --train-logs log168_rear_1 log168_rear_2 \
  --eval-logs log168_rear_1 log168_rear_2 \
  --metrics-out experiments/mag_calibration/rear-transfer.csv
```

Train a supervised monotonic oracle on one window and transfer it to another:

```bash
venv/bin/python tools/mag_calibration_experiment.py pair \
  --trainer oracle-isotonic \
  --train-log log103 --train-start-s 0 --train-stop-s 60 \
  --eval-log log104 \
  --calibration-out experiments/mag_calibration/log103-oracle.json \
  --metrics-out experiments/mag_calibration/log103-oracle-to-log104.csv
```

`--trainer oracle-power` fits reference travel with the same three-parameter
power curve used by the production learner (plus its otherwise-free absolute
offset). It isolates error due to self-supervised learning from error imposed by
the curve family itself.

`--trainer oracle-binned-median --oracle-bins 100` provides a less flexible,
easier-to-explain alternative. It takes the median magnetic value and reference
travel in uniform magnetic bins, then applies a weighted monotonic fit to those
points. `oracle-isotonic` fits all selected reference samples directly. Both
infer increasing/decreasing direction from the training window, clip predictions
outside its magnetic support, and never read reference travel from the evaluation
window. Their curves already include absolute reference travel, so the target's
self-supervised offset adjustment is not applied.

Apply a saved curve to another log/window without fitting again:

```bash
venv/bin/python tools/mag_calibration_experiment.py apply \
  --calibration experiments/mag_calibration/log103-first-60s.json \
  --eval-log log104 --eval-start-s 0 --eval-stop-s 60 \
  --metrics-out experiments/mag_calibration/log103-to-log104.csv
```

Run the same-block experiment and score the stitched output with one log-level
alignment:

```bash
venv/bin/python tools/mag_calibration_experiment.py blocks \
  --log log103 --block-s 60 \
  --metrics-out experiments/mag_calibration/log103-blocks-60s.csv \
  --prediction-out experiments/mag_calibration/log103-blocks-60s.npz
```

`active` is the default time basis and currently accumulates time using
`boring_mask`. That mask is reference-travel-derived and is therefore suitable
for these initial controlled experiments, not as the final production activity
detector. Pass `--*-time-basis elapsed` (or `--time-basis elapsed` for a matrix)
to use ordinary wall-clock time instead.

Training windows retain only self-supervised motion chunks fully contained in
the resolved wall-clock span. Evaluation windows are independently intersected
with `boring_mask`. The reported `aligned_*` metrics use one constant alignment
per requested evaluation window; the `blocks` aggregate applies it once to the
complete stitched prediction. `anchored_*` metrics use the target recording's
existing non-GT front/rear anchoring policy. Per-block rows are diagnostics;
compare the two aggregate rows for the fair stitched-versus-one-curve result.

The front and rear mag-to-travel pipeline steps also accept a
`provided_calibration` object. In that opt-in mode they bypass fitting, recreate
the standard mag-model outputs, and leave downstream solver stages unchanged.
Normal pipeline runs continue to fit from the current recording.

For the repeated random-window experiment, use the versioned TOML specs and
workflow in `experiments/mag_calibration/README.md`. The runner creates the full
schedule before fitting, uses identical nested windows across trainers, records
failures without redrawing, saves each fit atomically for resume, aggregates
within log before weighting logs equally, and emits CSV summaries plus PNG/PDF
learning curves and a Markdown report. `--max-trials-per-log` provides a quick
cohort-wide compatibility smoke test.

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
