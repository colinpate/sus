# Mag-calibration experiments

## What belongs in git

The durable experiment record is intentionally smaller than a runnable cache.
Track:

- every specification in `specs/`;
- `manifest.json`, `status.json`, and `trial_schedule.csv` for provenance and
  exact randomization;
- reports, figures, aggregate/per-log/setup summaries, failure tables, and the
  derived tables in `analysis/`;
- the experiment runners and analysis tools under `tools/`.

Do not track:

- `trials/`, `calibrations/`, or `predictions/`, which are resumable execution
  caches and can be regenerated;
- `trial_metrics.csv`, the large row-level analysis table. Keep it locally while
  actively analyzing a run, then regenerate it from the spec when necessary;
- `.DS_Store` or ad hoc copies of plots and tables.

The local `.gitignore` enforces this split while overriding the repository-wide
CSV/JSON exclusions for the compact outputs that should be versioned. Removing
the execution caches means `run` will recompute a study and `summarize` cannot
rebuild row-level tables until that study has been rerun.

## Retained experiment catalog

| Experiment | Role |
|---|---|
| `random-window-front-v1-shorter` | Very-short-window failure/readiness study (1–5 s). |
| `random-window-front-v1` | Controlled Stumpjumper/pod-v2 duration study and mechanism-analysis source. |
| `random-window-front-v2` | Setup-stratified source for the front window-length mechanism analysis. |
| `random-window-front-v4` | Higher-repeat, all-LSM6DSO32 front duration study. |
| `random-window-front-v4-jamaal` | Original-TR11 duration study. |
| `cross-log-front-*` | Same-setup calibration-transfer controls. |
| `cross-setup-front-three-setup-v1` | Only retained study containing Stumpjumper/pod-v1. |
| `cross-setup-front-pod-v2-sample-v2` | Balanced pilot retained to quantify sampling efficiency. |
| `cross-setup-front-pod-v2-full-v3` | Primary complete pod-v2 cross-setup result. |
| `solver-window-front-phase1` | Pilot and repeat-0 provenance for the downstream study. |
| `solver-window-front-phase2` | Primary repeated downstream-solver result. |

`random-window-front-v3` was retired because v4 is the higher-repeat version of
the same 24-log question, while v1 and the Jamaal study retain the useful longer
duration endpoints. The incomplete `random-window-rear-v1` smoke run was also
retired; rear evidence should start with a fresh, deliberately sized spec.

## Random-window sweeps

The random-window sweep uses deterministic, nested windows. For each log and
repeat it samples one random center that can contain the longest requested
duration; all shorter windows share that center. Every trainer therefore sees
the same samples at a given duration, and adding a new trainer does not change
the randomization.

Prepare a schedule without fitting anything:

```bash
venv/bin/python tools/mag_calibration_sweep.py schedule \
  experiments/mag_calibration/specs/random_window_front_v1.toml \
  --output-dir experiments/mag_calibration/runs/random-window-front-v1
```

Run or resume it (completed per-trial JSON files are skipped):

```bash
venv/bin/python tools/mag_calibration_sweep.py run \
  experiments/mag_calibration/specs/random_window_front_v1.toml \
  --output-dir experiments/mag_calibration/runs/random-window-front-v1 \
  --workers 4
```

Use `--max-trials N` for a bounded smoke run, or `--max-trials-per-log N` to
exercise every cohort member. Use `summarize` to rebuild all tables, plots, and
the Markdown report from the per-trial files without fitting.
Failures remain in the planned cohort and are reported rather than replaced by
new random windows; `--retry-failures` explicitly retries them.

After the metric schema changes, backfill successful fits and rebuild summaries
from the run's frozen manifest without refitting:

```bash
venv/bin/python tools/mag_calibration_sweep.py refresh \
  --output-dir experiments/mag_calibration/runs/random-window-front-v1
```

The current metric set reports locally centered, production-anchored, and
full-log-fixed-offset errors; normalization by reference-travel standard
deviation and robust 5th–95th percentile span; travel support; prediction scale;
and locally centered, anchored, and fixed-offset errors in fixed 30 mm travel
bins.

Fresh run directories contain:

- `manifest.json`: frozen spec, cache cohort, exclusions, and code state.
- `trial_schedule.csv`: the complete deterministic trial schedule.
- `trials/*.json`: atomic, resumable fit results and failure records (local).
- `trial_metrics.csv`: one row per fit and evaluation scope.
- `per_log_summary.csv`: repeat-level results collapsed within each log.
- `aggregate_summary.csv`: equally weighted log-level estimates and bootstrap CIs.
- `learning_curve.png` / `.pdf`: aligned RMSE versus training duration.
- `report.md` and `status.json`: readable results and completion state.

With the local row-level results and trial caches present, the front
window-length mechanism analysis can be regenerated with:

```bash
venv/bin/python tools/analyze_mag_calibration_window_sweep.py
```

## Cross-log transfer

Cross-log experiments use a separate spec runner because one fitted source
calibration is deliberately reused across multiple evaluation logs:

```bash
venv/bin/python tools/mag_calibration_transfer_sweep.py schedule \
  experiments/mag_calibration/specs/cross_log_front_stumpjumper_pod_v2_v1.toml \
  --output-dir experiments/mag_calibration/runs/cross-log-front-stumpjumper-pod-v2-v1

venv/bin/python tools/mag_calibration_transfer_sweep.py run \
  experiments/mag_calibration/specs/cross_log_front_stumpjumper_pod_v2_v1.toml \
  --output-dir experiments/mag_calibration/runs/cross-log-front-stumpjumper-pod-v2-v1
```

Use `--trainer oracle-power` or another configured trainer to run one matrix at
a time; `--max-trials` bounds a smoke run without changing the frozen schedule.

Each source log is fitted once per trainer and evaluated on every target log.
Diagonal pairs provide the per-log calibration baseline; off-diagonal pairs
measure transfer. The first spec intentionally uses only Stumpjumper/pod-v2
logs, so differences are not confounded with bike geometry or sensor generation.
It includes the production self-supervised learner, a supervised oracle in the
same power-curve family, and the paper-friendly binned-median monotonic oracle.

Transfer reports retain both aligned and production-anchored errors. Aligned
error isolates curve-shape transfer, while anchored error includes absolute
offset/mounting transfer. `comparison_summary.csv` directly subtracts each
target's self-supervised diagonal error from every shared source calibration.

### Three-setup transfer matrix

The complete front cross-setup experiment contains seven Stumpjumper/pod-v1,
11 Stumpjumper/pod-v2, and six TR11/pod-v2 logs. It fits each full source log
with the self-supervised learner and both paper-facing oracles, then evaluates
every curve on every target log. The diagonal and within-setup pairs are retained
as controls, producing 1,728 evaluations:

```bash
venv/bin/python tools/mag_calibration_transfer_sweep.py run \
  experiments/mag_calibration/specs/cross_setup_front_three_setup_v1.toml \
  --output-dir experiments/mag_calibration/runs/cross-setup-front-three-setup-v1

venv/bin/python tools/analyze_mag_calibration_cross_setup.py \
  --spec experiments/mag_calibration/specs/cross_setup_front_three_setup_v1.toml \
  --run-dir experiments/mag_calibration/runs/cross-setup-front-three-setup-v1
```

The setup-aware analysis reports source-balanced medians and crossed-bootstrap
intervals that independently resample source and target logs. Its primary metric
is the transfer penalty relative to the target log's own calibration from the
same trainer. It also records normalized and production-anchored errors, magnetic
support by setup, a directed transfer heatmap, and a standalone Markdown report.
The generic runner's pooled report is not the primary summary for this design.

### Expanded pod-v2 setup sample

The current expansion excludes Stumpjumper/pod-v1 and uses a deterministic,
setup-balanced sample from `front-default`: five independent recordings each
from Stumpjumper/pod-v2, original TR11/pod-v2, TR11-2025/pod-v2, and
Slayer/pod-v2. Seed `20260916` selected the logs without using their accuracy;
the referenced stats run was used only to confirm that all candidate outputs
were fresh and successful. One Slayer chunk was sampled from each of its five
parent recordings. The frozen 20-log design has 1,200 evaluations:

```bash
venv/bin/python tools/mag_calibration_transfer_sweep.py run \
  experiments/mag_calibration/specs/cross_setup_front_pod_v2_sample_v2.toml \
  --output-dir experiments/mag_calibration/runs/cross-setup-front-pod-v2-sample-v2

venv/bin/python tools/analyze_mag_calibration_cross_setup.py \
  --spec experiments/mag_calibration/specs/cross_setup_front_pod_v2_sample_v2.toml \
  --run-dir experiments/mag_calibration/runs/cross-setup-front-pod-v2-sample-v2
```

The setup analyzer accepts optional `analysis_units` in the spec. It collapses
multiple derived chunks within each source/target parent pair, excludes
same-parent pairs from within-setup transfer, and resamples parent recordings
rather than treating derived chunks as independent evidence.

### Complete pod-v2 setup matrix

The final study expands the pilot to all 38 usable pod-v2 logs in
`front-default`: 11 Stumpjumper, six original TR11, 14 TR11-2025, and seven
Slayer chunks representing five parent recordings. The parent-balanced analysis
therefore contains 36 independent units. The complete 38×38×3 matrix has 4,332
evaluations:

```bash
venv/bin/python tools/mag_calibration_transfer_sweep.py run \
  experiments/mag_calibration/specs/cross_setup_front_pod_v2_full_v3.toml \
  --output-dir experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v3

venv/bin/python tools/analyze_mag_calibration_cross_setup.py \
  --spec experiments/mag_calibration/specs/cross_setup_front_pod_v2_full_v3.toml \
  --run-dir experiments/mag_calibration/runs/cross-setup-front-pod-v2-full-v3
```

This complete run is the primary pod-v2 cross-setup result. The sampled v2 run
is retained as a pilot and as evidence that five independent recordings per
setup recovered the same qualitative result efficiently.

The three evaluation scopes answer different questions:

- `training_window`: local fit quality on the selected block.
- `full_log`: how much training data is needed to represent the whole recording.
- `full_log_excluding_training`: within-recording generalization without direct
  evaluation-sample overlap.

The initial controlled experiment uses the reference-derived `boring_mask` for
active-time selection and scoring. Replace it with a deployable activity signal
before making a production claim about automatic window selection.

## Downstream solver window-length experiment

Phase 1 takes selected calibrations from a completed random-window run, injects
each curve into the front pipeline's downstream stages, and runs both fusion
solves over the complete log:

```bash
venv/bin/python tools/mag_calibration_solver_sweep.py schedule \
  experiments/mag_calibration/specs/solver_window_front_phase1.toml \
  --output-dir experiments/mag_calibration/runs/solver-window-front-phase1

venv/bin/python tools/mag_calibration_solver_sweep.py run \
  experiments/mag_calibration/specs/solver_window_front_phase1.toml \
  --output-dir experiments/mag_calibration/runs/solver-window-front-phase1
```

The runner is resumable and records the raw mag-model output, first fusion,
nuisance correction, and final solved output. It scores the training window, a
fixed common core, the full log, and the full log excluding training. Window
metrics also retain a full-log-fixed alignment, preventing each window from
hiding local bias through its own offset fit. Use `--max-trials N` for a pilot.

The Phase 1 spec deliberately uses one deterministic repeat across 11 comparable
front logs and five durations. It is a representative downstream test rather
than a high-powered random-window significance study; additional repeats can be
added after selecting the most informative duration range.

Phase 2 expands that design to four independent nested-window centers per log:

```bash
venv/bin/python tools/mag_calibration_solver_sweep.py run \
  experiments/mag_calibration/specs/solver_window_front_phase2.toml \
  --output-dir experiments/mag_calibration/runs/solver-window-front-phase2 \
  --workers 3
```

Its spec imports the already-completed Phase 1 center, so only missing source
calibrations are solved. Repeats are paired across durations, collapsed within
each log, and only then aggregated across logs. Phase 2 omits the large prediction
arrays while retaining stagewise trial metrics and runtime diagnostics.
