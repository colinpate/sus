# Table 2 evaluation baselines

The primary filter selection is [development-v3/report.md](development-v3/report.md).
It uses all 38 front-default and 12 rear-default entries available on 2026-09-27,
with derived chunks collapsed within parent recordings before setup balancing.
This is development-set tuning, not held-out validation.

- `development-v3`: primary expanded grid (0.1 through 32 Hz); front winner is
  both integrals at 1 Hz, rear winner is displacement only at 4 Hz.

Reproduce with a new output directory:

```sh
venv/bin/python tools/sweep_imu_baseline.py \
  --output-dir experiments/imu_baseline/new-sweep \
  --cutoffs .1 .2 .3 .5 .75 1 1.5 2 3 4 6 8 12 16 24 32
```

The sweep records source-cache hashes/fingerprints, code hashes, cohort, failures,
per-log errors, candidate scores, and selected settings. It intentionally evaluates
frozen cached preprocessing, which is identified by its source fingerprint; it
never stamps those caches as fresh for a changed backend. A changed preprocessing
pipeline requires a fresh sweep before reusing its numerical conclusions.

## Pipeline steps

`backend/evaluation_baselines.py` appends two independent diagnostic steps after
production estimation in both pipelines. They do not feed the production stages.
Disable either using `steps.imu_integration_baseline.enabled = false` or
`steps.magnetic_power_oracle.enabled = false` in resolved recording configuration.
These settings, like other processing overrides, affect run provenance.

### IMU-only integration

Output: `travel/baseline/accel` (mm).

Front input is raw projected relative acceleration; rear input is the existing
40 Hz low-pass projected acceleration, before magnetic ZV correction and before
its production high-pass filter. Both receive a second-order 40 Hz low-pass in
this baseline. There are no magnetic constraints or reference-value inputs.
Reference timestamps are used only to align the output with the evaluation grid.

Initial displacement and velocity are zero in every valid segment. Integrals use
trapezoidal integration. High-pass filters are second-order Butterworth filters
applied forward/backward (offline, zero phase). Filtering both integrals applies
the HPF twice; it is a distinct candidate, not equivalent to a single application.

Resample small timestamp gaps to a uniform grid before filtering. Restart at
nonfinite acceleration, flagged IMU dropout intervals, or timestamp gaps exceeding
100 ms or five nominal sample intervals, whichever is larger. Rear currently has
no dropout-mask step, so its baseline uses finite samples and timestamp gaps.
Exclude a fixed two-second margin at each segment edge for every candidate.
This common margin is not a guarantee of complete settling at every cutoff.

Overrides: `cutoff_hz`, `placement` (`velocity`, `displacement`, `both`),
`lowpass_hz`, `order`, `edge_s` under `steps.imu_integration_baseline`.
Tune outside the pipeline and freeze settings across recordings. The defaults
are the v3 selections, separately front/rear.

### Magnetic power oracle

Output: `travel/oracle/mag_power` (mm). Diagnostic array `oracle_power_fit` stores
`[x0, scale, power, travel_offset_mm, training_rmse_mm, training_samples]`.

The oracle fits each recording independently using its reference travel and the
same pre-nuisance magnetic scalar as the self-supervised learner. It uses the
existing multi-start supervised power fitter, now shared in
`backend/mag_calibration.py`. The calibration experiment tool imports this same
function. It estimates the power curve plus an absolute travel offset.

Training uses finite active samples, excludes reference corruption (including an
80 ms halo) and available IMU dropout intervals, and uses the pipeline's default
soft-tail scale (front 50 mG, rear 1 in its magnetic-feature units). The oracle step
has an explicit `pred_soft_mg` override. This is an **in-sample supervised
reference**, not a deployable baseline, an independent accuracy test, or a
mathematically guaranteed global optimum. Insufficient data or fitting errors
fail the run visibly; they are not replaced with a favorable subset.

## Statistics

Standard comparisons now include magnetic self-calibration, first front fusion,
final solved travel, IMU integration, and the supervised oracle. The adjusted
magnetic model and intermediate nuisance outputs remain in pipeline caches for
diagnostics/downstream use, but are removed from the standard comparison list.
The rear pipeline has no separate `travel/fusion1` stage.

When the IMU baseline is present, every standard comparison uses the intersection
of finite samples from all present methods, in addition to existing activity,
reference-quality, and IMU-quality masks. Thus its edge exclusions affect all
methods equally. These scores can differ slightly from older stats experiments
that used the full valid recording; compare methods within the same experiment.
Deep-dive diagnostics retain their existing diagnostic support.

Use the versioned stats workflow and require complete Table 2 outputs:

```sh
venv/bin/python tools/stats.py run table2-front --set front-default --process \
  --require-comparison travel/baseline/accel \
  --require-comparison travel/oracle/mag_power \
  --require-comparison travel/fusion1
venv/bin/python tools/stats.py run table2-rear --set rear-default --process \
  --require-comparison travel/baseline/accel \
  --require-comparison travel/oracle/mag_power
```

Do not use `--allow-partial` for the final table without explicitly reporting
exclusions. Pooling recordings alone overweights setups and split logs; final
paper summaries should aggregate parent recordings within each setup. First
fusion vs final travel measures the combined correction, anchoring, and second
solve, not an isolated nuisance-field ablation. Absolute IMU error is not an
absolute-position capability claim: the IMU baseline has no zero-travel anchor.

After both stats experiments finish, create a paper-oriented summary from their
saved directories:

```sh
venv/bin/python tools/summarize_table2_baselines.py \
  --stats experiments/stats/FRONT_EXPERIMENT experiments/stats/REAR_EXPERIMENT \
  --output-dir experiments/imu_baseline/table2-current
```

This saves per-log metrics, parent-balanced setup summaries (median and quartiles),
a Markdown comparison table, and a manifest linking exact source experiments.
It refuses missing/nonfinite metrics and incomplete source experiments.

The current completed result is `table2-current/report.md`. Its main qualification
is that the complete pipeline does not improve monotonically at every stage on
every setup: the first fusion has lower median centered RMSE on Slayer than the
final corrected/second-solve output. Preserve that exception in the manuscript.
