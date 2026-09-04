# Log registry and ingestion

`registry.toml` is the source of truth for log descriptions, quality status, processing
profiles, tags, and named sets. The old `*.meta.json` sidecars and `lists/` spreadsheet
are retained temporarily only for migration verification.

Run commands from the repository root with the project environment:

```bash
venv/bin/python tools/logs.py validate
venv/bin/python tools/logs.py list --set rear-default
venv/bin/python tools/logs.py show log185_rear
```

## Importing logs

Named presets carry the pod, bike, suspension, binary-format, and processing defaults.
New logs start as `pending-review` unless another status is supplied.

```bash
venv/bin/python tools/logs.py ingest ~/Downloads/log-0082.bin \
  --preset stumpjumper-front-v2 \
  --trail "Poppin Tops" \
  --notes "Wet; fork pressure 82 psi"
```

Batch imports use the same metadata and profiles:

```bash
venv/bin/python tools/logs.py ingest ~/Downloads/log-0082.bin ~/Downloads/log-0083.bin \
  --preset stumpjumper-front-v2 \
  --trail "Ccdh"
```

Use `--copy-from LOG` instead of `--preset` to reuse a known log's profile references
and descriptive fields. Identity, checksums, quality status, and old notes are not copied.

The importer preserves the binary under `logs/raw/`, writes the generated CSV under
`logs/converted/`, records conversion validation results, and updates the registry only
after conversion succeeds. These data files remain ignored by Git.

## Notes, tags, sets, and quality

```bash
venv/bin/python tools/logs.py annotate log-0082 --trail "Ccdh" --tag park --set front-default
venv/bin/python tools/logs.py mark log-0082 usable
venv/bin/python tools/logs.py mark log-0082 corrupt --reason "Magnetometer became constant"
```

Normal group selection includes only `usable` logs. Pass `--all-statuses` when listing or
`--include-nonusable` when processing a log for diagnosis.

Descriptive metadata such as trail, notes, tags, and status does not affect the pipeline
fingerprint. Profile configuration, per-log processing overrides, input contents, linkage
assets, backend source contents, and runtime dependency versions do.

## Processing

```bash
venv/bin/python tools/logs.py process log-0082
venv/bin/python tools/logs.py process --set rear-default
venv/bin/python tools/logs.py process --where pod_version=2
```

Each successful pipeline run writes `backend/run_artifacts/<log>/run.json` and embeds the
same run fingerprint in its NPZ caches. The fingerprint includes uncommitted backend source
changes because it hashes file contents rather than relying on Git commits or timestamps.

## Stats experiments

Use the stats experiment workflow instead of reading caches directly:

```bash
venv/bin/python tools/stats.py run solver-baseline --set front-default --process \
  --notes "Baseline before changing the magnetic correction" --tag baseline

venv/bin/python tools/stats.py run solver-candidate --set front-default --process \
  --notes "Test lower magnetic correction threshold" --tag mag-correction \
  --baseline solver-baseline
```

`--process` refreshes only missing or stale selected caches. Without it, experiment creation
stops if any selected cache does not match the current log input, resolved processing config,
pipeline source (including uncommitted edits), linkage assets, or runtime dependencies. Use
`--allow-partial` only when intentionally excluding those logs; exclusions are recorded.

Every experiment saves both centered and uncentered metrics under `experiments/stats/`, along
with the exact group query, included log fingerprints, notes, tags, readable report, and CSV
tables. These experiment directories should be tracked in Git; pipeline caches remain local.

Browse and compare saved work by memorable name or experiment ID:

```bash
venv/bin/python tools/stats.py list --verbose
venv/bin/python tools/stats.py show solver-candidate
venv/bin/python tools/stats.py compare solver-baseline solver-candidate
```

Comparisons recompute summaries using only the logs present in both experiments, so differently
sized groups can be compared without accidentally attributing group composition to a code change.
Use separate output keys when comparing a new estimator against the old solver on the same inputs:

```bash
venv/bin/python tools/stats.py compare baseline mag-corrected \
  --centering centered \
  --baseline-comparison travel/solved \
  --current-comparison travel/solved/mag_nuisance/fusion2
```

Magnetic-correction outputs are optional: old caches remain valid, while experiments generated
from caches containing `delta_lifted`, `corrected`, or `fusion2` save those metrics automatically.
