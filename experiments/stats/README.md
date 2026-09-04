# Stats experiment catalog

Each child directory is a durable snapshot produced by `tools/stats.py run`. Track the full
directory in Git: `experiment.json` records intent and exact provenance, `metrics.csv` supports
future comparisons after caches are overwritten, `logs.csv` is a compact fingerprint index,
and `report.txt` plus `tables/` make the result readable without rerunning the pipeline.

Pipeline caches in `backend/run_artifacts/` are disposable and are not the experiment catalog.
Old reports under `reports/stats_aggregator/` are retained as a legacy backup.
