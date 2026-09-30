#!/usr/bin/env python3
"""Select fixed offline IMU baseline filters from a frozen development cohort."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'backend'))
sys.path.insert(0, str(REPO_ROOT / 'tools'))
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp')
from evaluation_baselines import integrate_imu, resample_valid
from log_registry import LogRegistry
from stats_aggregator import build_mask, build_imu_bad_mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cutoffs', type=float, nargs='+', default=[.1,.2,.3,.5,.75,1,1.5,2,3,4])
    p.add_argument('--sets', nargs='+', default=['front-default','rear-default'])
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise SystemExit('Choose a new output directory; existing sweep records are immutable')
    registry = LogRegistry.load()
    spec = tomllib.loads((REPO_ROOT/'experiments/mag_calibration/specs/cross_setup_front_pod_v2_full_v4.toml').read_text())
    setups = {log: setup for setup, logs in spec['setup_logs'].items() for log in logs}
    units = spec.get('analysis_units', {})
    cohort = {l.log_id:l for group in args.sets for l in registry.select(set_name=group)}
    rows, records, failures = [], [], []
    for log in cohort.values():
        path = REPO_ROOT/'backend/run_artifacts'/log.log_id/'cache/all.npz'
        try:
            with np.load(path, allow_pickle=False) as cache:
                key = 'accel/proj' if log.pipeline == 'front' else 'accel/lpf/proj'
                t, a = cache[key+'__t'], cache[key+'__x']
                target_t, gt = cache['travel__t'], cache['travel__x'].reshape(-1)
                mask = build_mask(cache, 'travel/solved', 'travel')
                bad = build_imu_bad_mask(cache, t)
                record = {'log':log.log_id, 'pipeline':log.pipeline, 'setup':setups.get(log.log_id, log.pipeline+'-'+str(log.metadata.get('bike_model'))+'-pod'+str(log.metadata.get('pod_version'))),
                          'unit':units.get(log.log_id, log.log_id.split('_rear_')[0] if '_rear_' in log.log_id else log.log_id),
                          'source_fingerprint':str(cache['__run_fingerprint'].item()), 'accel_key':key,
                          'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            predictions = {}
            for placement in ['velocity','displacement','both']:
                for cutoff in args.cutoffs:
                    pred = integrate_imu(t, a, cutoff_hz=cutoff, placement=placement, bad=bad)
                    predictions[placement,cutoff] = resample_valid(t,pred,target_t)
            common = mask.copy()
            for pred in predictions.values():
                common &= np.isfinite(pred)
            if not common.any():
                raise ValueError('No common valid scoring samples')
            record['scored_samples'] = int(common.sum())
            records.append(record)
            for (placement,cutoff), pred in predictions.items():
                error = pred[common]-gt[common]
                rmse = float(np.sqrt(np.mean((error-error.mean())**2)))
                rows.append({**{k:record[k] for k in ['log','pipeline','setup','unit']}, 'placement':placement,'cutoff_hz':cutoff,'centered_rmse_mm':rmse,'samples':int(common.sum())})
            print(log.log_id, 'scored', int(common.sum()), flush=True)
        except Exception as exc:
            failures.append({'log':log.log_id,'error':str(exc)})
            print(log.log_id, 'FAILED', exc, flush=True)
    summaries=[]
    for pipeline in sorted({r['pipeline'] for r in rows}):
        for placement in ['velocity','displacement','both']:
            for cutoff in args.cutoffs:
                selected=[r for r in rows if r['pipeline']==pipeline and r['placement']==placement and r['cutoff_hz']==cutoff]
                setup_scores=[]
                for setup in sorted({r['setup'] for r in selected}):
                    sr=[r for r in selected if r['setup']==setup]
                    unit_scores=[np.mean([r['centered_rmse_mm'] for r in sr if r['unit']==unit]) for unit in sorted({r['unit'] for r in sr})]
                    setup_scores.append(float(np.mean(unit_scores)))
                summaries.append({'pipeline':pipeline,'placement':placement,'cutoff_hz':cutoff,'setup_balanced_rmse_mm':float(np.mean(setup_scores))})
    winners={kind:min([r for r in summaries if r['pipeline']==kind],key=lambda r:r['setup_balanced_rmse_mm']) for kind in sorted({r['pipeline'] for r in summaries})}
    for name,data in [('per_log.csv',rows),('summary.csv',summaries)]:
        if data:
            with (out/name).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    manifest={'sets':args.sets,'cutoffs_hz':args.cutoffs,'placements':['velocity','displacement','both'],'filter_order':2,'lowpass_hz':40,'edge_s':2,
              'objective':'Mean RMSE within parent recording, then mean within setup, then equal mean across setups; separately front/rear',
              'protocol':'Development-set tuning; zero-phase filters; zero initial velocity/displacement; fixed common scoring samples per recording across candidates',
              'sources':records,'failures':failures,'selected':winners,
              'code_sha256':{str(path.relative_to(REPO_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__).resolve(),REPO_ROOT/'backend/evaluation_baselines.py',REPO_ROOT/'tools/stats_aggregator.py']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# IMU integration baseline filter sweep','',manifest['protocol']+'.','',manifest['objective']+'.','',f'Completed {len(records)} of {len(cohort)} recordings; {len(failures)} failures.','', 'Input: front raw projected acceleration; rear 40 Hz low-pass projected acceleration before magnetic ZV correction. The baseline applies a 40 Hz second-order low-pass in both cases.','', 'Two seconds at each valid segment boundary are excluded for every candidate. This is a common edge policy, not a claim that every cutoff fully settles within two seconds.','', '| Pipeline | HPF location | Cutoff Hz | Setup-balanced centered RMSE mm |','|---|---|---:|---:|']
    for kind,r in winners.items():
        lines.append(f"| {kind} | {r['placement']} | {r['cutoff_hz']} | {r['setup_balanced_rmse_mm']:.3f} |")
    lines += ['', 'Selection uses the evaluation cohort and is not held-out validation. Scores include only the common finite active samples passing the existing reference/IMU quality masks. Source cache hashes and fingerprints are frozen in the manifest; these are historical preprocessing inputs, not a claim that every cache matches the later edited backend.', '', 'Failures: '+json.dumps(failures)]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(winners,indent=2))
    if failures:
        raise SystemExit('Incomplete cohort: inspect recorded failures before using the selected settings')

if __name__=='__main__':
    main()
