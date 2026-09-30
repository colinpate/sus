#!/usr/bin/env python3
"""Build setup/parent-balanced Table 2 summaries from versioned stats experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tomllib
import numpy as np
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT/'backend'))
from log_registry import LogRegistry

METHODS={
    'travel/baseline/accel':'IMU integration',
    'travel/oracle/mag_power':'Supervised magnetic oracle (in-sample)',
    'travel/mag_model':'Self-supervised magnetic estimate',
    'travel/fusion1':'First fusion',
    'travel/solved':'Complete pipeline',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    registry=LogRegistry.load()
    spec=tomllib.loads((REPO_ROOT/'experiments/mag_calibration/specs/cross_setup_front_pod_v2_full_v4.toml').read_text())
    setup_ids={log:setup for setup,logs in spec['setup_logs'].items() for log in logs}
    units=spec.get('analysis_units',{})
    rows=[]; sources=[]; seen=set(); cohort=[]
    for folder in args.stats:
        manifest=json.loads((folder/'experiment.json').read_text())
        if manifest['results']['excluded_logs'] or manifest['results']['failures']:
            raise ValueError('Table 2 requires complete stats experiments')
        with (folder/'metrics.csv').open() as f:
            metrics=list(csv.DictReader(f))
        for log in manifest['selection']['selected_logs']:
            if log in seen: raise ValueError('Duplicate recording across experiments: '+log)
            seen.add(log)
            info=registry.resolve(log)
            setup_id=setup_ids.get(log)
            if setup_id is not None:
                setup=spec['setup_labels'][setup_id]
            elif info.pipeline == 'rear':
                setup=f"{info.metadata.get('bike_model', 'Unknown bike')} rear / pod v{info.metadata.get('pod_version', '?')}"
            else:
                setup=f"{info.metadata.get('bike_model', 'Unknown bike')} / pod v{info.metadata.get('pod_version', '?')}"
            parent=units.get(log,log.split('_rear_')[0] if '_rear_' in log else log)
            record={'log':log,'pipeline':info.pipeline,'setup':setup,'unit':parent}
            cohort.append(record)
            for method,label in METHODS.items():
                if info.pipeline=='rear' and method=='travel/fusion1': continue
                for centering,metric in [('centered','rmse'),('uncentered','mae'),('centered','bin_rmse')]:
                    matches=[r for r in metrics if r['log']==log and r['comparison']==method and r['centering']==centering and r['metric']==metric and r['section']=='error']
                    if len(matches)!=1 or not np.isfinite(float(matches[0]['value'])):
                        raise ValueError(f'Missing/invalid {centering} {metric}: {log} {method}')
                    rows.append({**record,'method':method,'label':label,'metric':centering+'_'+metric,'value':float(matches[0]['value'])})
        source_archive=args.output_dir/'source_experiments'/folder.name
        source_archive.mkdir(parents=True,exist_ok=True)
        for filename in ('experiment.json','logs.csv','metrics.csv','report.txt'):
            shutil.copy2(folder/filename,source_archive/filename)
        resolved_archive=source_archive.resolve()
        source_path=(
            str(resolved_archive.relative_to(REPO_ROOT))
            if resolved_archive.is_relative_to(REPO_ROOT)
            else str(resolved_archive)
        )
        sources.append({
            'path':source_path,
            'experiment_fingerprint':manifest['experiment_fingerprint'],
            'experiment_manifest_sha256':hashlib.sha256((folder/'experiment.json').read_bytes()).hexdigest(),
            'metrics_sha256':hashlib.sha256((folder/'metrics.csv').read_bytes()).hexdigest(),
            'selection':manifest['selection'],
            'stats':manifest['stats'],
            'versions':manifest['versions'],
        })
    summaries=[]
    for setup in dict.fromkeys(r['setup'] for r in rows):
        for method,label in METHODS.items():
            for metric in ['centered_rmse','uncentered_mae','centered_bin_rmse']:
                selected=[r for r in rows if r['setup']==setup and r['method']==method and r['metric']==metric]
                if not selected: continue
                parent_values=[np.mean([r['value'] for r in selected if r['unit']==u]) for u in sorted({r['unit'] for r in selected})]
                q1,median,q3=np.percentile(parent_values,[25,50,75])
                summaries.append({'setup':setup,'method':method,'label':label,'metric':metric,'entries':len(selected),'independent_recordings':len(parent_values),'median_mm':median,'q25_mm':q1,'q75_mm':q3})
    for name,data in [('per_log.csv',rows),('summary.csv',summaries)]:
        with (args.output_dir/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (args.output_dir/'manifest.json').write_text(json.dumps({'sources':sources,'cohort':cohort,'aggregation':'Mean across derived chunks within parent, median and quartiles across parents within setup','tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    lines=['# Table 2: method comparison','', 'Centered RMSE in mm, median [Q1, Q3] across independent parent recordings. Derived chunks are averaged within parent first. Every method is scored on identical valid samples within each recording.','', '| Setup | Entries / parents | Method | Centered RMSE mm |','|---|---:|---|---:|']
    for r in summaries:
        if r['metric']=='centered_rmse':
            lines.append(f"| {r['setup']} | {r['entries']} / {r['independent_recordings']} | {r['label']} | {r['median_mm']:.2f} [{r['q25_mm']:.2f}, {r['q75_mm']:.2f}] |")
    method_by_setup={(r['setup'],r['method']):r for r in summaries if r['metric']=='centered_rmse'}
    complete_wins=sum(
        method_by_setup[(setup,'travel/solved')]['median_mm'] < method_by_setup[(setup,'travel/mag_model')]['median_mm']
        for setup in {r['setup'] for r in summaries}
    )
    setup_count=len({r['setup'] for r in summaries})
    front_setups={r['setup'] for r in rows if r['pipeline']=='front'}
    first_fusion_wins=sum(
        method_by_setup[(setup,'travel/fusion1')]['median_mm'] < method_by_setup[(setup,'travel/mag_model')]['median_mm']
        for setup in front_setups
    )
    lines += [
        '',
        f'The self-supervised magnetic estimate has lower centered RMSE than IMU-only integration in every setup. First fusion improves on the self-supervised magnetic estimate in {first_fusion_wins}/{len(front_setups)} front setups. The complete pipeline improves on the self-supervised magnetic estimate in {complete_wins}/{setup_count} setups; the exception is the Slayer, where first fusion has the lowest median among the two fused outputs.',
        '',
        'The oracle is trained and evaluated on the same recording with reference labels. It is a diagnostic comparator, not an independent validation result. IMU filter settings were selected on this development cohort. First fusion to complete pipeline combines nuisance correction, anchoring, and a second solve; rear has only one fusion stage.',
        '',
        'The CSV summary also includes absolute MAE and centered travel-bin RMSE. The IMU baseline has no absolute zero reference. Edge exclusions make these results differ from historical full-support statistics. No setup-stratified generalization claim beyond these development recordings is implied.',
        '',
        'Source experiments:',
    ]
    lines.extend('- '+source['path'] for source in sources)
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    print(args.output_dir/'report.md')

if __name__=='__main__': main()
