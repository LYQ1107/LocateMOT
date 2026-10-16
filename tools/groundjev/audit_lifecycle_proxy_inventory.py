"""Read immutable curriculum targets; do not render media or alter supervision."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def rows(path,bucket=None):
    path=ROOT/path
    if str(path).endswith('.sources.json'):
        manifest=json.loads(path.read_text())
        for source in manifest['sources']:
            if bucket is not None and source['bucket']!=bucket:continue
            eligible=(r for r in rows(Path(source['path'])) if source['families'] is None or r['task'] in source['families'])
            if 'row_indices' in source:
                eligible=list(eligible);assert len(eligible)==source['source_rows']
                selected=[eligible[i] for i in source['row_indices']]
                assert len(selected)==source['rows'];yield from selected
            else:
                count=0
                for row in eligible:count+=1;yield row
                assert count==source['rows']
    else:
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt') as f:
            for line in f:yield json.loads(line)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    sources={
        'full_epoch_MOT_sources':('outputs/groundjev/training_data/full_joint_source_manifest_v1/train.sources.json','mot'),
        'balanced6000_24000':('outputs/groundjev/training_data/broader_shared_onejev6000_v2_fla/train.jsonl.gz',None),
        'paired_short_24000':('outputs/groundjev/training_data/paired_temporal_teacher24000_v2_indexed/short/train.sources.json',None),
        'paired_long_24000':('outputs/groundjev/training_data/paired_temporal_teacher24000_v2_indexed/long/train.sources.json',None)}
    results={}
    for name,(path,bucket) in sources.items():
        labels=Counter();families=Counter()
        for row in rows(Path(path),bucket):
            families[row['task']]+=1
            if row['task']=='lifecycle':
                assert len(row['target'])==1
                label,mass=next(iter(row['target'].items()))
                assert label in ('ACTIVE','LOST','REACTIVATE','TERMINATE') and mass==1
                labels[label]+=1
        results[name]=dict(source=path,selected_bucket=bucket,rows=sum(families.values()),families=dict(families),
            lifecycle_targets=dict(labels),terminate_fraction=labels['TERMINATE']/sum(labels.values()))
    assert results['full_epoch_MOT_sources']['rows']==252258
    assert results['full_epoch_MOT_sources']['families']['lifecycle']==93624
    assert all(results[key]['rows']==24000 for key in sources if key!='full_epoch_MOT_sources')
    assert results['paired_short_24000']['lifecycle_targets']==results['paired_long_24000']['lifecycle_targets']
    report=dict(status='IMMUTABLE_NATIVE_LIFECYCLE_TARGET_INVENTORY_COMPLETE',sources=results,
        exporter_contract='export_broader_native_full_groups.py assigns TERMINATE iff its external birth GT binding is None. That proxy is not independent evidence an object permanently left the video; unknown remains unknown. Inventory counts literal row targets, not a new per-row GT validity certification or a causal demonstration of model failure.',
        progress_scope='Curriculum target inventory only. Full epoch and paired stages remain incomplete; this is not a claim every listed row has already been consumed.',
        original_rows_targets_or_training_jobs_changed=False,new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
