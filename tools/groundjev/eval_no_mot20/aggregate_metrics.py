"""Truthful partial/final tables; official MOT combination, never mean HOTA."""
from __future__ import annotations

import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save,G2_SHA,TRAIN_SHA


def combine_mot(paths):
    import numpy as np
    from locatemot.unified.runtime.u2_ordinary_evaluation import verify_trackeval_source
    verify_trackeval_source()
    import trackeval
    components={str(p.parent):load(p) for p in paths}
    metrics=[trackeval.metrics.HOTA(),trackeval.metrics.CLEAR({'PRINT_CONFIG':False}),trackeval.metrics.Identity({'PRINT_CONFIG':False})]
    combined={}
    for metric in metrics:
        rows={name:{k:np.asarray(v) if isinstance(v,list) else v for k,v in row[metric.get_name()].items()}
              for name,row in components.items()}
        combined[metric.get_name()]=metric.combine_sequences(rows)
    hota=combined['HOTA'];clear=combined['CLEAR'];identity=combined['Identity']
    return {**{k:float(np.mean(hota[k])) for k in ('HOTA','DetA','AssA')},
            'IDF1':float(identity['IDF1']),'IDSW':int(clear['IDSW']),
            'MOTA':float(clear['MOTA']),'FP':int(clear['CLR_FP']),'FN':int(clear['CLR_FN'])}


def aggregate(run=RUN):
    registry=load(run/'jobs.json');groups=defaultdict(list)
    for job in registry['jobs']:groups[job['dataset']].append(job)
    report={'status':'INCOMPLETE','created_utc_seconds':time.time(),'G2_step':18227,'G2_SHA':G2_SHA,
            'TRAIN_manifest_SHA':TRAIN_SHA,'MOT20_jobs':0,'datasets':{},
            'full_goal_complete':False,'training_on_validation':False}
    for dataset,jobs in groups.items():
        detection=[];caches=[];arm_rows=defaultdict(list);completed_frames=0;measured=[];requests=0
        for job in jobs:
            path=Path(job['path'])
            if (path/'cache_receipt.json').exists():
                cache=load(path/'cache_receipt.json');caches.append(cache)
                completed_frames+=cache['frames'];requests+=cache['requests']
            elif (path/'detector.sqlite').exists():
                db=sqlite3.connect(f'file:{path}/detector.sqlite?mode=ro',uri=True,timeout=30)
                try:
                    completed_frames+=db.execute('SELECT COUNT(*) FROM frames').fetchone()[0]
                    requests+=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
                finally:db.close()
            if (path/'detection_metrics.json').exists():detection.append(load(path/'detection_metrics.json'))
            for arm in ('simple','j1','coupled'):
                p=path/arm/'official_metric_components.json'
                if p.exists() and (path/arm/'frozen_receipt.json').exists():arm_rows[arm].append(p)
        # Estimate from actual cached native request/frame records. This is a
        # dataset-specific measurement, not a borrowed MOT20 speed.
        for worker in (run/'workers').glob('*/preflight.json'):
            p=load(worker)
            if p['dataset']!=dataset:continue
            from tools.groundjev.eval_no_mot20.cache_store import CacheStore
            store=CacheStore(Path(p['job']))
            try:
                for frame in store.manifest['frame_numbers'][:p['inputs']]:
                    row=store.frame(frame)
                    if row:measured.append(row['seconds'])
            finally:store.close()
        total_frames=sum(j['frames'] for j in jobs);gt=sum(d['GT_boxes'] for d in detection)
        value={'scope':'OFFICIAL_FULL_VAL' if len(detection)==len(jobs) and dataset=='sportsmot' else
                       'REGISTERED_COMPLETE_VIDEO' if detection else 'INCOMPLETE',
               'registered_videos':len({j['sequence'] for j in jobs}),'registered_jobs':len(jobs),
               'registered_query_frames':total_frames,'planned_DLM_requests':sum(j['planned_requests'] for j in jobs),
               'completed_detection_frames':completed_frames,'persisted_requests':requests,
               'complete_detection_jobs':len(caches),'scored_detection_jobs':len(detection),
               'scored_frames':sum(d['frames'] for d in detection),'GT_boxes':gt,
               'Recall@0.50':sum(d['covered50'] for d in detection)/gt if gt else None,
               'Recall@0.75':sum(d['covered75'] for d in detection)/gt if gt else None,
               'rejected_complete_jobs':sum(c['rejected'] for c in caches),
               'preflight_inputs':len(measured),
               'measured_preflight_FPS':len(measured)/sum(measured) if measured and sum(measured) else None,
               'remaining_detection_GPU_hours_estimate':
                   (total_frames-completed_frames)*sum(measured)/len(measured)/3600 if measured else None,
               'tracking':{}}
        for arm,paths in arm_rows.items():
            value['tracking'][arm]={'official_combined_metrics':combine_mot(paths),
                'complete_jobs':len(paths),'aggregation':'unmodified TrackEval metric.combine_sequences',
                'scope':'REGISTERED_COMPLETE_VIDEO','job_ids':[p.parent.parent.name for p in paths]}
        if dataset=='tao_amodal':
            value['tracking_note']='TETA must be recomputed jointly across completed videos; per-video scores are not averaged'
        report['datasets'][dataset]=value
    report['exclusion_reasons']=dict(Counter(e['reason'] for e in registry['excluded']))
    save(run/'aggregate_metrics.json',report)
    return report


def render_docs(report,run):
    docs=ROOT/'docs/GROUNDJEV';docs.mkdir(exist_ok=True)
    rows=['# Non-MOT20 frozen data scope','',f'Output root: `{run}`. Model: G2 final step18227 `{G2_SHA}`.','',
          '| Dataset | Videos | Jobs/queries | Query-frames | DLM requests |','|---|---:|---:|---:|---:|']
    for name,d in report['datasets'].items():rows.append(f"| {name} | {d['registered_videos']} | {d['registered_jobs']} | {d['registered_query_frames']} | {d['planned_DLM_requests']} |")
    rows+=['','TAO uses all 1,203 fixed LVIS-v1 labels in 13 blocks. Only annotated frames are inferred; no complete physical-video FPS claim.','',
           'Every job has a frozen detector manifest and a separate evaluation-only GT manifest. No MOT20 job is registered.','',
           'MOT17 labeled videos and all 25 local DanceTrack val videos overlap G2 TRAIN, including RMOT26 cross-dataset copies. They are excluded from independent evaluation. AerialMind available videos all occur in G2 dense TRAIN. RMOT26 local test files do not supply a complete, disjoint frame/identity-GT release. These are explicit coverage limits.','',
           'The full row audit and immutable manifests are stored locally as `train_overlap_audit.json`, `jobs.json`, and `jobs/*/manifest.json`; no labels or large caches are published.']
    (docs/'EVAL_NO_MOT20_DATA_SCOPE.md').write_text('\n'.join(rows)+'\n')
    rows=['# Measured runtime and remaining detection cost','',
          'Status: INCOMPLETE. Estimates use actual input-specific native request times, including all TAO vocabulary blocks. They are provisional and exclude future JEV cost.','',
          '| Dataset | Cached frames / registered | Requests saved | Measured inputs | FPS | Remaining GPU-hours |',
          '|---|---:|---:|---:|---:|---:|']
    def number(v):return 'PENDING' if v is None else f'{v:.3f}'
    for name,d in report['datasets'].items():rows.append(f"| {name} | {d['completed_detection_frames']}/{d['registered_query_frames']} | {d['persisted_requests']} | {d['preflight_inputs']} | {number(d['measured_preflight_FPS'])} | {number(d['remaining_detection_GPU_hours_estimate'])} |")
    rows+=['','Workers: four shared A800 GPUs, physical 3/4/6/1, UUIDs and memory limits in the committed configuration. A service runs one native request at a time. Allocation time and sampled utilization are recorded separately. No fourfold speedup is assumed.','',
           'Association replay FPS is separate from detector throughput. A sum of measured stage costs is an estimate, never a directly measured end-to-end FPS.']
    (docs/'EVAL_NO_MOT20_RUNTIME_ESTIMATE.md').write_text('\n'.join(rows)+'\n')
    rows=['# G2 perception results','', 'Status: INCOMPLETE. Only officially scored complete registered videos/expressions appear below.','',
          '| Dataset | Complete jobs | Scored frames | GT boxes | Recall@.50 | Recall@.75 | Rejected (complete caches) |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name,d in report['datasets'].items():rows.append(f"| {name} | {d['scored_detection_jobs']} | {d['scored_frames']} | {d['GT_boxes']} | {number(d['Recall@0.50'])} | {number(d['Recall@0.75'])} | {d['rejected_complete_jobs']} |")
    rows+=['','AP50/AP75/mAP: N/A. GroundAnything supplies no genuine per-box confidence. No constant-score ranking AP is reported. TAO is non-exhaustive: unmatched proposals are not all verified false positives. Full per-frame coverage, groups, IoU and empty-query diagnostics are in each local `detection_metrics.json`.']
    (docs/'EVAL_NO_MOT20_G2_DETECTION.md').write_text('\n'.join(rows)+'\n')
    for filename,arms in [('EVAL_NO_MOT20_SIMPLE_RESULTS.md',('simple',)),('EVAL_NO_MOT20_JEV_RESULTS.md',('j1','coupled'))]:
        rows=['# Fixed final-G2 candidate tracking results','', 'Status: INCOMPLETE. Values are fractions except counts.','',
              '| Dataset | Arm | Complete jobs | HOTA | DetA | AssA | IDF1 | IDSW |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
        for name,d in report['datasets'].items():
            for arm in arms:
                if arm not in d['tracking']:continue
                t=d['tracking'][arm];m=t['official_combined_metrics']
                rows.append(f"| {name} | {arm} | {t['complete_jobs']} | {number(m['HOTA'])} | {number(m['DetA'])} | {number(m['AssA'])} | {number(m['IDF1'])} | {m['IDSW']} |")
        rows+=['','No unweighted mean of per-video HOTA is used. Only paired complete scopes support net-gain claims. OVMOT requires official TETA, not MOT HOTA.','',
               'Coupled JEV currently supports ordinary MOT only. RMOT/OVMOT: `NOT_SUPPORTED_BY_CURRENT_CHECKPOINT`. Its selected checkpoint was trained on base-G0 cached features; evaluation on the final-G2 projector is an explicitly recorded feature-distribution shift. J1 step1200 and the separate step8227 OneJev model are not interchanged.']
        (docs/filename).write_text('\n'.join(rows)+'\n')
    (docs/'EVAL_NO_MOT20_FINAL_REPORT.md').write_text(
        '# Non-MOT20 evaluation status\n\nINCOMPLETE — the registered scope is not finished.\n\n'
        'No training was started. G2, J1 and Coupled checkpoints remain immutable. '
        'See the scope, runtime, perception, Simple and JEV reports for completed work and explicit limits. '
        'A future training decision requires paired full-video evidence; code tests and preflight throughput are not performance gains.\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RUN)
    parser.add_argument('--write-docs',action='store_true');args=parser.parse_args()
    report=aggregate(args.root)
    if args.write_docs:render_docs(report,args.root)
    print(json.dumps({name:{k:d[k] for k in ('completed_detection_frames','complete_detection_jobs','scored_detection_jobs','preflight_inputs')}
                      for name,d in report['datasets'].items()}),flush=True)


if __name__=='__main__':main()
