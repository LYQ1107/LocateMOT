"""Publish small, immutable evidence from actually completed video scopes."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import RUN,load,save,sha_file,arm_directory
from tools.groundjev.eval_no_mot20.aggregate_metrics import aggregate,render_docs


def snapshot(run=RUN):
    report=aggregate(run,with_teta=True);render_docs(report,run)
    docs=ROOT/'docs/GROUNDJEV';jobs=[]
    for entry in load(run/'jobs.json')['jobs']:
        job=Path(entry['path'])
        if not (job/'cache_receipt.json').exists():continue
        manifest=load(job/'manifest.json');cache=load(job/'cache_receipt.json')
        result={k:entry[k] for k in ('id','dataset','task','sequence','split','frames') if k in entry}
        result['expression']=manifest.get('expression');result['cache']=cache
        result['detection']=load(job/'detection_metrics.json') if (job/'detection_metrics.json').exists() else None
        result['arms']={}
        for arm in ('simple','j1','coupled'):
            physical_arm=arm_directory(arm);path=job/physical_arm
            if not (path/'frozen_receipt.json').exists():
                if (path/'status.json').exists():result['arms'][arm]=load(path/'status.json')
                continue
            frozen=load(path/'frozen_receipt.json');timing=load(path/'result.json')
            metrics=load(path/'official_metrics.json')
            result['arms'][arm]={'status':frozen['status'],'artifact_directory':physical_arm,'input':frozen['input'],
                'prediction_sha256':frozen['prediction_sha256'],'metrics':metrics['metrics'],
                'metric_scale':metrics.get('metric_scale'),'official_frames':metrics['frames'],
                'GT_boxes':metrics.get('GT_boxes'),'base_novel':metrics.get('base_novel'),
                'G2_calls':frozen.get('G2_calls',frozen.get('G2_detector_calls')),
                'peak_decision_GPU_bytes':timing.get('peak_decision_gpu_memory_bytes',timing.get('peak_gpu_memory_bytes',0)),
                'timing':{k:timing.get(k) for k in ('association_only_seconds','association_seconds',
                    'association_replay_fps','feature_seconds','image_io_seconds','model_initialization_seconds',
                    'complete_video_wall_seconds','end_to_end_fps','resumed_completed_frames')},
                'official_receipt_path':str(path/'official_metrics.json')}
            comparison=job/f'comparison_simple_{physical_arm}.json'
            if comparison.exists():result['arms'][arm]['paired_attribution']=load(comparison)
        jobs.append(result)
    gpu_peaks={}
    telemetry=run/'gpu_telemetry.jsonl'
    if telemetry.exists():
        import json
        with telemetry.open() as handle:
            for line in handle:
                sample=json.loads(line);resident={}
                for process in sample.get('compute_processes',[]):
                    if process['owned_G2_service'] or process.get('owned_evaluation',False):
                        uuid=process['uuid'];resident[uuid]=resident.get(uuid,0)+process['memory_MiB']
                for uuid,memory in resident.items():gpu_peaks[uuid]=max(memory,gpu_peaks.get(uuid,0))
    payload={'status':'PARTIAL_COMPLETE_VIDEO_RESULTS_FULL_GOAL_INCOMPLETE',
        'created_utc':datetime.now(timezone.utc).isoformat(),'project':'GroundJEV',
        'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runtime_root':str(run),'engineering':load(docs/'EVAL_NO_MOT20_ENGINEERING_RECEIPT.json'),
        'aggregate':report,'completed_jobs':jobs,'MOT20_jobs':0,'new_training':False,
        'sampled_owned_resident_peak_MiB_by_GPU':gpu_peaks,
        'resource_measurement_scope':'5-second resident-memory samples: lower bound on instantaneous peaks; shared total-device utilization is not attributed to this task',
        'full_goal_complete':False,'remaining_registered_scope_preserved':True}
    save(docs/'EVAL_NO_MOT20_RESULTS_SNAPSHOT.json',payload)
    overlap=load(run/'train_overlap_audit.json')
    save(docs/'EVAL_NO_MOT20_SOURCE_AUDIT.json',{'status':overlap['status'],
        'G2_rows':overlap['G2_rows'],'G2_rows_sha256':overlap['G2_rows_sha256'],
        'J1_manifest_sha256':overlap['J1_manifest_sha256'],'MOT20_registered':overlap['MOT20_registered'],
        'cross_source_policy':overlap['cross_source_policy'],
        'full_audit_sha256':sha_file(run/'train_overlap_audit.json'),
        'full_audit_path':str(run/'train_overlap_audit.json'),
        'source_video_ids':overlap['G2_video_ids_by_source'],
        'registered_scope':load(run/'jobs.json')['summary']})
    if (run/'native_output_diagnosis.json').exists():
        save(docs/'EVAL_NO_MOT20_NATIVE_OUTPUT_DIAGNOSIS.json',load(run/'native_output_diagnosis.json'))
    return {'status':payload['status'],'complete_detection_jobs':len(jobs),
            'frozen_arms':{arm:sum(j['arms'].get(arm,{}).get('prediction_sha256') is not None for j in jobs)
                           for arm in ('simple','j1','coupled')}}


def main():
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RUN);args=parser.parse_args()
    print(json.dumps(snapshot(args.root)),flush=True)


if __name__=='__main__':main()
