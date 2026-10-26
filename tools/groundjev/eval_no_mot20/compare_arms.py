"""Post-hoc paired association attribution. GT never enters runtime decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import load,save,merged_eval_manifest
from tools.groundjev.eval_no_mot20.evaluate_g2_detection import ground_truth,iou_matrix
from tools.groundjev.eval_no_mot20.replay_simple import audit_input


def transitions(job,arm):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    manifest=merged_eval_manifest(job);truth,_=ground_truth(manifest)
    rows=[json.loads(l) for l in (job/arm/'frame_decisions.jsonl').open()]
    previous={};track_truth={};seen_tracks=set();edges={}
    wrong_births=unmatched_births=wrong_writes=occlusion_cases=occlusion_restored=0
    for position,row in enumerate(rows):
        gt=truth[row['frame']];pred=row['outputs'];matrix=iou_matrix([g['box'] for g in gt],[p['box'] for p in pred])
        matches={};prediction_truth={}
        if matrix.size:
            rr,cc=linear_sum_assignment(-(matrix>=.5).astype(float)-matrix*1e-6)
            for i,j in zip(rr,cc):
                if matrix[i,j]>=.5:
                    identity=int(gt[i]['identity']);tid=int(pred[j]['track_id'])
                    matches[identity]=tid;prediction_truth[tid]=identity
        for target in gt:
            identity=int(target['identity']);tid=matches.get(identity)
            if identity in previous:
                old_tid,old_position=previous[identity]
                correct=tid is not None and old_tid is not None and tid==old_tid
                edges[(row['frame'],identity)]={'correct':correct,'missing_current':tid is None,
                                                'gap_registered_frames':position-old_position}
                if position-old_position>1:
                    occlusion_cases+=1;occlusion_restored+=int(correct)
                if tid is not None and tid not in seen_tracks:wrong_births+=1
            previous[identity]=(tid,position)
        for prediction in pred:
            tid=int(prediction['track_id'])
            if tid not in seen_tracks and tid not in prediction_truth:unmatched_births+=1
        # Count only explicitly logged writes, and only contradictions to the
        # track's earlier GT attribution. No missing log is guessed as UPDATE.
        for event in row.get('state',{}).get('events',[]):
            tid=event.get('track_id')
            if event.get('memory_action')=='UPDATE' and tid in prediction_truth and tid in track_truth:
                wrong_writes+=int(track_truth[tid]!=prediction_truth[tid])
        track_truth.update(prediction_truth);seen_tracks.update(int(p['track_id']) for p in pred)
    return edges,{'new_id_on_previously_observed_GT':wrong_births,'unmatched_new_ids':unmatched_births,
                 'explicit_memory_writes_with_GT_identity_contradiction':wrong_writes if arm!='coupled' else None,
                 'occlusion_cases':occlusion_cases,'occlusion_restored_same_predicted_id':occlusion_restored,
                 'memory_write_diagnostic_supported':arm!='coupled',
                 'diagnostic_scope':'post-hoc IoU>=.5 matched transitions; missing detections count as unsuccessful; not official IDSW replacement'}


def compare(job,arm):
    if not (job/'simple/frozen_receipt.json').exists() or not (job/arm/'frozen_receipt.json').exists():
        raise ValueError('Both complete, officially scored arms are required')
    first=audit_input(job,job/'simple');second=audit_input(job,job/arm)
    if first!=second:raise ValueError('Paired arms changed native input')
    a,aa=transitions(job,'simple');b,bb=transitions(job,arm)
    if set(a)!=set(b):raise ValueError('GT transition scopes differ')
    corrected=sum(not a[k]['correct'] and b[k]['correct'] for k in a)
    harmed=sum(a[k]['correct'] and not b[k]['correct'] for k in a)
    left=load(job/'simple/official_metrics.json')['metrics'];right=load(job/arm/'official_metrics.json')['metrics']
    delta={k:right[k]-left[k] for k in ('HOTA','TETA','AssA','AssocA','IDF1','IDSW') if k in left and k in right}
    result={'status':'PAIRED_COMPLETE_SCOPE_COMPARISON','baseline':'simple','arm':arm,'input':first,
            'official_metric_delta':delta,'Corrected':corrected,'Harmed':harmed,'transition_count':len(a),
            'simple_diagnostics':aa,'arm_diagnostics':bb,'GT_access':'post-inference evaluation only'}
    save(job/f'comparison_simple_{arm}.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--arm',choices=['j1','coupled'],required=True);args=parser.parse_args()
    print(json.dumps(compare(args.job,args.arm)),flush=True)


if __name__=='__main__':main()
