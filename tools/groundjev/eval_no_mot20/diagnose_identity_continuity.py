"""Post-hoc native probability evidence at adjacent GT-matched observations.

GT is read only after immutable inference/scoring. No diagnostic modifies a
decision, candidate, model, threshold or official metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.groundjev.eval_no_mot20.common import load,save,merged_eval_manifest
from tools.groundjev.eval_no_mot20.evaluate_g2_detection import ground_truth,iou_matrix
from tools.groundjev.eval_no_mot20.replay_simple import audit_input


def diagnose(job,arm):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    frozen=load(job/arm/'frozen_receipt.json')
    if audit_input(job,job/arm)!=frozen['input']:raise ValueError('Frozen candidates differ')
    truth,_=ground_truth(merged_eval_manifest(job))
    def match(gt,boxes):
        matrix=iou_matrix([g['box'] for g in gt],boxes)
        if not matrix.size:return {}
        rows,cols=linear_sum_assignment(-(matrix>=.5).astype(float)-matrix*1e-6)
        return {int(gt[i]['identity']):int(j) for i,j in zip(rows,cols) if matrix[i,j]>=.5}
    previous={};counts=Counter();values=Counter();examples=[];image_counts=Counter()
    for line in (job/arm/'frame_decisions.jsonl').open():
        row=json.loads(line);gt=truth[row['frame']]
        candidate_truth=match(gt,[c['box'] for c in row['candidates']])
        output_truth=match(gt,[p['box'] for p in row['outputs']])
        metadata=row['state']['decision_metadata']
        for batch in metadata.get('batches',[]):
            image_counts[str(batch['metadata'].get('media',{}).get('images'))]+=1
        events={e['candidate']:e for e in row['state']['events'] if 'candidate' in e}
        masked=set(metadata.get('no_match_infeasible_candidate_indices') or [])
        for identity,index in candidate_truth.items():
            if identity not in previous:continue
            counts['adjacent_previous_output_and_current_candidate_GT_matches']+=1
            old_id=str(previous[identity]);event=events[index];p=event['decision']['probabilities']
            if old_id not in p:
                counts['previous_identity_not_in_current_options']+=1;continue
            counts['previous_identity_in_options']+=1
            existing={key:value for key,value in p.items() if key.isdigit()}
            best=max(existing,key=existing.get)
            feasible={key:value for key,value in p.items() if key!='NO_MATCH' or index not in masked}
            counts['raw_best_existing_preserves_previous_GT_matched_identity']+=best==old_id
            counts['raw_best_feasible_action_preserves_previous_identity']+=max(feasible,key=feasible.get)==old_id
            actual=event.get('track_id')
            counts['actual_same_identity']+=str(actual)==old_id
            counts['actual_different_identity']+=actual is not None and str(actual)!=old_id
            counts['actual_no_output_for_candidate']+=actual is None
            values['previous_identity_probability']+=p[old_id]
            values['best_other_identity_probability']+=max((v for key,v in existing.items() if key!=old_id),default=0.)
            if actual is not None and str(actual)!=old_id and len(examples)<8:
                examples.append({'frame':row['frame'],'candidate':index,'evaluation_GT_identity':identity,
                    'previous_output_id':int(old_id),'actual_output_id':actual,'raw_best_existing_id':int(best),
                    'previous_identity_probability':p[old_id],'actual_identity_probability':p[str(actual)],
                    'shortlist_ids':metadata.get('shortlist_ids')})
        previous={identity:int(row['outputs'][index]['track_id']) for identity,index in output_truth.items()}
    n=counts['previous_identity_in_options']
    return {'status':'POSTHOC_ADJACENT_IDENTITY_PROBABILITY_DIAGNOSTIC','job':str(job),'arm':arm,
        'input':frozen['input'],'official_metrics':load(job/arm/'official_metrics.json')['metrics'],
        'counts':dict(counts),'mean_native_probability':{key:value/n for key,value in values.items()} if n else {},
        'native_identity_request_image_counts':dict(image_counts),'switch_examples':examples,
        'GT_access':'post-inference diagnostic only','prediction_or_metrics_modified':False,
        'scope':'adjacent registered frames; GT/candidate and GT/output Hungarian IoU>=.5 matching. '
                'Not an official IDSW replacement or an independent causal decomposition.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--arm',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=diagnose(args.job,args.arm);save(args.out,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
