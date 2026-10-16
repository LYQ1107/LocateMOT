"""Audit GT-supported actual births omitted by all-proposal target assignment."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def audit(manifest,run,registry=None):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    from tools.groundjev.analyze_tracking_events import load_truth
    manifest=dict(manifest)
    if 'gt_path_for_evaluation_only' not in manifest:manifest['gt_path_for_evaluation_only']=manifest['gt_path']
    truth,_=load_truth(manifest);counts=Counter();examples=[]
    if registry:
        with gzip.open(registry,'rt') as file:
            original={(r['frame'],r['candidate']):r for r in map(json.loads,file)}
    else:original=None
    with (run/'frame_decisions.jsonl').open() as file:
        for frame_index,line in enumerate(file):
            row=json.loads(line);frame=row['frame'];assert frame==manifest['frame_numbers'][frame_index]
            assert row['image_path']==manifest['images'][frame_index]
            candidates=row['candidates'];gt=truth[frame];n,m=len(candidates),len(gt)
            matrix=np.asarray([[overlap(c['box'],g['box']) for g in gt] for c in candidates]).reshape(n,m)
            matched={}
            if n and m:
                rr,cc=linear_sum_assignment(-matrix)
                matched={int(i):int(j) for i,j in zip(rr,cc) if matrix[i,j]>=.5}
            if original is not None:
                for i in range(n):
                    old=original[frame,i]['posthoc_GT_identity']
                    assert old==(gt[matched[i]]['identity'] if i in matched else None)
                    counts['original_candidate_registry_targets_exact']+=1
            valid=matrix>=.5
            events={e['candidate']:e for e in row['state']['events'] if 'candidate' in e}
            emitted=[i for i,e in events.items() if e['action'] in ('NEW','ACTIVE','REACTIVATE')]
            assert len(emitted)==len(row['outputs'])
            counts['frames']+=1;counts['candidates']+=n;counts['original_matched_candidates']+=len(matched)
            for i in range(n):
                if i not in matched:
                    counts['unmatched_candidates']+=1
                    counts['unmatched_with_any_GT_IoU_at_least_half']+=bool(valid[i].any())
                columns=np.flatnonzero(valid[i])
                if len(columns)==1 and int(valid[:,columns[0]].sum())==1 and i not in matched:
                    counts['isolated_valid_edges_lost_by_assignment_before_threshold']+=1
            for i in emitted:
                counts['actual_emissions']+=1
                if i in matched:counts['actual_emissions_bound_by_original_assignment']+=1
                event=events[i]
                if event['action']=='NEW':
                    counts['actual_NEW_births']+=1
                    if i not in matched:counts['NEW_births_without_original_GT_binding']+=1
                columns=np.flatnonzero(valid[i])
                if len(columns)!=1:continue
                j=int(columns[0])
                if int(valid[emitted,j].sum())!=1:continue
                counts['unique_bipartite_valid_emissions']+=1
                if i in matched:continue
                counts['unique_GT_supported_emissions_omitted_by_original_assignment']+=1
                other=next((k for k,v in matched.items() if v==j),None)
                assert other not in emitted
                counts['omitted_GT_assigned_to_unemitted_candidate']+=other is not None
                counts['omitted_GT_unassigned_after_global_threshold']+=other is None
                if event['action']=='NEW':counts['unique_GT_supported_NEW_births_left_unbound']+=1
                if len(examples)<50:
                    examples.append(dict(sequence=manifest['sequence'],frame=frame,image_path=row['image_path'],
                        candidate=i,box=candidates[i]['box'],actual_action=event['action'],actual_ID=event.get('track_id'),
                        unique_supported_GT_ID=gt[j]['identity'],GT_box=gt[j]['box'],IoU=float(matrix[i,j]),
                        original_assignment_GT_ID=None,competing_unemitted_candidate=other,
                        competing_box=candidates[other]['box'] if other is not None else None,
                        competing_IoU=float(matrix[other,j]) if other is not None else None))
    assert counts['frames']==len(manifest['images'])
    if original is not None:assert counts['original_candidate_registry_targets_exact']==len(original)
    return dict(sequence=manifest['sequence'],counts=dict(counts),examples=examples)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    index=json.loads((ROOT/plan['train_index']).read_text());train=[];total=Counter()
    for unit,info in index['units'].items():
        if not unit.startswith('mot/'):continue
        manifest=json.loads((ROOT/info['manifest']).read_text())
        result=audit(manifest,ROOT/info['original_run'],Path(info['full_source_rows']).parent/'candidate_registry.jsonl.gz')
        train.append(result);total.update(result['counts'])
        print(json.dumps(dict(status='AUDITING_ORIGINAL_TRAIN_LABEL_SUPPORT',completed_units=len(train),frames=total['frames'])),flush=True)
    assert len(train)==plan['expected_train_units'] and total['frames']==plan['expected_train_frames']
    assert total['candidates']==plan['expected_train_candidates']
    validation=audit(json.loads((ROOT/plan['validation_manifest']).read_text()),ROOT/plan['validation_run'])
    assert validation['counts']['frames']==plan['expected_validation_frames']
    assert validation['counts']['candidates']==plan['expected_validation_candidates']
    result=dict(status='ORIGINAL_MOT_GT_BIRTH_SUPPORT_AUDIT_COMPLETE',train_totals=dict(total),
        train_units=train,validation=validation,all_original_TRAIN_registry_candidate_targets_exact=True,
        scope=plan['audit'],source_labels_or_model_states_changed=False,training_or_metric_protocol_changed=False,
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<2*2**20
    compact={k:v for k,v in result.items() if k not in ('train_units','validation')}
    print(json.dumps(dict(compact,validation_counts=validation['counts'])),flush=True)


if __name__=='__main__':main()
