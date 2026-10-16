"""Audit whole-frame IGNORE labels against the actual native request groups."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def audit(unit,alternative_grouping=None):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    from tools.groundjev.analyze_tracking_events import load_truth
    from tools.groundjev.group_reference_rows import REFERENCE_KEY

    manifest=json.loads((ROOT/unit['manifest']).read_text())
    manifest['gt_path_for_evaluation_only']=manifest.get('gt_path_for_evaluation_only',manifest.get('gt_path'))
    truth,_=load_truth(manifest)
    folder=Path(unit['full_source_rows']).parent
    with gzip.open(folder/'groups.jsonl.gz','rt') as f:
        groups={g['id']:g for g in map(json.loads,f) if not g['id'].endswith('_assigned')}
    with gzip.open(folder/'candidate_registry.jsonl.gz','rt') as f:
        registry={(r['frame'],r['candidate']):r for r in map(json.loads,f)}
    targets={}
    with gzip.open(folder/'train_refs.jsonl.gz','rt') as f:
        for row in map(json.loads,f):
            if row['task']!='identity':continue
            ref=row[REFERENCE_KEY];key=(ref['group'],ref['question'])
            assert key not in targets and len(row['target'])==1
            targets[key]=next(iter(row['target']))
    counts=Counter();examples=[];seen_groups=set()
    with (ROOT/unit['original_run']/'frame_decisions.jsonl').open() as f:
        for line in f:
            row=json.loads(line);frame=row['frame'];candidates=row['candidates'];gt=truth[frame]
            assert frame==manifest['frame_numbers'][counts['frames']]
            assert row['image_path']==manifest['images'][counts['frames']]
            matrix=np.asarray([[overlap(c['box'],g['box']) for g in gt] for c in candidates]).reshape(len(candidates),len(gt))
            matched={}
            if candidates and gt:
                rr,cc=linear_sum_assignment(-matrix)
                matched={int(i):int(j) for i,j in zip(rr,cc) if matrix[i,j]>=.5}
            winners={j:i for i,j in matched.items()}
            group_for=[registry[frame,i]['ordinary_group'] for i in range(len(candidates))]
            alternative=None
            if alternative_grouping:
                from locatemot.groundjev.candidate_groups import candidate_group_indices
                new_groups=candidate_group_indices(candidates,16,alternative_grouping)
                assert sorted(i for group in new_groups for i in group)==list(range(len(candidates)))
                assert all(len(group)<=16 for group in new_groups)
                alternative={i:k for k,group in enumerate(new_groups) for i in group}
            local_offsets=Counter()
            for i,candidate in enumerate(candidates):
                record=registry[frame,i];identifier=group_for[i];group=groups[identifier]
                local=local_offsets[identifier];local_offsets[identifier]+=1;seen_groups.add(identifier)
                assert group['media'][0]['path']==row['image_path']
                assert group['state']['candidates'][local]['box']==candidate['box']
                assert group['state']['candidates'][local]['index']==local
                assert record['posthoc_GT_identity']==(gt[matched[i]]['identity'] if i in matched else None)
                target=targets[identifier,f'match_{local}']
                assert (target=='IGNORE')==(i not in matched)
                counts['original_registry_binding_and_IGNORE_target_exact']+=1
                if i in matched:
                    counts['matched_positive_candidates']+=1
                    rivals=[k for k in np.flatnonzero(matrix[:,matched[i]]>=.5) if k!=i]
                    counts['positive_with_any_GT_supported_rival']+=bool(rivals)
                    counts['positive_with_rival_outside_group']+=any(group_for[k]!=identifier for k in rivals)
                    if alternative is not None:
                        counts['alternative_positive_with_rival_outside_group']+=any(alternative[k]!=alternative[i] for k in rivals)
                    continue
                counts['IGNORE_candidates']+=1
                valid=list(map(int,np.flatnonzero(matrix[i]>=.5)))
                if not valid:
                    counts['IGNORE_without_GT_IoU_at_least_half']+=1;continue
                counts['IGNORE_with_GT_IoU_at_least_half']+=1
                counts['IGNORE_with_multiple_supported_GT']+=len(valid)>1
                inside=[winners[j] for j in valid if j in winners and group_for[winners[j]]==identifier]
                outside=[winners[j] for j in valid if j in winners and group_for[winners[j]]!=identifier]
                counts['GT_supported_IGNORE_with_winner_inside_group']+=bool(inside)
                counts['GT_supported_IGNORE_with_winner_outside_group']+=bool(outside)
                counts['GT_supported_IGNORE_with_no_assigned_winner']+=not inside and not outside
                if alternative is not None:
                    new_inside=[winners[j] for j in valid if j in winners and alternative[winners[j]]==alternative[i]]
                    new_outside=[winners[j] for j in valid if j in winners and alternative[winners[j]]!=alternative[i]]
                    counts['alternative_GT_supported_IGNORE_with_winner_inside_group']+=bool(new_inside)
                    counts['alternative_GT_supported_IGNORE_with_winner_outside_group']+=bool(new_outside)
                    counts['alternative_GT_supported_IGNORE_with_only_outside_winners']+=bool(new_outside) and not new_inside
                if outside and not inside:
                    counts['GT_supported_IGNORE_with_only_outside_winners']+=1
                    if len(valid)==1:counts['single_GT_IGNORE_with_only_outside_winner']+=1
                    if len(examples)<3:
                        k=outside[0]
                        examples.append(dict(frame=frame,candidate=i,group=identifier,box=candidate['box'],
                            target=target,original_winning_candidate=k,winner_group=group_for[k],
                            winner_box=candidates[k]['box'],supported_GT_count=len(valid),
                            full_frame_image_still_visible=True))
            for identifier,size in local_offsets.items():
                assert size==len(groups[identifier]['state']['candidates'])
                counts['request_groups']+=1
                counts['largest_request_candidates']=max(counts['largest_request_candidates'],size)
            counts['frames']+=1;counts['candidates']+=len(candidates)
    assert counts['frames']==len(manifest['images'])
    assert counts['candidates']==len(registry)==len(targets)
    return dict(sequence=manifest['sequence'],counts=dict(counts),examples=examples,
        empty_request_groups_without_identity_questions=len(groups)-len(seen_groups))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--prior-audit',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--alternative-grouping',choices=['overlap_neighbors'])
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    index=json.loads(args.index.read_text());prior=json.loads(args.prior_audit.read_text());units={};total=Counter()
    for name,unit in index['units'].items():
        if unit['task']!='mot':continue
        result=audit(unit,args.alternative_grouping);units[name]=result
        largest=max(total['largest_request_candidates'],result['counts']['largest_request_candidates'])
        total.update(result['counts']);total['largest_request_candidates']=largest
    expected=prior['train_totals']
    assert len(units)==43 and total['frames']==expected['frames']==1056
    assert total['candidates']==expected['candidates']==48051
    assert total['IGNORE_candidates']==expected['unmatched_candidates']==7992
    assert total['IGNORE_with_GT_IoU_at_least_half']==expected['unmatched_with_any_GT_IoU_at_least_half']==5832
    report=dict(status='ALL43_TRAIN_GLOBAL_IGNORE_LABELS_AND_ACTUAL_GROUP_COMPETITION_AUDITED',
        alternative_grouping=args.alternative_grouping,
        index=str(args.index),prior_audit=str(args.prior_audit),train_totals=dict(total),units=units,
        interpretation='GT-supported IGNORE is a whole-frame one-to-one labeling outcome, not background truth. Outside-group means outside the explicitly enumerated candidate list; full-frame image pixels are still visible, so this is not proof labels are unobservable or training must fail. Multiple-GT overlaps remain explicitly counted. TRAIN only; no held-out threshold tuning or new causal tracking scores.',
        original_training_targets_inputs_or_predictions_changed=False,new_GPU_or_optimizer_updates=0,
        new_ROI_or_image_bank_bytes=0,metadata_suballocation='At most128KiB from existing2MiB NO_MATCH audit/control reservation.',
        combined_registered_feature_upper_bytes=26829071326,global_temporary_feature_budget_bytes=26843545600,
        full_goal_complete=False)
    content=json.dumps(report,indent=2)+'\n';assert len(content.encode())<=128*1024
    (args.out/'summary.json').write_text(content)
    print(json.dumps(dict(status=report['status'],train_totals=dict(total))),flush=True)


if __name__=='__main__':main()
