"""Posthoc fixed-birth diagnostic of original learned identity/action probabilities."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    results={}
    actions={'NEW','IGNORE','NO_MATCH'}
    for scope,job in plan['scopes'].items():
        source=Path(plan['reference_root'])/(scope+'_before')
        diagnostic=json.loads((source/'association_events.json').read_text())
        offline=iter(diagnostic['events']);counts=Counter();examples=[]
        groups={};maximum_error=0.
        with (source/'frame_decisions.jsonl').open() as f:
            for line in f:
                row=json.loads(line);counts['frames']+=1
                assert row['state']['decision_metadata']['native_probabilities']
                for event in row['state']['events']:
                    if 'candidate' not in event:continue
                    target=next(offline)
                    assert (target['frame'],target['candidate'])==(row['frame'],event['candidate'])
                    counts['candidates']+=1
                    probabilities=event['decision']['probabilities']
                    assert all(math.isfinite(p) and p>=0 for p in probabilities.values())
                    existing={key:p for key,p in probabilities.items() if key not in actions}
                    mass=sum(existing.values());largest=max(existing.values(),default=0.)
                    best_action=max((p for key,p in probabilities.items() if key in actions),default=0.)
                    maximum_error=max(maximum_error,abs(sum(probabilities.values())-1))
                    if target['matched_gt_identity'] is None:
                        counts['GT_unmatched_candidates']+=1;continue
                    counts['GT_matched_candidates']+=1
                    correct={str(tid) for tid in target['correct_surviving_anchor_ids']}&existing.keys()
                    if not correct:
                        counts['GT_matched_without_correct_shortlisted_birth_anchor']+=1;continue
                    counts['correct_birth_option_present']+=1
                    bucket='one_correct_option' if len(correct)==1 else 'multiple_correct_options'
                    group=groups.setdefault(bucket,Counter());group['observations']+=1
                    diluted=mass>best_action and largest<best_action
                    selected=str(event.get('track_id')) in correct
                    raw_choice=max(probabilities,key=probabilities.get)
                    group['original_raw_choice_is_correct_ID']+=raw_choice in correct
                    group['original_global_assignment_is_correct_ID']+=selected
                    group['existing_mass_exceeds_best_action']+=mass>best_action
                    group['all_existing_IDs_below_best_action']+=largest<best_action
                    group['existing_mass_diluted_below_best_action']+=diluted
                    group['diluted_with_current_correct_existing_rank']+=diluted and max(existing,key=existing.get) in correct
                    group['diluted_original_global_assignment_is_wrong']+=diluted and not selected
                    group['non_diluted_existing_mass_cannot_beat_best_action']+=mass<=best_action
                    if diluted and len(examples)<8:
                        examples.append(dict(frame=row['frame'],candidate=event['candidate'],correct_options=sorted(correct),
                            existing_options=len(existing),existing_mass=mass,max_existing=largest,best_action_probability=best_action,
                            raw_choice=raw_choice,actual_action=event['action'],actual_assigned_id=event.get('track_id')))
        assert next(offline,None) is None
        assert counts['frames']==job['frames'] and counts['candidates']==job['candidates_count']
        assert counts['GT_matched_candidates']+counts['GT_unmatched_candidates']==counts['candidates']
        results[scope]=dict(source=str(source),counts=dict(counts),positive_groups={k:dict(v) for k,v in groups.items()},
                            input_probability_mass_max_error=maximum_error,examples=examples)
    report=dict(status='ORIGINAL_LEARNED_EXISTING_MASS_DILUTION_POSTHOC_AUDIT_COMPLETE',scopes=results,
        criterion='Correct immutable birth-ID option exists in the actual shortlist; old sumP(existing)>maxP(action), but every single existing-ID probability is below that action.',
        scope='All five fixed original learned trajectories. GT/birth alignment from original external diagnostic only; no GT enters inference. Potential is an oracle bound, not proof appearance selects a correct identity, global assignment recovery, HOTA improvement or calibrated posteriors.',
        new_GPU_or_optimizer_updates=0,original_predictions_or_protocol_changed=False,full_goal_complete=False)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],scopes={name:dict(counts=r['counts'],positive_groups=r['positive_groups']) for name,r in results.items()})))


if __name__=='__main__':main()
