"""Posthoc entity-level birth/match/duplicate/retrieval/write diagnostics."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def analyze(manifest,run):
    from tools.groundjev.analyze_tracking_events import load_truth
    from tools.groundjev.audit_global_identity_labels import frame_support
    truth,_=load_truth(manifest);counts=Counter();anchors={};prior=set();ever=set();last_candidate={};last_correct={}
    examples=[];frames=0
    with (run/'frame_decisions.jsonl').open() as file:
        for row in map(json.loads,file):
            frame=row['frame'];gt=truth[frame];matrix,matched,support=frame_support(row['candidates'],gt)
            unique=[gt[s[0]]['identity'] if len(s)==1 else None for s in support]
            before=dict(anchors);before_ever=set(ever);by_gt=defaultdict(list)
            for i,g in enumerate(unique):
                if g is not None:by_gt[g].append(i)
            events={e['candidate']:e for e in row['state']['events'] if 'candidate' in e};accepted=defaultdict(list)
            birth_eligible={g for g in by_gt if g not in ever and not any(before.get(t) is None for t in prior)}
            existing_gt={g for g in by_gt if any(before.get(t)==g for t in prior)}
            true_birth=set();true_match=set()
            for i,event in events.items():
                identity=unique[i];tid=event.get('track_id');decision=event['decision']
                existing=[t for t in prior if identity is not None and before.get(t)==identity]
                if identity is not None and tid is not None:accepted[identity].append(i)
                if identity is not None and identity in before_ever:
                    counts['retrieval_previously_confirmed_candidate_denominator']+=1
                    counts['retrieval_all_live_correct_covered']+=bool(existing)
                    counts['retrieval_initial_Top32_correct_covered']+=bool(set(existing)&set(decision.get('initial_retrieved_top32_ids',decision.get('shortlisted_ids',[]))))
                    counts['retrieval_all_live_conditional_denominator']+=bool(existing)
                raw=decision.get('pre_competition_probabilities',decision['probabilities'])
                if existing:
                    top=max(raw,key=raw.get);correct={str(t) for t in existing};actual_correct=tid in existing
                    counts['known_existing_candidate_denominator']+=1
                    counts['existing_rank_above_NEW']+=max(raw.get(t,0.) for t in correct)>raw['NEW']
                    counts['wrong_to_right_after_global_allocation']+=top not in correct and actual_correct
                    counts['right_to_wrong_after_global_allocation']+=top in correct and not actual_correct
                if event['action']=='NEW':
                    if identity in birth_eligible and identity not in true_birth:
                        counts['NEW_known_TP']+=1;true_birth.add(identity)
                    elif identity is not None and (existing or identity in true_birth):
                        counts['NEW_known_FP']+=1
                        counts['existing_live_identity_wrongly_NEW']+=bool(existing)
                        counts['same_frame_duplicate_NEW']+=identity in true_birth
                    else:counts['NEW_unknown_or_retired_reappearance']+=1
                    if identity is not None:anchors[tid]=identity;ever.add(identity)
                    else:anchors[tid]=None
                elif tid is not None:
                    if identity is not None and before.get(tid) is not None:
                        if before[tid]==identity:counts['MATCH_known_TP']+=1;true_match.add(identity)
                        else:
                            counts['MATCH_known_FP']+=1
                            counts['newborn_wrongly_MATCHed']+=identity in birth_eligible
                    else:counts['MATCH_unknown_binding_or_annotation']+=1
                else:
                    reason=decision.get('admission_action','REJECT')
                    counts['reject:'+reason]+=1
                    counts['uniquely_GT_supported_entity_REJECT']+=identity is not None
                    counts['known_existing_identity_REJECT']+=bool(existing)
                    if identity is None:counts['REJECT_annotation_unknown_not_background_truth']+=1
                if 'memory_action' in event and event['action']!='NEW':
                    counts['assigned_existing_observations']+=1
                    if identity is not None and before.get(tid) is not None:
                        wrong=before[tid]!=identity
                        counts['known_existing_write_eligibility_denominator']+=1
                        counts['wrong_recent_observational_writes']+=wrong
                        if event['memory_action']=='UPDATE':
                            counts['known_long_UPDATE_denominator']+=1;counts['wrong_long_memory_writes']+=wrong
                            counts['low_visibility_long_writes']+=gt[support[i][0]].get('visibility') is not None and gt[support[i][0]]['visibility']<.6
                        else:counts['wrong_binding_long_HOLD']+=wrong
                        if wrong and len(examples)<6:
                            examples.append(dict(frame=frame,candidate=i,assigned_ID=tid,birth_GT=before[tid],current_GT=identity,
                                memory_action=event['memory_action']))
                    else:counts['existing_write_annotation_or_birth_unknown']+=1
            counts['NEW_entity_FN']+=len(birth_eligible-true_birth)
            counts['MATCH_entity_FN']+=len(existing_gt-true_match)
            counts['MATCH_entity_TP']+=len(existing_gt&true_match)
            for identity,indices in by_gt.items():
                observations=accepted.get(identity,[])
                counts['supported_entity_frame_denominator']+=1
                counts['supported_entity_no_representative']+=not observations
                counts['supported_duplicate_proposal_extras']+=max(len(indices)-1,0)
                counts['extra_accepted_duplicate_identities']+=max(len(observations)-1,0)
                counts['supported_duplicate_extras_suppressed_with_entity_retained']+=max(len(indices)-len(observations),0) if observations else 0
                prior_candidate=last_candidate.get(identity)
                if prior_candidate and frames-prior_candidate[0]>1:
                    counts['natural_candidate_gap_returns']+=1
                    correct=[events[i]['track_id'] for i in observations if anchors.get(events[i]['track_id'])==identity]
                    counts['natural_gap_old_ID_recovered']+=last_correct.get(identity) in correct
                correct=[events[i]['track_id'] for i in observations if anchors.get(events[i]['track_id'])==identity]
                if correct:last_correct[identity]=correct[0]
                last_candidate[identity]=(frames,row['state']['timestamp'])
            present={g['identity'] for g in gt}
            for event in row['state']['events']:
                if event['action'] in ('RETIRED','TERMINATE'):
                    counts['retired_events']+=1
                    counts['retired_known_identity_still_annotated_present']+=before.get(event['track_id']) in present
            report=row['state']['decision_metadata'].get('assigned_identity_update',{})
            counts['actual_UPDATE_HOLD_flips_vs_original']+=report.get('update_HOLD_flips_vs_original',0)
            counts['frames']+=1;counts['candidates']+=len(row['candidates']);frames+=1;prior=set(row['state']['persistent_ids'])
    def ratio(a,b):return counts[a]/counts[b] if counts[b] else None
    def fraction(a,b):return a/b if b else None
    result=dict(status='GLOBAL_ENTITY_AND_MEMORY_POSTHOC_DIAGNOSTICS_COMPLETE',diagnostic_version=2,counts=dict(counts),
        NEW_precision=fraction(counts['NEW_known_TP'],counts['NEW_known_TP']+counts['NEW_known_FP']),
        NEW_recall=fraction(counts['NEW_known_TP'],counts['NEW_known_TP']+counts['NEW_entity_FN']),
        MATCH_precision=fraction(counts['MATCH_known_TP'],counts['MATCH_known_TP']+counts['MATCH_known_FP']),
        MATCH_recall=fraction(counts['MATCH_entity_TP'],counts['MATCH_entity_TP']+counts['MATCH_entity_FN']),
        all_live_retrieval_coverage=ratio('retrieval_all_live_correct_covered','retrieval_previously_confirmed_candidate_denominator'),
        initial_Top32_retrieval_coverage=ratio('retrieval_initial_Top32_correct_covered','retrieval_previously_confirmed_candidate_denominator'),
        existing_identity_rank_accuracy=ratio('existing_rank_above_NEW','known_existing_candidate_denominator'),
        wrong_long_write_examples=examples,GT_used_posthoc_only=True,
        GT_scope='Marked MOT pedestrians / release physical RMOT objects / nonexhaustive TAO annotations. Unmatched and unknown births are not universal negatives.',
        clock_unit='physical_seconds' if manifest.get('physical_video_fps_verified',True) else 'annotation_steps',
        memory_scope='UPDATE/HOLD gates protected long memory; recent observational evidence always follows the original runtime and its wrong writes are counted separately.',
        metric_scope='Known-label event diagnostics plus entity-frame FN counts; not replacement official HOTA/IDF1/TETA, not full-dataset results.')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads(args.video.read_text());result=analyze(manifest,args.run)
    path=args.run/'global_identity_diagnostics.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
