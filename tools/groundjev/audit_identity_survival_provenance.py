"""Reconstruct fixed-birth identity availability and actual retirement causes offline."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.groundjev.analyze_tracking_events import load_truth


def audit(source,manifest):
    runtime=json.loads((source/'result.json').read_text())
    diagnostic=json.loads((source/'association_events.json').read_text())
    truth,_=load_truth(manifest);offline=iter(diagnostic['events'])
    born,live,ever_bound,last_retirement={}, {},defaultdict(list),{}
    seen_gt=set();counts=Counter();missing=Counter();retired=Counter();examples=[]
    timeout=runtime['lost_seconds']
    clock_units='physical_seconds' if manifest.get('physical_video_fps_verified',True) else 'annotation_steps'
    with (source/'frame_decisions.jsonl').open() as f:
        for expected,line in zip(manifest['frame_numbers'],f):
            row=json.loads(line);assert row['frame']==expected
            counts['frames']+=1;timestamp=row['state']['timestamp']
            prior=set(live);before={tid:born[tid] for tid in prior}
            actual,arrivals={},[];pending=[]
            for event in row['state']['events']:
                if 'candidate' not in event:continue
                target=next(offline);index=event['candidate']
                assert (target['frame'],target['candidate'])==(expected,index)
                counts['candidates']+=1;identity=target['matched_gt_identity']
                existing={tid for tid in prior if identity is not None and before[tid]==identity}
                assert existing==set(target['correct_surviving_anchor_ids'])
                options={str(k) for k in event['decision']['probabilities']} - {'NEW','IGNORE','NO_MATCH'}
                correct={str(tid) for tid in existing}&options
                assert bool(correct)==target['correct_option_present']
                if identity is not None:
                    counts['GT_matched_candidates']+=1;arrivals.append(identity)
                    if identity not in seen_gt:
                        category='first_GT_matched_candidate_observation'
                    elif correct:
                        category='verified_correct_birth_in_actual_shortlist'
                    elif existing:
                        category='verified_correct_birth_alive_but_shortlist_missing'
                    elif ever_bound[identity]:
                        category='all_previously_GT_bound_births_retired'
                        cause=last_retirement[identity]['cause']
                        missing['retired_latest_cause_'+cause]+=1
                    else:
                        category='seen_GT_candidate_before_but_no_GT_bound_birth_ever'
                    missing[category]+=1
                    missing[category+'_action_'+event['action']]+=1
                    if category in ('all_previously_GT_bound_births_retired','seen_GT_candidate_before_but_no_GT_bound_birth_ever'):
                        missing[category+'_unknown_birth_tracks_alive']+=any(before[tid] is None for tid in prior)
                        if len(examples)<12:
                            examples.append(dict(frame=expected,candidate=index,GT_identity=identity,category=category,
                                actual_action=event['action'],latest_retirement=last_retirement.get(identity)))
                tid=event.get('track_id')
                if event['action']=='NEW':
                    assert tid not in born
                    pending.append((tid,identity));live[tid]=timestamp
                    counts['GT_bound_births' if identity is not None else 'GT_unknown_births']+=1
                elif tid is not None:
                    assert tid in prior
                    live[tid]=timestamp
                actual[index]=identity
            assert len(actual)==len(row['candidates'])
            current_gt={g['identity'] for g in truth[expected]}
            supported={identity for identity in actual.values() if identity is not None}
            lifecycle={e['track_id']:e for e in row['state']['events'] if 'candidate' not in e}
            final=set(row['state']['persistent_ids']);deaths=prior-final
            for tid in deaths:
                event=lifecycle[tid];assert event['action']=='TERMINATE'
                decision=event['lifecycle_decision']
                explicit=bool(decision and decision['choice']=='TERMINATE')
                expired=timestamp-live[tid]>timeout
                assert explicit or expired or runtime['memory_policy']=='frame_only'
                cause=('learned_and_timeout' if explicit and expired else 'learned_before_timeout' if explicit else
                       'timeout_without_learned_terminate' if expired else 'frame_only')
                identity=born[tid];prefix='GT_unknown' if identity is None else 'GT_bound'
                retired[prefix+'_'+cause]+=1
                retired[prefix+'_'+cause+'_GT_present']+=identity in current_gt
                retired[prefix+'_'+cause+'_native_candidate_present']+=identity in supported
                if identity is not None:
                    last_retirement[identity]=dict(frame=expected,timestamp=timestamp,track_id=tid,cause=cause,
                        registered_clock_gap_since_last_assignment=timestamp-live[tid],clock_units=clock_units,GT_present=identity in current_gt,
                        native_candidate_present=identity in supported)
                del live[tid]
            for tid,identity in pending:
                born[tid]=identity
                if identity is not None:ever_bound[identity].append(tid)
            assert set(live)==final
            seen_gt.update(arrivals)
    assert next(offline,None) is None
    assert counts['frames']==len(manifest['frame_numbers'])==runtime['frames']
    assert counts['candidates']==diagnostic['counts']['candidates']
    return dict(run=str(source),clock_units=clock_units,timeout_in_registered_clock_units=timeout,
                counts=dict(counts),availability=dict(missing),retirements=dict(retired),examples=examples)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--appearance-plan',type=Path,required=True)
    parser.add_argument('--native-completed',type=Path,required=True);parser.add_argument('--native-plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    appearance=json.loads(args.appearance_plan.read_text());native=json.loads(args.native_completed.read_text())
    native_plan=json.loads(args.native_plan.read_text());results={}
    for scope,job in appearance['scopes'].items():
        manifest=json.loads((ROOT/job['manifest']).read_text())
        results['compact_original_'+scope]=audit(Path(appearance['reference_root'])/(scope+'_before'),manifest)
    for name,record in native['completed'].items():
        scope=next(s for s in ('mot_full1203','rmot_single800','rmot_multi800') if name.endswith(s))
        manifest=json.loads((ROOT/native_plan['scopes'][scope]['manifest']).read_text())
        results['native_'+name]=audit(Path(record['run']),manifest)
    report=dict(status='FROZEN_IDENTITY_BIRTH_SURVIVAL_RETIREMENT_PROVENANCE_AUDIT_COMPLETE',units=results,
        scope='Five original Compact scopes and nine completed native controls. Known identities use the original immutable GT-at-birth binding; unknown births remain unknown. No GT corrects a history or prediction. Never-bound is not proof no physical identity existed. Latest retirement cause is temporal provenance, not sole causal attribution or official metric.',
        original_predictions_or_protocol_changed=False,new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],units={k:dict(counts=v['counts'],availability=v['availability'],retirements=v['retirements']) for k,v in results.items()})))


if __name__=='__main__':main()
