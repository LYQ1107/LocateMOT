"""Posthoc exposure of the complete live identity pool for original NO_MATCHs."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    sources=json.loads(args.provenance.read_text())['units'];results={}
    for name,source in sources.items():
        run=Path(source['run']);diagnostic=json.loads((run/'association_events.json').read_text())
        external=iter(diagnostic['events']);prior=set();counts=Counter();groups={};examples=[];minimum=1.
        with (run/'frame_decisions.jsonl').open() as f:
            for line in f:
                row=json.loads(line);counts['frames']+=1
                for event in row['state']['events']:
                    if 'candidate' not in event:continue
                    d=event['decision'];p=d['probabilities'];target=next(external)
                    assert (target['frame'],target['candidate'])==(row['frame'],event['candidate'])
                    offered={int(k) for k in p if k not in ('NEW','IGNORE','NO_MATCH')}
                    assert offered<=prior
                    if 'shortlisted_ids' in d:assert offered==set(d['shortlisted_ids'])
                    counts['candidates']+=1;complete=offered==prior
                    counts['complete_pool_rows' if complete else 'omitted_live_ID_rows']+=1
                    counts['NO_MATCH_selected']+=event['action']=='NO_MATCH'
                    if not complete or 'NO_MATCH' not in p:continue
                    counts['complete_pool_with_NO_MATCH_option']+=1
                    mass=sum(value for key,value in p.items() if key!='NO_MATCH')
                    minimum=min(minimum,mass);counts['zero_remaining_printed_probability_mass']+=mass==0
                    if event['action']!='NO_MATCH':continue
                    counts['complete_pool_NO_MATCH_selected']+=1
                    if target['matched_gt_identity'] is None:label='GT_unmatched_not_universal_negative'
                    elif target['correct_surviving_anchor_ids']:label='GT_matched_correct_live_anchor_offered'
                    else:label='GT_matched_without_GT_verified_live_anchor'
                    group=groups.setdefault(label,Counter());group['observations']+=1
                    chosen=max((key for key in p if key!='NO_MATCH'),key=p.get)
                    group['best_remaining_'+('EXISTING' if chosen.isdecimal() else chosen)]+=1
                    if len(examples)<8:
                        examples.append(dict(frame=row['frame'],candidate=event['candidate'],pool_size=len(prior),
                            offered_ids=len(offered),group=label,NO_MATCH_probability=p['NO_MATCH'],
                            best_remaining_option=chosen,best_remaining_probability=p[chosen]))
                prior=set(row['state']['persistent_ids'])
        assert next(external,None) is None
        assert counts['frames']==source['counts']['frames'] and counts['candidates']==source['counts']['candidates']
        results[name]=dict(run=str(run),counts=dict(counts),selected_NO_MATCH_groups={k:dict(v) for k,v in groups.items()},
            minimum_remaining_printed_probability_mass=minimum,examples=examples)
    report=dict(status='ALL14_ORIGINAL_NO_MATCH_LIVE_POOL_EXPOSURE_AUDITED',units=results,
        interpretation='Complete pool is an observable causal property, no GT required. Under the exporter active-pool contract, NO_MATCH requires an omitted live ID. Old V3prompt says previously observed without that explicit qualification; this audit diagnoses a contract discrepancy, not a proof every original NO_MATCH was impossible under old wording. Unknown births remain unknown; unmatched GT is not universal negative truth.',
        hypothetical_remaining_option='Pointwise best remaining original score only; not global assignment, a new trajectory, calibrated posterior, or measured quality gain.',
        new_GPU_or_optimizer_updates=0,original_predictions_or_targets_changed=False,full_goal_complete=False)
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],units={k:dict(counts=v['counts'],selected_NO_MATCH_groups=v['selected_NO_MATCH_groups']) for k,v in results.items()})))


if __name__=='__main__':main()
