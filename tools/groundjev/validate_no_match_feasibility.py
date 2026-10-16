"""All frozen real assignments for default parity and active-pool feasibility."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from locatemot.groundjev.assignment import assign_candidates
    parser=argparse.ArgumentParser();parser.add_argument('--source-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=json.loads(args.source_audit.read_text());units={}
    for name,unit in source['units'].items():
        prior=[];frames=0;candidates=0;changed=0;masked_rows=0;unmasked_frames=0
        with (Path(unit['run'])/'frame_decisions.jsonl').open() as f:
            for line in f:
                row=json.loads(line);events=[e for e in row['state']['events'] if 'candidate' in e]
                assert [e['candidate'] for e in events]==list(range(len(row['candidates'])))
                decisions=[e['decision'] for e in events];tracks=[SimpleNamespace(track_id=tid) for tid in prior]
                original=[dict(d['probabilities']) for d in decisions];meta=row['state']['decision_metadata']
                options=dict(native_probabilities=meta['native_probabilities'],explicit_no_match=meta.get('explicit_no_match',False))
                baseline,actions,none=assign_candidates(decisions,tracks,**options)
                constrained,other_actions,masked=assign_candidates(decisions,tracks,no_match_policy='only_if_omitted',**options)
                assert not none and other_actions==actions
                assert original==[d['probabilities'] for d in decisions]
                def decode(column,index):
                    return ('EXISTING',prior[column]) if column<len(prior) else (actions[(column-len(prior))%len(actions)],None)
                for i,event in enumerate(events):
                    old=decode(baseline[i],i);new=decode(constrained[i],i)
                    expected=('EXISTING',event['track_id']) if event['action'] in ('ACTIVE','REACTIVATE') else (event['action'],None)
                    assert old==expected,(name,row['frame'],i,old,expected)
                    if i in masked:assert new[0]!='NO_MATCH'
                    changed+=new!=old
                expected=[i for i,d in enumerate(decisions) if 'NO_MATCH' in actions and 'NO_MATCH' in d['probabilities'] and
                          all(str(tid) in d['probabilities'] for tid in prior)]
                assert masked==expected
                if not masked:assert baseline==constrained;unmasked_frames+=1
                masked_rows+=len(masked);frames+=1;candidates+=len(events);prior=row['state']['persistent_ids']
        assert frames==unit['counts']['frames'] and candidates==unit['counts']['candidates']
        units[name]=dict(frames=frames,candidates=candidates,default_original_assignments_exact=True,
            raw_probability_values_unchanged=True,constrained_ineligible_NO_MATCH_never_selected=True,
            actual_masked_rows=masked_rows,no_mask_frames_exact=unmasked_frames,
            same_input_shadow_changed_assignments=changed)
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(status='ALL14_REAL_DEFAULT_ASSIGNMENTS_AND_NO_MATCH_FEASIBILITY_VALIDATED',units=units,
        scope='Same original saved model probabilities and ordered physical live pools. Default assignments match every original candidate; new feasibility only restricts private NO_MATCH costs. No model/lifecycle/semantic inference or counterfactual trajectory is reconstructed, and changed shadow assignments are not HOTA gains.',
        raw_probabilities_not_renormalized=True,new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],units=len(units),candidates=sum(v['candidates'] for v in units.values()),
        changed_shadow_assignments={k:v['same_input_shadow_changed_assignments'] for k,v in units.items()})))


if __name__=='__main__':main()
