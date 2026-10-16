"""Real causal prefixes for opt-in early-termination deferral; no model or GT run."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def verify(source,manifest):
    from PIL import Image
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import restore_completed_prefix,ReplayFrames
    runtime=json.loads((source/'result.json').read_text())
    rows=[];last={};early=set()
    with (source/'frame_decisions.jsonl').open() as f:
        for line in f:
            row=json.loads(line);rows.append(row);timestamp=row['state']['timestamp']
            for event in row['state']['events']:
                if 'candidate' in event and event.get('track_id') is not None:
                    last[event['track_id']]=timestamp
                elif 'candidate' not in event and event['action']=='TERMINATE':
                    decision=event['lifecycle_decision'];tid=event['track_id']
                    if decision and decision['choice']=='TERMINATE' and timestamp-last[tid]<=runtime['lost_seconds']:
                        early.add(tid)
            if early:break
    assert early
    raw=ReplayFrames(Path(runtime['shared_input_features_source']))
    try:
        for row in rows:
            original=raw.frame(row['frame'])['candidates']
            strip=lambda c:{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')}
            assert [strip(c) for c in row['candidates']]==[strip(c) for c in original]
            row['candidates']=original
    finally:raw.close()
    chosen=rows[-1]
    class Recorded:
        def decide(self,*unused):
            decisions=[None]*len(chosen['candidates']);lifecycle={}
            for event in chosen['state']['events']:
                if 'candidate' in event:decisions[event['candidate']]=copy.deepcopy(event['decision'])
                elif event.get('lifecycle_decision') is not None:lifecycle[event['track_id']]=copy.deepcopy(event['lifecycle_decision'])
            return decisions,lifecycle,copy.deepcopy(chosen['state']['decision_metadata'])
    original=OnlineTracker(Recorded(),lost_seconds=runtime['lost_seconds'],memory_policy=runtime['memory_policy'])
    frames=list(zip(manifest['images'],manifest['frame_numbers']))[:len(rows)]
    timestamps=dict(zip(manifest['frame_numbers'],manifest['timestamps'])) if 'timestamps' in manifest else None
    restore_completed_prefix(original,rows[:-1],frames[:-1],manifest['fps'],runtime['specification'],timestamps)
    deferred=copy.deepcopy(original);deferred.lifecycle_termination_policy='timeout_only'
    with Image.open(chosen['image_path']) as f:image=f.convert('RGB')
    old_outputs,old_state=original.step(image,chosen['candidates'],chosen['state']['timestamp'],runtime['specification'])
    assert old_outputs==chosen['outputs'] and old_state==chosen['state']
    new_outputs,new_state=deferred.step(image,chosen['candidates'],chosen['state']['timestamp'],runtime['specification'])
    assert new_outputs==old_outputs
    assert set(deferred.tracks)==set(original.tracks)|early
    assert all(e['track_id'] in early for e in new_state['events'] if e.get('learned_termination_deferred'))
    a={r['track_id']:r for r in original.state_dict()['tracks']}
    b={r['track_id']:r for r in deferred.state_dict()['tracks']}
    assert all(b[tid]==value for tid,value in a.items())
    class EmptyObservation:
        def decide(self,image,candidates,tracks,timestamp,specification):
            life={t.track_id:dict(choice='TERMINATE',probabilities=dict(ACTIVE=0.,LOST=0.,REACTIVATE=0.,TERMINATE=1.)) for t in tracks}
            return [],life,dict(native_probabilities=True)
    deferred.decision=EmptyObservation()
    deferred.step(image,[],chosen['state']['timestamp']+runtime['lost_seconds']+.01,runtime['specification'])
    assert not deferred.tracks
    return dict(source=str(source),prefix_frames=len(rows),first_early_frame=chosen['frame'],
        actual_early_IDs=sorted(early),original_outputs_and_entire_prefix_state_exact=True,
        changed_ID_set_exactly_original_early_deletions=True,common_identity_states_exact=True,
        timeout_expiry_on_restored_state_verified=True,
        expiry_check_scope='Functional empty-observation/forced-head counterfactual after original3s, not a measured video frame or tracking metric.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=json.loads(args.plan.read_text());out=ROOT/plan['termination_validation_output'];out.mkdir(parents=True,exist_ok=False)
    compact=Path(plan['reference_root'])/'rmot_single800_before'
    native=ROOT/'outputs/groundjev/training/broader_shared6000_fla_owner_v2/controls/default_mot_full1203'
    cases={}
    for name,source,scope in [('compact',compact,'rmot_single800'),('native',native,'mot_full1203')]:
        cases[name]=verify(source,json.loads((ROOT/plan['scopes'][scope]['manifest']).read_text()))
    report=dict(status='REAL_NATIVE_AND_COMPACT_DEFAULT_PREFIX_EARLY_DEFERRAL_AND_EXPIRY_VALIDATED',cases=cases,
        GT_or_model_or_GPU_run=False,all_original_auxiliary_and_matching_decisions_retained=True,
        default_policy_unchanged=True,new_optimizer_updates=0,full_goal_complete=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
