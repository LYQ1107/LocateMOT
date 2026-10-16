"""Geometry/appearance option coverage on one unchanged real causal trajectory."""
import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from locatemot.groundjev.decision import overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import ReplayFrames,restore_completed_prefix
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest=json.loads((ROOT/plan['manifest']).read_text())
    run=ROOT/plan['run'];runtime=json.loads((run/'result.json').read_text())
    assert runtime['all_input_frames_processed'] and runtime['frames']==plan['frames']
    assert runtime['memory_policy']=='gated' and runtime['lost_seconds']==3
    diagnostic=json.loads((run/'association_events.json').read_text())
    observations={(e['frame'],e['candidate']):e for e in diagnostic['events']}
    tracker=OnlineTracker(None);spec=runtime['specification'];source=ReplayFrames(ROOT/plan['source'])
    counts=Counter();buckets={};examples=[]
    try:
        with (run/'frame_decisions.jsonl').open() as trace:
            for expected,line in zip_longest(zip(manifest['frame_numbers'],manifest['images']),trace):
                assert expected is not None and line is not None
                frame,path=expected;saved=json.loads(line);raw=source.frame(frame);candidates=raw['candidates']
                assert saved['frame']==frame and saved['image_path']==path==raw['image_path']
                assert saved['candidates']==[{k:v for k,v in c.items() if k not in ('native_region_feature','temporal_region_feature')} for c in candidates]
                tracks=list(tracker.tracks.values());timestamp=saved['state']['timestamp']
                events={e['candidate']:e for e in saved['state']['events'] if 'candidate'in e}
                similarities=None
                if candidates and tracks:
                    memory=np.asarray([[t.anchor_feature,t.recent_feature,t.long_feature] for t in tracks],dtype=float)
                    current=np.asarray([c['native_region_feature'] for c in candidates],dtype=float)
                    assert memory.shape==(len(tracks),3,2560) and current.shape==(len(candidates),2560)
                    memory=memory/np.maximum(np.linalg.norm(memory,axis=-1,keepdims=True),1e-12)
                    current=current/np.maximum(np.linalg.norm(current,axis=-1,keepdims=True),1e-12)
                    similarities=(current@memory.reshape(-1,2560).T).reshape(len(candidates),len(tracks),3).mean(-1)
                for begin in range(0,len(candidates),16):
                    group=candidates[begin:begin+16]
                    ranked=sorted(tracks,key=lambda t:(-max(overlap(c['box'],t.predicted_box(timestamp)) for c in group),t.lost_age,t.track_id))
                    geometry=[t.track_id for t in ranked[:32]]
                    hybrid=geometry[:16]
                    if similarities is not None:
                        scores=similarities[begin:begin+len(group)].max(0)
                        order=sorted(range(len(tracks)),key=lambda i:(-float(scores[i]),tracks[i].lost_age,tracks[i].track_id))
                        for index in order:
                            if len(hybrid)==32:break
                            if tracks[index].track_id not in hybrid:hybrid.append(tracks[index].track_id)
                    assert len(hybrid)==len(set(hybrid))==len(geometry)
                    geo_ids,hybrid_ids=set(geometry),set(hybrid)
                    for local in range(len(group)):
                        index=begin+local
                        assert events[index]['decision']['shortlisted_ids']==geometry
                        observation=observations[(frame,index)]
                        correct=set(observation['correct_surviving_anchor_ids'])
                        if not correct:continue
                        assert correct<={t.track_id for t in tracks}
                        old,new=bool(correct&geo_ids),bool(correct&hybrid_ids)
                        counts['eligible_existing_identity_events']+=1
                        counts['geometry_correct_option_present']+=old
                        counts['hybrid_correct_option_present']+=new
                        counts['both_present']+=old and new
                        counts['hybrid_adds_correct_option']+=new and not old
                        counts['hybrid_loses_correct_option']+=old and not new
                        ages=[t.age for t in tracks if t.track_id in correct]
                        bucket='age1to31' if min(ages)<=31 else 'age32to127' if min(ages)<=127 else 'age128plus'
                        cell=buckets.setdefault(bucket,Counter());cell['events']+=1;cell['geometry']+=old;cell['hybrid']+=new
                        if old!=new and len(examples)<16:
                            examples.append(dict(frame=frame,candidate=index,correct_surviving_birth_IDs=sorted(correct),
                                geometry_present=old,hybrid_present=new,history_tracks=len(tracks)))
                complete=dict(saved,candidates=candidates)
                restore_completed_prefix(tracker,[complete],[(path,frame)],manifest['fps'],spec,{frame:timestamp})
                counts['frames']+=1;counts['candidates']+=len(candidates)
                counts['rejected_frames']+=bool(saved['perception'].get('model_response_rejected'))
    finally:source.close()
    assert counts['frames']==plan['frames'] and counts['candidates']==plan['candidates']
    assert counts['eligible_existing_identity_events']==diagnostic['counts']['existing_identity_events']
    assert counts['geometry_correct_option_present']==diagnostic['counts']['correct_identity_option_present']
    n=counts['eligible_existing_identity_events']
    report=dict(status='FULL_NATURAL_FROZEN_TRAJECTORY_NATIVE_HYBRID_SHORTLIST_COVERAGE_VALIDATED',
        source_plan=str(args.plan),counts=dict(counts),
        fixed_trajectory_geometry32_coverage=counts['geometry_correct_option_present']/n if n else None,
        fixed_trajectory_hybrid32_coverage=counts['hybrid_correct_option_present']/n if n else None,
        correct_anchor_age_buckets={k:dict(v) for k,v in buckets.items()},first16_changed_examples=examples,
        every_original_output_event_and_state_replayed_exactly=True,
        no_GT_in_retrieval_or_tracker=True,no_new_model_inference_or_feature_archive=True,
        policy='Existing fixed hybrid16_16; no parameter tuned on these events.',
        limitation='Only surviving known-birth identities under unchanged original history. Does not establish counterfactual HOTA/IDF1, improve already-deleted identities, or prove learned Full performance.',
        full_goal_complete=False)
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<plan['report_cap_bytes']
    print(json.dumps({k:v for k,v in report.items() if k not in ['first16_changed_examples']}),flush=True)


if __name__=='__main__':main()
