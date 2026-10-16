"""Real five-scope prefixes: legacy parity and regrouped neural row alignment."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.candidate_groups import candidate_group_indices
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import ReplayFrames
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4)
    plan=json.loads(args.plan.read_text());args.out.mkdir(parents=True,exist_ok=False);cases={}
    for scope in plan['scopes']:
        source=Path(plan['reference_root'])/(scope+'_before')
        runtime=json.loads((source/'result.json').read_text())
        base=CompactDecision(ROOT/plan['checkpoint'],device='cpu')
        changed=CompactDecision(ROOT/plan['checkpoint'],device='cpu',candidate_grouping='overlap_neighbors')
        old_tracker,new_tracker=OnlineTracker(base),OnlineTracker(changed)
        raw=ReplayFrames(Path(runtime['shared_input_features_source']));frames=candidates_count=groups_count=0
        try:
            with (source/'frame_decisions.jsonl').open() as f:
                for line in f:
                    row=json.loads(line);candidates=raw.frame(row['frame'])['candidates']
                    indices=candidate_group_indices(candidates,16,'overlap_neighbors')
                    assert sorted(i for g in indices for i in g)==list(range(len(candidates)))
                    expected={};called=[]
                    def capture(group,ranked,e):
                        selected=indices[len(called)];called.append(selected)
                        assert group==[candidates[i] for i in selected]
                        actual=torch.tensor([candidates[i]['native_region_feature'] for i in selected]).reshape(len(selected),2560)
                        assert torch.equal(actual,e['candidates'].cpu())
                        with torch.no_grad():y=changed.model(e)
                        names=[str(t.track_id) for t in ranked]+['NEW','IGNORE','NO_MATCH']
                        p=y['match'].softmax(-1).tolist();u=y['update'].sigmoid().tolist()
                        members=y['membership'].sigmoid().tolist() if 'membership' in y else None
                        classes=y['category'].softmax(-1) if 'category' in y else None
                        for local,index in enumerate(selected):
                            expected[index]=dict(probabilities=dict(zip(names,p[local])),update=u[local])
                            if runtime['specification']['task']=='rmot':
                                expected[index]['member']=members[local]
                            if runtime['specification']['task']=='ovmot':
                                cp=classes[local];j=int(cp.argmax())
                                expected[index]['category']=(runtime['specification']['vocabulary'][j],float(cp[j]))
                    changed.accepted_batch=capture
                    with Image.open(row['image_path']) as f:image=f.convert('RGB')
                    outputs,state=old_tracker.step(image,candidates,row['state']['timestamp'],runtime['specification'])
                    assert outputs==row['outputs'] and state==row['state'],(scope,row['frame'],'default parity')
                    _,new_state=new_tracker.step(image,candidates,row['state']['timestamp'],runtime['specification'])
                    assert new_state['decision_metadata']['candidate_group_indices']==indices
                    events=[e for e in new_state['events'] if 'candidate' in e]
                    assert [e['candidate'] for e in events]==list(range(len(candidates)))
                    assert set(expected)==set(range(len(candidates)))
                    for event in events:
                        result=event['decision'];reference=expected[event['candidate']]
                        assert result['probabilities']==reference['probabilities']
                        assert result['update_probability']==reference['update']
                        if 'member' in reference:assert result['query_membership_probability']==reference['member']
                        if 'category' in reference:
                            assert (result['category_decision']['label'],result['category_decision']['probability'])==reference['category']
                    frames+=1;candidates_count+=len(candidates);groups_count+=len(called)
                    if frames==3:break
        finally:raw.close()
        assert frames==3
        cases[scope]=dict(frames=frames,candidates=candidates_count,actual_model_groups=groups_count,
            entire_default_outputs_and_states_exact=True,all_candidates_once_original_output_order=True,
            actual_regrouped_model_input_feature_rows_exact=True,
            regrouped_probabilities_updates_and_task_semantics_aligned_to_original_rows=True)
    assert len(cases)==5 and not torch.cuda.is_initialized()
    report=dict(status='ALL5_REAL_DEFAULT_PREFIXES_AND_OVERLAP_GROUP_ROW_ALIGNMENT_VALIDATED',cases=cases,
        scope='Three actual causal frames per scope, frozen original checkpoint. Alternative model inputs and output row ownership checked against direct regrouped neural outputs. No GT inputs, full-video quality or speed claim.',
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
