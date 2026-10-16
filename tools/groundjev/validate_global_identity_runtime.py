"""Real final-checkpoint reload, physical-query invariance and allocation checks."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.assignment import assign_candidates
    from locatemot.groundjev.global_identity_runtime import GlobalIdentityDecision
    from tools.groundjev.global_identity_data import training_frames
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=json.loads(args.plan.read_text());torch.set_num_threads(4)
    iterator=training_frames(plan)
    frame=next(f for f in iterator if len(f['tracks']) and len(f['candidates']))
    with Image.open(frame['source_row']['image_path']) as image_source:image=image_source.convert('RGB')
    specs=[dict(task='mot',vocabulary=['person'],expression=None),
        dict(task='rmot',vocabulary=['person'],expression='the person on the left'),
        dict(task='rmot',vocabulary=['person'],expression='the person on the right'),
        dict(task='ovmot',vocabulary=['person','car','bicycle'],expression=None)]
    cases={}
    for arm in ('A1','A2'):
        root=Path(plan['model_output'])/arm;report=json.loads((root/'training_validation.json').read_text())
        assert report['epochs']==4 and report['counts']['frames']==4224 and report['counts']['candidates']==192204
        state=torch.load(root/'latest_training_state.pt',map_location='cpu',weights_only=True)
        assert state['steps']==report['actual_steps'] and state['epoch']==4
        assert all(torch.isfinite(t).all() for t in state['model'].values())
        optimizer_elements=0
        for moment in state['optimizer']['state'].values():
            assert 0<int(moment['step'])<=state['steps']
            for key in ('exp_avg','exp_avg_sq'):
                assert torch.isfinite(moment[key]).all();optimizer_elements+=moment[key].numel()
        decision=GlobalIdentityDecision(root/'checkpoint.pt')
        reference=None;semantic_changes=0;selected_count=0
        for spec in specs:
            rows,life,meta=decision.decide(image,frame['candidates'],frame['tracks'],frame['timestamp'],spec)
            physical=[{k:v for k,v in row.items() if k not in ('category_decision','query_membership_probability')} for row in rows]
            if reference is None:reference=physical
            else:assert physical==reference
            semantic_changes+=sum('category_decision' in r or 'query_membership_probability' in r for r in rows)
            assigned,actions,_=assign_candidates(rows,frame['tracks'],native_probabilities=True,explicit_no_match=False)
            selected={i:frame['tracks'][j].track_id for i,j in assigned.items() if j<len(frame['tracks'])}
            assert len(set(selected.values()))==len(selected)
            for component in meta['competition_components']:
                accepted=[i for i in component['candidates'] if assigned[i]<len(frame['tracks']) or
                    actions[(assigned[i]-len(frame['tracks']))%len(actions)]=='NEW']
                assert len(accepted)<=1
            updates,receipt=decision.decide_assigned_updates(selected)
            assert updates.keys()==selected.keys() and receipt['GT_read']==False
            assert all(updates[i]==rows[i]['update_probability'] for i in selected)
            selected_count+=len(selected)
        fresh=GlobalIdentityDecision(root/'checkpoint.pt')
        for name,tensor in decision.model.state_dict().items():assert torch.equal(tensor,fresh.model.state_dict()[name])
        with torch.no_grad():a=decision.model(decision.cache['inputs']);b=fresh.model(decision.cache['inputs'])
        difference=max(float((a[k]-b[k]).abs().max()) for k in a if torch.is_tensor(a[k]) and a[k].numel())
        assert difference==0
        cases[arm]=dict(actual_updates=report['actual_steps'],processed_frames=4224,candidate_exposures=192204,
            physical_parameters=report['parameters'],all_weights_and_Adam_moments_finite=True,
            Adam_moment_elements_checked=optimizer_elements,real_input_reload_max_difference=difference,
            task_query_interventions=len(specs),all_physical_rows_and_update_probabilities_exact=True,
            semantic_output_rows_present=semantic_changes,actual_existing_assignments_checked=selected_count,
            unique_existing_ID_and_predicted_duplicate_component_constraints_valid=True,
            original_assigned_callback_before_write_reused=True)
        del decision,fresh,state
    assert not torch.cuda.is_initialized()
    result=dict(status='GLOBAL_IDENTITY_FINAL_CPU_CHECKPOINTS_AND_REAL_RUNTIME_REGRESSION_VALIDATED',
        real_TRAIN_unit=frame['unit'],frame=frame['frame'],candidates=len(frame['candidates']),history=len(frame['tracks']),
        cases=cases,scope='One actual causal TRAIN history with four fixed task/query interventions; not whole-video query invariance or accuracy evidence. Same parameter file per arm for all tasks. Model inputs contain only candidate/history physical evidence, no GT labels.',
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    out=ROOT/plan['output']/'runtime_validation.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
