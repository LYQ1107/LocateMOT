"""Real native tensors: canonical physical decisions, original task semantics."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from locatemot.groundjev.compact import CompactCore
    from tools.groundjev.compact_question_data import CompactQuestionData
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(4)
    plan=json.loads(args.plan.read_text());args.out.mkdir(parents=True,exist_ok=False)
    data=CompactQuestionData(ROOT/plan['validation_curriculum'])
    original_data=torch.load(ROOT/plan['checkpoint'],map_location='cpu',weights_only=True)
    canonical_data=torch.load(plan['canonical_checkpoint'],map_location='cpu',weights_only=True)
    assert set(canonical_data['state_dict'])==set(original_data['state_dict'])|{'physical_spec_text'}
    assert all(torch.equal(v,canonical_data['state_dict'][k]) for k,v in original_data['state_dict'].items())
    original=CompactCore(original_data['architecture'],semantic=True);original.load_state_dict(original_data['state_dict']);original.eval()
    canonical=CompactCore(original_data['architecture'],semantic=True,canonical_physical_core=True)
    canonical.load_state_dict(canonical_data['state_dict']);canonical.eval()
    readout=CompactCore(original_data['architecture'],semantic=True,canonical_physical_core=True,task_conditioned_semantic_readout=True)
    readout.load_state_dict(canonical_data['state_dict']);readout.eval()
    assert all(torch.equal(v,readout.state_dict()[k]) for k,v in canonical.state_dict().items())
    conditions=[dict(task='mot',vocabulary=['person'],expression=None),
        dict(task='rmot',vocabulary=['person'],expression='a person wearing red'),
        dict(task='rmot',vocabulary=['person'],expression='a person wearing blue'),
        dict(task='rmot',vocabulary=['person'],expression='people walking on the left'),
        dict(task='ovmot',vocabulary=['person','car'],expression=None),
        dict(task='ovmot',vocabulary=['car','person'],expression=None)]
    texts=[data.native.text.specification(s) for s in conditions]
    groups=[]
    for unit in data.native.units:
        candidates=[r for (u,_),r in data.native.records.items() if u==unit and r['candidate_refs']]
        row=max(candidates,key=lambda r:bool(r['history_refs']))
        groups.append((unit+'/'+row['id'],data.native.inputs(unit,row['id'])))
    for index in data.base.sampled_draw_rows[:6]:groups.append(('Base/'+str(index),data.base.inputs(index)))
    physical=('match','update','reliability','life');semantic=('category','membership')
    comparisons=elements=history=changed=0
    with torch.no_grad():
        for name,inputs in groups:
            neutral=dict(inputs,task_index=0,spec_text=texts[0])
            before=original(neutral);actual=readout(neutral)
            assert all(torch.equal(before[k],actual[k]) for k in before),(name,'MOT all-head parity')
            history+=bool(len(inputs['history']))
            for condition,text in zip(conditions,texts):
                e=dict(inputs,task_index=('mot','rmot','ovmot').index(condition['task']),spec_text=text)
                old=original(e);fixed=canonical(e);new=readout(e)
                assert all(torch.equal(new[k],fixed[k]) and torch.equal(new[k],actual[k]) for k in physical),(name,condition,'physical')
                assert all(torch.equal(new[k],old[k]) for k in semantic),(name,condition,'task semantics')
                changed+=any(not torch.equal(new[k],fixed[k]) for k in semantic)
                elements+=sum(v.numel() for v in new.values());comparisons+=1
    assert len(data.native.units)==61 and history and changed and not torch.cuda.is_initialized()
    path=Path(plan['task_readout_checkpoint']);assert not path.exists()
    variant=dict(canonical_data,task_conditioned_semantic_readout=True,
        source_canonical_checkpoint=plan['canonical_checkpoint'],source_coupled_checkpoint=plan['checkpoint'],
        new_optimizer_updates=0,variant='Same frozen physical and semantic parameters; task-conditioned second shared-core readout pass')
    torch.save(variant,path)
    reloaded=torch.load(path,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,reloaded['state_dict'][k]) for k,v in canonical.state_dict().items())
    report=dict(status='REAL_NATIVE_CANONICAL_PHYSICAL_AND_TASK_CONDITIONED_SEMANTIC_READOUT_EXACT',
        real_native_units=len(data.native.units),actual_native_and_Base_groups=len(groups),groups_with_history=history,
        task_interventions=comparisons,output_tensor_elements_checked=elements,canonical_semantic_changed_interventions=changed,
        same_all_original_parameter_values_and_no_new_parameter_tensors=True,
        physical_outputs_equal_canonical_and_invariant_to_declared_task_queries=True,
        semantic_logits_equal_original_query_coupled_model_on_identical_inputs=True,
        all_MOT_output_tensors_equal_original=True,task_readout_checkpoint=str(path),
        scope='Identical native current features/history/geometry/class_text, task/spec_text interventions. No actor state repair, GT runtime inputs, complete video or tracking-quality claim; actual new trajectories must be evaluated.',
        new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
