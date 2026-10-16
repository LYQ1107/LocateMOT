"""Real immutable observations: physical query invariance and canonical MOT parity."""
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
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=json.loads(args.plan.read_text());torch.set_num_threads(plan['cpu_threads'])
    args.out.mkdir(parents=True,exist_ok=False)
    dataset=CompactQuestionData(ROOT/plan['output'])
    source=torch.load(ROOT/plan['initial_checkpoint'],map_location='cpu',weights_only=True)
    original=CompactCore(source['architecture'],semantic=True)
    original.load_state_dict(source['state_dict']);original.eval()
    canonical=CompactCore(source['architecture'],semantic=True,canonical_physical_core=True)
    missing,unexpected=canonical.load_state_dict(source['state_dict'],strict=False)
    assert missing==['physical_spec_text'] and not unexpected
    canonical.physical_spec_text.copy_(dataset.native.text.specification(plan['physical_specification']))
    canonical.eval()
    checkpoint=dict(source,state_dict=canonical.state_dict(),canonical_physical_core=True,
        physical_specification=plan['physical_specification'],trained_updates=0)
    path=Path(plan['untrained_checkpoint']);assert not path.exists();torch.save(checkpoint,path)
    conditions=[plan['physical_specification'],
        dict(task='rmot',vocabulary=['person'],expression='a person wearing red'),
        dict(task='rmot',vocabulary=['person'],expression='a person wearing blue'),
        dict(task='rmot',vocabulary=['person'],expression='people walking on the left'),
        dict(task='ovmot',vocabulary=['person','car'],expression=None),
        dict(task='ovmot',vocabulary=['car','person'],expression=None)]
    texts=[dataset.native.text.specification(s) for s in conditions]
    selected=[]
    for unit in dataset.native.units:
        rows=[r for (u,_),r in dataset.native.records.items() if u==unit and r['candidate_refs']]
        assert rows,unit
        row=max(rows,key=lambda r:bool(r['history_refs']))
        selected.append((unit+'/'+row['id'],dataset.native.inputs(unit,row['id'])))
    # Include real single-object Base observations, which have no historical tracks.
    for index in dataset.base.sampled_draw_rows[:6]:selected.append(('Base/'+str(index),dataset.base.inputs(index)))
    physical=('match','update','reliability','life')
    changed_original=0;changed_membership=0;comparisons=0;elements=0;with_history=0
    with torch.no_grad():
        for name,inputs in selected:
            reference=dict(inputs,task_index=0,spec_text=texts[0])
            expected=original(reference);actual=canonical(reference)
            assert all(torch.equal(expected[k],actual[k]) for k in expected),name
            elements+=sum(v.numel() for v in actual.values());with_history+=bool(len(inputs['history']))
            for condition,text in zip(conditions[1:],texts[1:]):
                alternative=dict(inputs,task_index=('mot','rmot','ovmot').index(condition['task']),spec_text=text)
                y=canonical(alternative);old=original(alternative)
                assert all(torch.equal(y[k],actual[k]) for k in physical),(name,condition)
                changed_original+=any(not torch.equal(old[k],expected[k]) for k in physical)
                changed_membership+=not torch.equal(y['membership'],actual['membership'])
                comparisons+=1
    assert changed_original and changed_membership and with_history
    assert not torch.cuda.is_initialized()
    result=dict(status='REAL_NATIVE_CANONICAL_PHYSICAL_CORE_QUERY_INVARIANCE_AND_MOT_OUTPUT_PARITY',
        conditional_physical_query_invariance_exact=True,canonical_MOT_all_output_tensors_equal_original=True,
        actual_native_units=len(dataset.native.units),real_observation_groups=len(selected),
        groups_with_history=with_history,query_interventions=comparisons,
        canonical_MOT_output_elements_checked=elements,original_physical_changed_interventions=changed_original,
        canonical_membership_changed_interventions=changed_membership,
        condition_scope='Identical candidates, history, geometry and class_text; only declared task/spec_text changes. Does not assert invariance when query changes frontend proposals or rendered output filtering.',
        physical_specification=plan['physical_specification'],conditions=conditions,
        untrained_checkpoint=str(path),new_GPU_or_production_training_updates=0,
        comparison='Factorized runtime intervention; no tracking gain or unique method novelty established.',
        full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
