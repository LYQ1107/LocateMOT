"""Final-model physical allocation/write invariance on one real causal TRAIN input."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def validate(checkpoint,plan):
    import torch
    from PIL import Image
    from locatemot.groundjev.assignment import assign_candidates
    from locatemot.groundjev.global_identity_runtime import GlobalIdentityDecision
    from tools.groundjev.global_identity_data import training_frames
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(4)
    frame=next(f for f in training_frames(plan) if f['tracks'] and f['candidates'])
    with Image.open(frame['source_row']['image_path']) as raw:image=raw.convert('RGB')
    decision=GlobalIdentityDecision(checkpoint,memory_policy='assigned')
    assert all(torch.isfinite(v).all() for v in decision.model.state_dict().values())
    specs=[dict(task='mot',vocabulary=['person'],expression=None),
        dict(task='rmot',vocabulary=['person'],expression='the person on the left'),
        dict(task='rmot',vocabulary=['person'],expression='the person on the right'),
        dict(task='ovmot',vocabulary=['person','car','bicycle'],expression=None)]
    reference=None;assignments=0
    for spec in specs:
        rows,_,metadata=decision.decide(image,frame['candidates'],frame['tracks'],frame['timestamp'],spec)
        physical=[{k:v for k,v in r.items() if k not in ('category_decision','query_membership_probability')} for r in rows]
        assigned,actions,_=assign_candidates(rows,frame['tracks'],native_probabilities=True,explicit_no_match=False)
        selected={i:frame['tracks'][j].track_id for i,j in assigned.items() if j<len(frame['tracks'])}
        assert len(selected)==len(set(selected.values()))
        updates,receipt=decision.decide_assigned_updates(selected)
        assert receipt['GT_read'] is False and updates.keys()==selected.keys()
        for component in metadata['competition_components']:
            accepted=[i for i in component['candidates'] if assigned[i]<len(frame['tracks']) or
                actions[(assigned[i]-len(frame['tracks']))%len(actions)]=='NEW']
            assert len(accepted)<=1
        evidence=(physical,assigned,updates)
        if reference is None:reference=evidence
        else:assert evidence==reference
        assignments+=len(selected)
    fresh=GlobalIdentityDecision(checkpoint,memory_policy='assigned')
    assert all(torch.equal(v,fresh.model.state_dict()[k]) for k,v in decision.model.state_dict().items())
    with torch.no_grad():a=decision.model(decision.cache['inputs']);b=fresh.model(decision.cache['inputs'])
    difference=max(float((a[k]-b[k]).abs().max()) for k in a if torch.is_tensor(a[k]) and a[k].numel())
    assert difference==0 and not torch.cuda.is_initialized()
    return dict(status='FINAL_SHARED_PHYSICAL_ALLOCATION_AND_ASSIGNED_WRITE_QUERY_INTERVENTIONS_VALIDATED',
        checkpoint=str(Path(checkpoint).resolve()),real_TRAIN_unit=frame['unit'],frame=frame['frame'],
        candidates=len(frame['candidates']),history=len(frame['tracks']),task_query_interventions=4,
        physical_logits_admission_global_assignments_and_assigned_updates_exact=True,
        actual_existing_assignments_checked=assignments,all_weights_finite=True,reload_max_difference=difference,
        physical_parameters=sum(p.numel() for p in decision.model.parameters()),new_GPU_processes=0,
        scope='One identical real causal TRAIN history across MOT/two RMOT queries/OVMOT; not a whole-video or changed-detector-candidate invariance or tracking-quality claim.',full_goal_complete=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/groundjev/global_identity_v1.json').read_text());result=validate(a.checkpoint,plan)
    assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
