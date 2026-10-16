"""Real-feature controller regression against the published native affinity."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from locatemot.groundjev.compact import CompactCore,CompactAffinityControl
    from locatemot.groundjev.decision import overlap
    from tools.groundjev.compact_question_data import CompactQuestionData
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(plan['cpu_threads']);args.out.mkdir(parents=True,exist_ok=False)
    old={};exec(compile(subprocess.check_output(['git','show',plan['original_module_commit']+':locatemot/groundjev/compact.py']),
        'published_compact.py','exec'),old)
    data=CompactQuestionData(ROOT/'outputs/groundjev/compact/native_relation_question_curriculum_v1')
    source=torch.load(ROOT/plan['initial_checkpoint'],map_location='cpu',weights_only=True)
    model=CompactCore(source['architecture'],semantic=True);model.load_state_dict(source['state_dict']);model.eval()
    selected=[]
    for unit in data.native.units:
        rows=[r for (u,_),r in data.native.records.items() if u==unit and r['candidate_refs']]
        row=max(rows,key=lambda r:bool(r['history_refs']))
        selected.append(data.native.inputs(unit,row['id']))
    selected.extend(data.base.inputs(index) for index in data.base.sampled_draw_rows[:6])
    class Frozen:
        memory_feature_view='all'
        def __init__(self,values):self.model=model;self.text_features=data.native.text;self.values=values
        def decide(self,*unused):return copy.deepcopy(self.values)
    maxerror=0.;comparisons=0;representation_changes=0
    for inputs in selected:
        n,m=len(inputs['candidates']),len(inputs['history'])
        with torch.no_grad():y=model(inputs)
        base=[]
        for i in range(n):
            base.append(dict(probabilities={str(j+1):1/(m+2) for j in range(m)},shortlisted_ids=list(range(1,m+1)),
                update_probability=float(y['update'][i].sigmoid()),reliability=1.,
                query_membership_probability=float(y['membership'][i].sigmoid()),
                category_probabilities=y['category'][i].softmax(-1).tolist()))
        frozen=Frozen((base,{j+1:{'type':'choice','choice':'ACTIVE'} for j in range(m)},dict(native_probabilities=True)))
        candidates=[dict(box=(inputs['candidate_geo'][i]*1000).tolist(),native_region_feature=inputs['candidates'][i].tolist()) for i in range(n)]
        tracks=[]
        for j in range(m):
            box=(inputs['history_geo'][j,:4]*1000).tolist()
            tracks.append(SimpleNamespace(track_id=j+1,anchor_feature=inputs['history'][j,0].tolist(),
                recent_feature=inputs['history'][j,1].tolist(),long_feature=inputs['history'][j,2].tolist(),predicted_box=lambda _,b=box:b))
        args_=(None,candidates,tracks,0.,dict(task='mot'))
        before=old['CompactAffinityControl'](frozen).decide(*args_)
        default=CompactAffinityControl(frozen).decide(*args_)
        assert before==default
        values=[]
        for checkpoint in [Path(plan['training_output'])/'initial_projection.pt',Path(plan['training_output'])/'projection.pt']:
            control=CompactAffinityControl(frozen,appearance_projection_checkpoint=checkpoint)
            predicted=control.decide(*args_);values.append(predicted)
            current=control.appearance_projector([c['native_region_feature'] for c in candidates])
            history=control.appearance_projector([v for t in tracks for v in (t.anchor_feature,t.recent_feature,t.long_feature)]).reshape(m,3,64)
            for i,row in enumerate(predicted[0]):
                for key in ('query_membership_probability','category_probabilities'):
                    assert row[key]==default[0][i][key]
                for j,track in enumerate(tracks):
                    expected=.5*overlap(candidates[i]['box'],track.predicted_box(0.))+.5*max(0.,float((history[j]@current[i]).mean()))
                    error=abs(expected-row['probabilities'][str(j+1)])
                    assert error<=2e-7
                    maxerror=max(maxerror,error);comparisons+=1
                assert row['probabilities']['NEW']==.3 and row['probabilities']['IGNORE']==0.
                assert row['update_probability']==1. and row['reliability'] is None
            assert predicted[1]=={} and not predicted[2]['native_probabilities']
        representation_changes+=values[0][0]!=values[1][0]
    assert representation_changes and comparisons and not torch.cuda.is_initialized()
    result=dict(status='REAL_NATIVE_TEMPORAL_APPEARANCE_CONTROL_REGRESSION_VALIDATED',
        observation_groups=len(selected),real_native_units=len(data.native.units),projected_affinity_pairs_checked=comparisons,
        all_native_default_scores_and_semantic_fields_exact=True,all_projected_semantic_fields_equal_native=True,
        fixed_fusion_formula_max_absolute_error=maxerror,initial_vs_trained_affinities_changed_groups=representation_changes,
        geometry_scope='Original real normalized boxes rescaled consistently to1000x1000 for formula checks; original native feature/semantic tensor inputs. Representative groups, not full causal video trajectory parity.',
        no_runtime_GT_or_query_input_to_projection=True,new_GPU_or_optimizer_updates=0,
        official_tracking_gain_claimed=False,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
