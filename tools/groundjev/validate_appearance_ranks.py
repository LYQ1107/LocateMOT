"""Conservation of native Compact actions/values on real observation groups."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    from locatemot.groundjev.compact import CompactCore
    from locatemot.groundjev.appearance_ranking import AppearanceRankDecision
    from tools.groundjev.compact_question_data import CompactQuestionData
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=False)
    data=CompactQuestionData(ROOT/'outputs/groundjev/compact/native_relation_question_curriculum_v1')
    saved=torch.load(ROOT/plan['checkpoint'],map_location='cpu',weights_only=True)
    model=CompactCore(saved['architecture'],semantic=True);model.load_state_dict(saved['state_dict']);model.eval()
    inputs=[]
    for unit in data.native.units:
        rows=[r for (u,_),r in data.native.records.items() if u==unit and r['candidate_refs']]
        row=max(rows,key=lambda r:bool(r['history_refs']));inputs.append(data.native.inputs(unit,row['id']))
    inputs.extend(data.base.inputs(index) for index in data.base.sampled_draw_rows[:6])
    class Frozen:
        memory_feature_view='all'
        def __init__(self,output):self.model=model;self.text_features=data.native.text;self.output=output
        def decide(self,*unused):return copy.deepcopy(self.output)
    controls={a:config for a,config in plan['arms'].items() if a!='before'}
    counts={a:dict(rows=0,existing_ID_options=0,rank_changed_rows=0) for a in controls};error=0.
    for e in inputs:
        with torch.no_grad():y=model(e)
        n,m=len(e['candidates']),len(e['history']);names=[str(j+1) for j in range(m)]+['NEW','IGNORE','NO_MATCH']
        original=[]
        for i in range(n):
            original.append(dict(probabilities=dict(zip(names,y['match'][i].softmax(-1).tolist())),
                shortlisted_ids=list(range(1,m+1)),update_probability=float(y['update'][i].sigmoid()),
                reliability_probabilities=y['reliability'][i].softmax(-1).tolist(),
                query_membership_probability=float(y['membership'][i].sigmoid()),category_probabilities=y['category'][i].softmax(-1).tolist()))
        life={str(j+1):y['life'][j].softmax(-1).tolist() for j in range(m)}
        frozen=Frozen((original,life,dict(native_probabilities=True,explicit_no_match=True)))
        candidates=[dict(native_region_feature=v.tolist()) for v in e['candidates']]
        tracks=[SimpleNamespace(track_id=j+1,anchor_feature=e['history'][j,0].tolist(),
            recent_feature=e['history'][j,1].tolist(),long_feature=e['history'][j,2].tolist()) for j in range(m)]
        for arm,config in controls.items():
            control=AppearanceRankDecision(frozen,config.get('appearance_projection_checkpoint'),config['appearance_ranking_temperature'])
            predicted,actual_life,metadata=control.decide(None,candidates,tracks,0.,dict(task='mot'))
            assert actual_life==life and metadata['native_probabilities']
            assert metadata['original_action_probabilities_and_probability_multiset_preserved']
            counts[arm]['rank_changed_rows']+=metadata['appearance_rank_changed_rows']
            for before,after in zip(original,predicted):
                bp,ap=before['probabilities'],after['probabilities']
                assert sorted(bp.values())==sorted(ap.values())
                assert all(bp[k]==ap[k] for k in ('NEW','IGNORE','NO_MATCH'))
                assert sorted(bp[str(j+1)] for j in range(m))==sorted(ap[str(j+1)] for j in range(m))
                assert {k:v for k,v in before.items() if k!='probabilities'}=={k:v for k,v in after.items() if k!='probabilities'}
                error=max(error,abs(sum(bp.values())-sum(ap.values())))
                counts[arm]['rows']+=1;counts[arm]['existing_ID_options']+=m
    assert error<=1e-12 and not torch.cuda.is_initialized()
    report=dict(status='REAL_NATIVE_APPEARANCE_RANK_ACTION_VALUE_AND_SEMANTIC_CONSERVATION_VALIDATED',
        real_native_units=61,observation_groups=len(inputs),arms=counts,
        original_action_values_and_full_probability_multisets_exact=True,
        all_auxiliary_semantic_and_lifecycle_outputs_exact=True,total_probability_mass_max_absolute_error=error,
        scope='Representative real original model tensors and actual native ROI features; property proof on identical inputs, not a full video trajectory or fixed FP guarantee.',
        no_new_GPU_or_optimizer_updates=True,full_goal_complete=False)
    (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
