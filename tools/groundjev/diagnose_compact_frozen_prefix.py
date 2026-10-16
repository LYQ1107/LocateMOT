"""Held-out single-step decisions on exact original-actor prefixes, not rollouts."""
import argparse
from collections import Counter, defaultdict
import gzip
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def assignments(probabilities, identities):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n,m=len(probabilities),len(identities);actions=['NEW','IGNORE','NO_MATCH']
    costs=np.full((n,m+3*n),1e9)
    for i,distribution in enumerate(probabilities):
        for j,identity in enumerate(identities):
            if str(identity) in distribution:costs[i,j]=-math.log(max(distribution[str(identity)],1e-12))
        for j,action in enumerate(actions):costs[i,m+3*i+j]=-math.log(max(distribution[action],1e-12))
    rr,cc=linear_sum_assignment(costs) if n else ([],[])
    result=[None]*n
    for i,j in zip(rr,cc):result[i]=str(identities[j]) if j<m else actions[(j-m)%3]
    assert all(v is not None for v in result)
    return result


def main():
    import numpy as np
    import torch
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.compact import CompactCore,CompactDecision
    from locatemot.groundjev.decision import overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.analyze_tracking_events import load_truth
    from tools.groundjev.run_video import ReplayFrames
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    torch.set_num_threads(plan['cpu_threads'])
    manifest=json.loads((ROOT/plan['manifest']).read_text())
    previous=ROOT/plan['original_run'];runtime=json.loads((previous/'result.json').read_text())
    assert runtime['decision_device']=='cpu' and runtime['memory_policy']=='gated' and runtime['lost_seconds']==3
    assert len(manifest['images'])==runtime['frames']==plan['expected_frames']
    truth,_=load_truth(manifest)
    class Capture(CompactDecision):
        def accepted_batch(self,group,ranked,inputs):
            self.captured.append(dict(inputs=inputs,ids=[t.track_id for t in ranked]))
    actor=Capture(Path(runtime['checkpoint']),device='cpu')
    actor.model.register_forward_hook(lambda module,inputs,outputs: actor.captured[-1].update(outputs=outputs))
    tracker=OnlineTracker(actor);models={}
    for arm,path in plan['checkpoints'].items():
        source=torch.load(path,weights_only=True,map_location='cpu')
        model=CompactCore(source['architecture'],semantic=source['semantic'],
            input_evidence=source.get('input_evidence','geometry_appearance_language'),
            native_relation_residual=source.get('native_relation_residual',False))
        model.load_state_dict(source['state_dict']);model.eval();models[arm]=model
    for arm,condition in plan.get('input_interventions',{}).items():
        models[arm]=models[condition['model']]
    arms=['original',*models];counts={arm:Counter() for arm in arms};confusion={arm:Counter() for arm in arms}
    births={};totals=Counter();history_domain=[];correct_history_domain=[]
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='EVALUATING_IDENTICAL_OLD_ACTOR_VALIDATION_PREFIXES',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        frames=0,candidates=0,new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save();replay=ReplayFrames(Path(runtime['shared_input_features_source']))
    specification=dict(task='mot',vocabulary=['person'],expression=None)
    try:
        with (previous/'frame_decisions.jsonl').open() as trace,gzip.open(out/'decisions.jsonl.gz','wt') as report:
            for expected,line in zip_longest(zip(manifest['frame_numbers'],manifest['images']),trace):
                assert expected is not None and line is not None
                frame,path=expected;saved=json.loads(line);raw=replay.frame(frame);candidates=raw['candidates']
                assert saved['frame']==raw['frame']==frame and saved['image_path']==raw['image_path']==path
                before=dict(births);identities=list(tracker.tracks);actor.captured=[]
                with Image.open(path) as source:image=source.convert('RGB')
                outputs,actual_state=tracker.step(image,candidates,frame/manifest['fps'],specification)
                assert outputs==saved['outputs'] and actual_state==saved['state'],('original prefix',frame)
                events={e['candidate']:e for e in actual_state['events'] if 'candidate' in e}
                gt=truth[frame];matched={}
                if candidates and gt:
                    matrix=np.asarray([[overlap(c['box'],g['box']) for g in gt] for c in candidates])
                    rr,cc=linear_sum_assignment(-matrix)
                    matched={int(i):gt[int(j)]['identity'] for i,j in zip(rr,cc) if matrix[i,j]>=.5}
                distributions={arm:[] for arm in arms};shortlists=[];scalar_by_id={}
                for batch in actor.captured:
                    n=len(batch['inputs']['candidates']);names=[str(t) for t in batch['ids']]+['NEW','IGNORE','NO_MATCH']
                    shortlists.extend([batch['ids']]*n)
                    values=batch['inputs']['history_geo'][:,4:].tolist();history_domain.extend(values)
                    scalar_by_id.update(zip(batch['ids'],values))
                    with torch.no_grad():
                        results={'original':batch['outputs']}
                        for arm,model in models.items():
                            e=batch['inputs'];condition=plan.get('input_interventions',{}).get(arm)
                            if condition:
                                columns=condition['zero_history_geo_columns']
                                assert columns and set(columns)<={4,5,6,7}
                                e=dict(e,history_geo=e['history_geo'].clone())
                                e['history_geo'][:,columns]=0.
                            results[arm]=model(e)
                            if arm in plan.get('expect_original_outputs_exact',[]):
                                assert all(torch.equal(results[arm][key],batch['outputs'][key]) for key in batch['outputs'])
                    for arm,y in results.items():
                        distributions[arm].extend(dict(zip(names,p)) for p in y['match'].softmax(-1).tolist())
                assert len(shortlists)==len(candidates) and all(len(v)==len(candidates) for v in distributions.values())
                choices={arm:assignments(distributions[arm],identities) for arm in arms}
                for i in range(len(candidates)):
                    event=events[i];actual=str(event['track_id']) if event['action'] in ('ACTIVE','REACTIVATE') else event['action']
                    assert choices['original'][i]==actual,('original global assignment',frame,i)
                    identity=matched.get(i)
                    correct={str(t) for t in shortlists[i] if identity is not None and before.get(t)==identity}
                    known=any(identity is not None and before.get(t)==identity for t in identities)
                    target='IGNORE' if identity is None else 'EXISTING' if correct else 'NO_MATCH' if known else 'NEW'
                    acceptable=correct or {target};totals['target:'+target]+=1
                    totals['multiple_GT_consistent_aliases']+=len(correct)>1
                    correct_history_domain.extend(scalar_by_id[int(t)] for t in sorted(correct))
                    case=dict(frame=frame,candidate=i,target=target,correct_aliases=sorted(correct),predictions={})
                    for arm in arms:
                        distribution=distributions[arm][i];choice=max(distribution,key=distribution.get)
                        assigned=choices[arm][i];kind=choice if choice in ('NEW','IGNORE','NO_MATCH') else 'EXISTING'
                        confusion[arm][target+'->'+kind]+=1
                        counts[arm]['questions']+=1;counts[arm]['argmax_correct']+=choice in acceptable
                        counts[arm]['shadow_assignment_correct']+=assigned in acceptable
                        if target=='EXISTING':
                            counts[arm]['correct_existing_option_present']+=1
                            counts[arm]['argmax_existing_correct']+=choice in correct
                            counts[arm]['shadow_existing_correct']+=assigned in correct
                        case['predictions'][arm]=dict(argmax=choice,shadow_assignment=assigned,
                            correct_probability=sum(distribution.get(label,0.) for label in acceptable))
                    report.write(json.dumps(case,separators=(',',':'))+'\n')
                for i,event in events.items():
                    if event['action']=='NEW':births[event['track_id']]=matched.get(i)
                state['frames']+=1;state['candidates']+=len(candidates)
                if state['frames']%100==0:save()
        replay.close()
        assert state['frames']==plan['expected_frames'] and state['candidates']==plan['expected_candidates']
        def domain(values):
            return dict(observations=len(values),fields=['age/100','lost_age/3','seconds_since_reliable/3','confidence'],
                quantile_levels=[0,.5,.95,1],quantiles=np.quantile(values,[0,.5,.95,1],axis=0).tolist())
        result=dict(status='DT1203_IDENTICAL_REAL_OLD_PREFIXES_AND_SHADOW_ASSIGNMENTS_VALIDATED',
            frames=state['frames'],candidates=state['candidates'],original_every_output_state_and_assignment_exact=True,
            same_actual_native_tensor_inputs_for_all_models=not bool(plan.get('input_interventions')),
            native_visual_box_pair_geometry_and_specification_inputs_identical=True,
            explicitly_declared_history_scalar_interventions_only=bool(plan.get('input_interventions')),
            all_declared_untrained_model_output_equalities_exact=True,posthoc_target_counts=dict(totals),
            counts={arm:dict(c) for arm,c in counts.items()},argmax_confusion={arm:dict(c) for arm,c in confusion.items()},
            models=plan['checkpoints'],input_interventions=plan.get('input_interventions',{}),
            actual_old_prefix_history_scalar_domain=domain(history_domain),
            posthoc_correct_existing_alias_scalar_domain=domain(correct_history_domain),
            GT_used_only_by_external_posthoc_scoring=True,
            GT_actor_state_repair=False,original_actor_is_only_evolving_history=True,
            scope=plan['comparison'],new_GPU_or_optimizer_updates=0,official_tracking_gain_claimed=False,full_goal_complete=False)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<2*2**20
        assert not torch.cuda.is_initialized()
        state.update(status=result['status']);save();print(json.dumps(result),flush=True)
    except Exception as error:
        state.update(status='FROZEN_PREFIX_DECISION_DIAGNOSIS_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
