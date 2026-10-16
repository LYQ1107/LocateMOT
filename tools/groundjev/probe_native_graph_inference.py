"""Actual native Torch graph paths on fixed real tracking/category requests."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'external/OneJev')]


def cases():
    from PIL import Image
    from locatemot.groundjev.decision import OneJevDecision,SimpleDecision,overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from locatemot.groundjev.category import NativeCategoryHead
    from tools.groundjev.run_video import restore_completed_prefix,ReplayFrames
    result=[]
    sources=[('sparse_MOT',ROOT/'outputs/groundjev/runs/mot17_first_native_clip_repair2',
        ROOT/'outputs/groundjev/pilot_data/mot17_first4.json',1,3),
        ('dense_MOT20',ROOT/'outputs/groundjev/natural_memory/mot20_05_full_validation_gpu5_v1/controls/mot20_challenge_reference_gated3',
        ROOT/'outputs/groundjev/recovery_scope_20261015/registered_mot_challenge.json',16,32)]
    for name,source,manifest_path,n,m in sources:
        manifest=json.loads(manifest_path.read_text());runtime=json.loads((source/'result.json').read_text())
        with (source/'frame_decisions.jsonl').open() as f:rows=[json.loads(next(f)) for _ in range(2)]
        if runtime.get('replay_feature_vectors_omitted') or not any('native_region_feature' in c for c in rows[0]['candidates']):
            native_path=runtime.get('shared_input_features_source')
            if native_path:
                replay=ReplayFrames(Path(native_path))
                for row in rows:row['candidates']=replay.frame(row['frame'])['candidates']
                replay.close()
        tracker=OnlineTracker(SimpleDecision(),lost_seconds=runtime.get('lost_seconds',3.),memory_policy=runtime.get('memory_policy','gated'))
        specification=dict(task='mot',vocabulary=['person'],expression=None)
        restore_completed_prefix(tracker,rows[:1],list(zip(manifest['images'],manifest['frame_numbers']))[:1],manifest['fps'],specification)
        with Image.open(rows[1]['image_path']) as f:image=f.convert('RGB')
        group=rows[1]['candidates'][:n];timestamp=rows[1]['frame']/manifest['fps']
        ranked=sorted(tracker.tracks.values(),key=lambda t:(-max((overlap(c['box'],t.predicted_box(timestamp)) for c in group),default=0.),t.track_id))[:m]
        builder=OneJevDecision.__new__(OneJevDecision);builder.evidence_format='v3_crop_motion'
        request,media=builder.build_request(image,group,ranked,timestamp,specification)
        result.append((name,request,media,dict(source=str(source),frame=rows[1]['frame'],candidates=len(group),history=len(ranked))))
    source=ROOT/'outputs/groundjev/compact/physical_factorization_pair8000_v1/canonical_study/controls/tao_raw_full40_broader'
    manifest=json.loads((ROOT/'outputs/groundjev/multitask_pilot_data/ovmot_tao_base_novel63_full.json').read_text())
    with (source/'frame_decisions.jsonl').open() as f:row=json.loads(next(f))
    with Image.open(row['image_path']) as f:image=f.convert('RGB')
    head=NativeCategoryHead(None);batch=row['candidates'][:8]
    request,media,groups=head.build_root_request(image,batch,manifest['vocabulary'])
    result.append(('OV1203_root',request,media,dict(source=str(source),frame=row['frame'],objects=len(batch),supplied_classes=1203)))
    leaf=head.build_leaf_request(request.state,groups,{i:'0' for i in range(len(batch))})
    result.append(('OV128_fixed_leaf',leaf,media,dict(source=str(source),frame=row['frame'],objects=len(batch),
        group=0,scope='Fixed first vocabulary group for paired numerical/compute probe; not GT selection or end-to-end hierarchical classification.')))
    return result


def compare(a,b):
    maximum=0.;choices=True
    assert type(a) is type(b)
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:
            delta,exact=compare(a[key],b[key]);maximum=max(maximum,delta);choices&=exact
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            delta,exact=compare(x,y);maximum=max(maximum,delta);choices&=exact
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert math.isfinite(a) and math.isfinite(b);maximum=abs(a-b)
    else:choices=a==b
    return maximum,choices


def main():
    import torch
    from qev.mm_engine import MMDecisionEngine,DecoderGraphs,ForkGraphs
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(plan['physical_gpu'])=='7'
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    torch.set_num_threads(4)
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='PREPARING_REAL_NATIVE_GRAPH_PROBE',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],physical_gpu=7,
        completed={},new_optimizer_updates=0,old_jobs_or_backends_changed=False,full_goal_complete=False)
    def save():
        temp=out/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    save()
    try:
        requests=cases();state['request_cases']={name:details for name,_,_,details in requests};save()
        import subprocess
        state.update(status='WAITING_GPU7_FREE_MEMORY');save()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--id=7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=plan['minimum_free_MiB']:break
            time.sleep(20)
        state['launch_free_MiB']=free
        for arm in ('eager','graphs'):
            state.update(status=arm+'_LOAD_NATIVE_CHECKPOINT');save()
            engine=MMDecisionEngine(str(ROOT/plan['checkpoint']),device='cuda:0',dtype='bfloat16',cuda_graphs=False)
            if arm=='graphs':
                decoder=engine.model.model.language_model
                engine._graphs=DecoderGraphs(decoder,max_len=plan['graph_max_prefix'],max_graphs=1)
                engine._fork_graphs=ForkGraphs(decoder,engine._fork,engine.pad_id,max_len=plan['graph_max_prefix'],
                    max_width=plan['graph_max_suffix'],max_rows=plan['graph_max_rows'],max_graphs=1)
            calls=[];original_graph_call=engine._graph_call
            def graph_call(name,call):
                value=original_graph_call(name,call);calls.append(dict(path=name,used=value is not None));return value
            engine._graph_call=graph_call
            arm_results={}
            for name,request,media,details in requests:
                state.update(status=arm+'_'+name);save()
                engine.decide(request,media=media);torch.cuda.synchronize()
                answers=[];elapsed=[];hits=[]
                for repeat in range(plan['timed_repeats']):
                    calls.clear();torch.cuda.synchronize();start=time.perf_counter()
                    response,_=engine.decide(request,media=media);torch.cuda.synchronize()
                    elapsed.append(time.perf_counter()-start)
                    answers.append({k:v.model_dump() for k,v in response.answers.items()});hits.append(list(calls))
                    print(json.dumps(dict(arm=arm,case=name,repeat=repeat,seconds=elapsed[-1],graph_calls=hits[-1])),flush=True)
                repeat_differences=[compare(answers[0],x)[0] for x in answers[1:]]
                arm_results[name]=dict(answers=answers[0],seconds=elapsed,median_seconds=statistics.median(elapsed),
                    repeated_answer_max_difference=max(repeat_differences,default=0.),timed_graph_calls=hits,request=details,
                    fork_mode=engine.info.fork_mode,fork_max_abs_diff=engine.info.fork_max_abs_diff)
                state['completed'][arm+'_'+name]=dict(median_seconds=statistics.median(elapsed));save()
            (out/(arm+'_results.json')).write_text(json.dumps(arm_results,indent=2)+'\n')
            # Drop the bound-method instrumentation cycle before loading the other engine.
            engine._graph_call=original_graph_call;del original_graph_call,graph_call,engine
            gc.collect();torch.cuda.empty_cache()
        eager=json.loads((out/'eager_results.json').read_text());graphs=json.loads((out/'graphs_results.json').read_text())
        result={}
        for name in eager:
            delta,choices=compare(eager[name]['answers'],graphs[name]['answers'])
            used=all(calls and all(call['used'] for call in calls) for calls in graphs[name]['timed_graph_calls'])
            ratio=eager[name]['median_seconds']/graphs[name]['median_seconds']
            result[name]=dict(max_native_answer_absolute_difference=delta,all_discrete_answers_equal=choices,
                actual_graph_paths_used_for_every_timed_call=used,steady_component_speed_ratio=ratio,
                eager_median_seconds=eager[name]['median_seconds'],graph_median_seconds=graphs[name]['median_seconds'])
        candidate=all(r['max_native_answer_absolute_difference']<=plan['native_probability_tolerance'] and
            r['all_discrete_answers_equal'] and r['actual_graph_paths_used_for_every_timed_call'] for r in result.values())
        report=dict(status='REAL_NATIVE_GRAPH_INFERENCE_COMPONENT_PROBE_COMPLETE',cases=result,
            numerical_and_actual_graph_path_candidate=candidate,
            measured_speed_gate_passed=all(r['steady_component_speed_ratio']>=plan['minimum_component_speed_ratio'] for r in result.values()),
            adopted_by_original_jobs=False,official_native_Torch_graph_classes_no_vendor_or_model_edits=True,
            scope='Four fixed legal real-image tracking/category requests; warmup excluded, three timed repeats, shared GPU/concurrent jobs. Not fullvideo trajectory parity, sole-cause speed, FPS or E2E performance.',
            new_optimizer_updates=0,full_goal_complete=False)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        state.update(status=report['status']);save();print(json.dumps(report),flush=True)
    except Exception as error:
        state.update(status='NATIVE_GRAPH_COMPONENT_PROBE_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
