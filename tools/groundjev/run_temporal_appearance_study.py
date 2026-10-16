"""One blocking CPU owner for contrastive physical appearance and all controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    out=ROOT/plan['owner_output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='REGISTERED_TEMPORAL_APPEARANCE_STUDY',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        completed=[],CPU_only=True,new_physical_GPUs=[],full_goal_complete=False)
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'external/OneJev'))
    def save():
        temp=out/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'owner_state.json')
        print(json.dumps(state),flush=True)
    def run(stage,command):
        state.update(status=stage,command=command);save()
        with (out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        assert code==0,f'{stage} exited{code}'
        state.pop('child_pid',None);state['completed'].append(stage);save()
    def equal(a,b):
        assert type(a) is type(b)
        if torch.is_tensor(a):assert a.dtype==b.dtype and torch.equal(a,b) and torch.isfinite(a).all()
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal(x,y)
        else:assert a==b
    save()
    try:
        run('BUILD_TRAIN_IDENTITY_REFERENCES',[python,'tools/groundjev/temporal_identity_data.py','--plan',str(args.plan)])
        prefix=Path(plan['training_output']+'_prefix')
        for branch,stop,resume in [('continuous',8,False),('restored',4,False),('restored',8,True)]:
            command=[python,'tools/groundjev/train_temporal_appearance.py','--plan',str(args.plan),
                '--out',str(prefix/branch),'--stop-after',str(stop)]
            if resume:command.append('--resume')
            run(f'PREFIX_{branch}_{stop}',command)
        a=torch.load(prefix/'continuous/latest_training_state.pt',weights_only=False,map_location='cpu')
        b=torch.load(prefix/'restored/latest_training_state.pt',weights_only=False,map_location='cpu')
        for field in ['protocol','step','observations','state_dict','optimizer','torch_rng','sampler_rng','unit_batch_exposure']:equal(a[field],b[field])
        proof=dict(status='REAL_TEMPORAL_IDENTITY_8_VS_4_PLUS_4_RESUME_BIT_EXACT',actual_steps=8,
            actual_ROI_exposures_each_branch=a['observations'],all_model_Adam_RNG_sampler_and_exposure_counters_exact=True,
            full1000_training_complete=False,new_GPU_processes=0,full_goal_complete=False)
        (out/'prefix_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
        run('CONTRASTIVE1000_TRAINING',[python,'tools/groundjev/train_temporal_appearance.py','--plan',str(args.plan)])
        training=Path(plan['training_output']);result=json.loads((training/'owner_state.json').read_text())
        assert result['status']=='TEMPORAL_APPEARANCE_CONTRASTIVE_TRAINING_COMPLETE' and result['step']==1000
        (out/'completed_training_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        run('REAL_AFFINITY_REGRESSION',[python,'tools/groundjev/validate_temporal_appearance_control.py',
            '--plan',str(args.plan),'--out',str(out/'runtime_validation')])
        original=json.loads((ROOT/plan['fixed_affinity_protocol']).read_text())
        refs=out/'reference_runs';refs.mkdir()
        for scope,source in original['before_CPU_reference_runs'].items():
            (refs/(scope+'_before')).symlink_to((ROOT/source).resolve(),target_is_directory=True)
        fixed=dict(association='fixed_iou_cosine',auxiliary='fixed',memory_policy='gated',feature_view='all',lost_seconds=3.,shortlist=32)
        controls=dict(regression=original['control_regression'],checkpoint=plan['initial_checkpoint'],
            reference_root=str(refs),scopes=original['scopes'],cpu_blas_threads=4,
            arms=dict(before=dict(association='learned',auxiliary='learned',reference_arm='before'),
                native_cosine=fixed,initial_projection=dict(fixed,appearance_projection_checkpoint=str(training/'initial_projection.pt')),
                trained_projection=dict(fixed,appearance_projection_checkpoint=str(training/'projection.pt'))),
            comparison_limit=plan['controls']+' '+plan['checkpoint_selection'],memory_storage=plan['feature_storage'])
        path=out/'resolved_controls_plan.json';path.write_text(json.dumps(controls,indent=2)+'\n')
        run('ALL_FIVE_OFFICIAL_CONTROLS',[python,'tools/groundjev/run_fixed_affinity_controls.py','--plan',str(path),'--out',str(out/'controls')])
        final=json.loads((out/'controls/owner_state.json').read_text())
        assert final['status']=='ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(final['completed'])==20
        state.update(status='TEMPORAL_APPEARANCE1000_AND_ALL_FIVE_OFFICIAL_CONTROLS_COMPLETE');save()
    except Exception as error:
        state.update(status='TEMPORAL_APPEARANCE_STUDY_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
