"""Same A3 checkpoint, four actual memory policies and all five official scopes."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.run_global_identity_controls import validate
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    memory=json.loads(args.plan.read_text());plan=json.loads((ROOT/memory['physical_plan']).read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    model=Path(memory['output']);training=json.loads((model/'training_validation.json').read_text())
    assert training['all_A2_physical_parameters_bit_exact']
    source=Path(plan['immutable_source_root']);registered=json.loads((source/plan['evaluation_plan']).read_text())
    root=ROOT/plan['output']/'memory_controls';root.mkdir(exist_ok=False)
    state=dict(status='A3_ALL_FIVE_FOUR_MEMORY_POLICY_CONTROLS_RUNNING',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        same_checkpoint_all_arms_and_tasks=str(model/'checkpoint.pt'),completed={},new_GPU_processes=0,full_goal_complete=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT)+':/data2/user/LocateMOT/third_party/TrackEval')
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    def save():
        p=root/'owner_state.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(root/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    def run(stage,command):
        state['status']=stage;save()
        with gzip.open(root/(stage+'.log.gz'),'wt') as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:f.write(line)
            code=child.wait()
        state.pop('child_pid',None);assert code==0,f'{stage} exited{code}; preserve log and unfinished unit'
    save()
    try:
        for scope,job in registered['scopes'].items():
            manifest=json.loads((source/job['manifest']).read_text())
            for policy in memory['runtime_controls']:
                destination=root/(policy+'_'+scope)
                run(policy+'_'+scope+'_INFERENCE',[python,'tools/groundjev/run_video.py','--task',manifest['task'],
                    '--video',str(source/job['manifest']),'--backend','compact','--compact-device','cpu',
                    '--global-identity','--global-memory-policy',policy,'--checkpoint',str(model/'checkpoint.pt'),
                    '--candidate-replay',str(source/job['candidates']),'--omit-replay-feature-vectors',
                    '--memory-checkpoint-format','binary','--memory-export-interval','0','--out',str(destination)])
                scorer='evaluate_video.py' if manifest['task']=='mot' else 'evaluate_multitask.py'
                run(policy+'_'+scope+'_OFFICIAL',[python,'tools/groundjev/'+scorer,'--video',str(source/job['manifest']),'--run',str(destination)])
                run(policy+'_'+scope+'_EVENTS',[python,'tools/groundjev/analyze_tracking_events.py',
                    '--video',str(source/job['manifest']),'--run',str(destination)])
                record=validate(destination,job,source,model/'checkpoint.pt');flips=0;updates=0;holds=0
                with (destination/'frame_decisions.jsonl').open() as f:
                    for row in map(json.loads,f):
                        report=row['state']['decision_metadata']['assigned_identity_update']
                        assert report['policy']==policy;flips+=report['update_HOLD_flips_vs_original']
                        for event in row['state']['events']:
                            if 'memory_action' in event:
                                updates+=event['memory_action']=='UPDATE';holds+=event['memory_action']=='HOLD'
                state['completed'][policy+'_'+scope]=dict(record,actual_assigned_UPDATE_HOLD_flips_vs_original=flips,
                    long_memory_UPDATE_events=updates,long_memory_HOLD_events=holds,
                    recent_observational_memory='Original behavior remains; UPDATE/HOLD gates protected long memory.')
                save()
        state['status']='A3_ALL20_REGISTERED_MEMORY_POLICY_UNITS_COMPLETE';save()
        (root/'all_five_scopes.json').write_text(json.dumps(state,indent=2)+'\n')
    except Exception as error:
        state.update(status='A3_MEMORY_CONTROL_OWNER_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
