"""One blocking owner for replaying the interrupted original 500-step unit."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def main():
    name='onejev_own_history_v3_exact500_replay_stateful'
    out=ROOT/'outputs/groundjev/training'/name;out.mkdir(parents=True,exist_ok=False)
    state={'status':'REPLAYING_INTERRUPTED_ORIGINAL500_STEP_UNIT','owner_pid':os.getpid(),
        'physical_gpu':4,'target_steps':500,'original_interrupted_log_step':275,
        'original_BF16_weight_snapshot_step':250,'optimizer_reset_from250':False,
        'initialization':'original valid100 checkpoint, identical original500 protocol',
        'scope':'unfinished-unit replay; not exact reconstruction of missing original250Adam state',
        'new_checkpoint_policy':'one atomic FP32masters/Adam/scheduler/RNG/cursor training state'}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save();env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'external/OneJev')
    command=['python3','tools/groundjev/run_with_resources.py','--gpus','4','--',
        str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/onejev_resumable_sft.py',
        '--config','configs/groundjev/'+name+'.json']
    with (out/'owner_console.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        state.update(status='RUNNING_STATEFUL_OFFICIAL500_PROTOCOL',child_pid=child.pid);save()
        for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        code=child.wait()
    state['exit_code']=code
    if code:
        state['status']='STATEFUL_ONEJEV500_FAILED';save();raise RuntimeError('Preserve failure; resume latest complete training state')
    logs=[json.loads(x) for x in (out/'log.jsonl').read_text().splitlines()]
    done=next(x for x in logs if x.get('event')=='done');assert done['steps']==500
    state.update(status='STATEFUL_ONEJEV500_TRAINING_COMPLETE',done=done,
        checkpoint_reload_before_after_official_results='pending');save()


if __name__=='__main__':main()
