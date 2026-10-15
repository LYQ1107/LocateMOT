"""One blocking owner serializes matched old/refreshed-history SFT on GPU4."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    study=json.loads((args.study/'summary.json').read_text())
    preflight=json.loads((args.preflight/'owner_state.json').read_text())
    assert preflight['status']=='BOTH_HISTORY_REFRESH_OFFICIAL_CPU_PLANS_COMPLETE'
    assert preflight['paired_every_step_task_and_question_family_exact']
    assert study['combined_registered_upper_bytes']<study['global_temporary_feature_budget_bytes']
    previous=json.loads((ROOT/'outputs/groundjev/training_data/native600_fit_rollout_refresh_v1/owner_state.json').read_text())
    assert previous['status']=='CURRENT_NATIVE600_FIT48_REAL_ROLLOUT_COMPLETE' and previous['rollout_exit_code']==0
    master=json.loads((ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/latest_training_state.json').read_text())
    assert master['step']==600 and master['FP32_masters_and_Adam_moments_saved']
    args.out.mkdir(parents=True,exist_ok=False)
    state={'status':'WAITING_GPU4_FOR_PAIRED_HISTORY_REFRESH600',
        'owner_pid':os.getpid(),'owner_starttick':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'owner_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'physical_gpus':[4],'serial_order':['old','refreshed'],
        'new_stage_updates_each':600,'new_stage_samples_each':2400,
        'same_initial_FP32_state':study['initial_FP32_state'],
        'same_per_step_task_and_question_families':True,'changed_native_history_rows_each':400,
        'token_or_FLOP_matched':False,'completed':{},'new_training_started':False,
        'official_tracking_metrics':'pending; training and reload do not establish tracking accuracy',
        'ordinary_model_performance_achieved':False,'full_goal_complete':False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:v for k,v in state.items() if k!='completed'}),flush=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'external/OneJev'),WORLD_SIZE='1',RANK='0',LOCAL_RANK='0')
    def run(stage,command):
        state.update(status=stage,current_command=command)
        save()
        with (args.out/(stage.lower()+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['current_child_pid']=child.pid;save()
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        state['last_exit_code']=code
        state.pop('current_child_pid',None)
        if code:
            raise RuntimeError(f'{stage} exited{code}; retain prefix and latest valid full training state')
    prefix=['python3','tools/groundjev/run_with_resources.py','--gpus','4','--',str(ROOT/'.venv-onejev-py310/bin/python')]
    save()
    try:
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            if free>=45*1024:
                state['GPU4_free_MiB_before_launch']=free;break
            time.sleep(30)
        for name in state['serial_order']:
            config=json.loads(Path(study['configs'][name]).read_text())
            training=Path(config['runs_dir'])/config['name']
            assert not training.exists(),'Preserve prior training; resume its latest valid cursor explicitly'
            state['new_training_started']=True
            run(name+'_FULL600_TRAINING',prefix+['tools/groundjev/onejev_resumable_sft.py',
                '--config',study['configs'][name],'--paired-index-shuffle',
                '--initial-training-state',study['initial_FP32_state'],'--expected-initial-step','600'])
            records=[json.loads(line) for line in (training/'log.jsonl').open()]
            done=next(row for row in records if row.get('event')=='done')
            assert done['steps']==600 and done['examples']==2400
            assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in records if 'loss' in row)
            initial=json.loads((training/'initial_fp32_state_receipt.json').read_text())
            assert initial['source_step']==600 and initial['initialization_max_difference']==0
            latest=json.loads((training/'latest_training_state.json').read_text())
            assert latest['step']==600 and latest['seen']==2400
            run(name+'_NATIVE_MM_RELOAD',prefix+['tools/groundjev/check_onejev125_checkpoint.py',
                '--step','600','--checkpoint',str(training/'ckpt-step600'),
                '--training-root',str(training),'--reference-checkpoint',config['model']])
            validation=json.loads((training/'checkpoint600_validation.json').read_text())
            assert validation['status']=='REAL_ONEJEV600_NATIVE_MM_RELOAD_VALIDATED'
            state['completed'][name]={'training':str(training),'done':done,'initial_FP32':initial,
                'latest_training_state':latest,'native_MM_reload':validation}
            save()
        state.update(status='PAIRED_HISTORY_REFRESH600_TRAINING_AND_MM_RELOAD_COMPLETE')
        state.pop('current_command',None);save()
    except Exception as error:
        state.update(status='PAIRED_HISTORY_REFRESH600_OWNER_FAILED',error=repr(error));save()
        raise


if __name__=='__main__':
    main()
