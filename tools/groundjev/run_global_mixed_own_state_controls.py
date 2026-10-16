"""Fixed final own-state model: all five scopes with actual assigned memory."""
import argparse
import gzip
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


from tools.groundjev.run_global_identity_controls import validate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=Path(plan['immutable_source_root']);out=ROOT/plan['output'];owner=out/'A4b_mixed_owner';owner.mkdir(exist_ok=False)
    registered=json.loads((source/plan['evaluation_plan']).read_text())
    training=json.loads((Path(plan['model_output'])/'A4b_mixed_own_state_v2/training_validation.json').read_text())
    assert training['status']=='MIXED_MOT_RMOT_REAL_8_16_32_OWN_STATE_TRAINING_COMPLETE'
    assert training['counts']['frames']==14052
    from tools.groundjev.validate_final_global_identity import validate as validate_final
    final_validation=validate_final(Path(plan['model_output'])/'A4b_mixed_own_state_v2/checkpoint.pt',plan)
    (out/'A4b_final_runtime_validation.json').write_text(json.dumps(final_validation,indent=2)+'\n')
    state=dict(status='A4B_MIXED_OWN_STATE_FIXED_FINAL_FIVE_SCOPES_RUNNING',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        completed={},new_GPU_processes=0,full_goal_complete=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(ROOT)+':/data2/user/LocateMOT/third_party/TrackEval')
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    def save():
        path=owner/'owner_state.tmp';path.write_text(json.dumps(state,indent=2)+'\n');path.replace(owner/'owner_state.json')
        print(json.dumps(dict(status=state['status'],completed=list(state['completed']))),flush=True)
    def run(stage,command):
        state['status']=stage;save()
        with gzip.open(owner/(stage+'.log.gz'),'wt') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line)
            code=child.wait()
        state.pop('child_pid',None)
        assert code==0,f'{stage} exited{code}; complete compressed log and unfinished source prefix retained'
    save()
    try:
        for scope,job in registered['scopes'].items():
            manifest=json.loads((source/job['manifest']).read_text())
            for arm in ('A4b_mixed_own_state_v2',):
                checkpoint=Path(plan['model_output'])/arm/'checkpoint.pt';assert checkpoint.is_file()
                destination=out/'mixed_own_state_controls'/(arm+'_'+scope)
                if not (destination/'result.json').exists():
                    assert not destination.exists(),f'Unfinished unit {destination}; resume its retained causal prefix explicitly'
                    run(arm+'_'+scope+'_INFERENCE',[python,'tools/groundjev/run_video.py','--task',manifest['task'],
                        '--video',str(source/job['manifest']),'--backend','compact','--compact-device','cpu',
                        '--global-identity','--global-memory-policy','assigned','--checkpoint',str(checkpoint),'--candidate-replay',str(source/job['candidates']),
                        '--omit-replay-feature-vectors','--memory-checkpoint-format','binary','--memory-export-interval','0','--out',str(destination)])
                if not (destination/'official_metrics.json').exists():
                    scorer='evaluate_video.py' if manifest['task']=='mot' else 'evaluate_multitask.py'
                    run(arm+'_'+scope+'_OFFICIAL',[python,'tools/groundjev/'+scorer,'--video',str(source/job['manifest']),'--run',str(destination)])
                run(arm+'_'+scope+'_EVENTS',[python,'tools/groundjev/analyze_tracking_events.py',
                    '--video',str(source/job['manifest']),'--run',str(destination)])
                from tools.groundjev.analyze_global_identity import analyze
                record=validate(destination,job,source,checkpoint)
                record['identity_diagnostics']=analyze(manifest,destination)
                state['completed'][arm+'_'+scope]=record;save()
        state['status']='A4B_MIXED_OWN_STATE_ALL_FIVE_REGISTERED_SCOPES_COMPLETE';save()
        (out/'A4b_mixed_all_five_scopes.json').write_text(json.dumps(state,indent=2)+'\n')
    except Exception as error:
        state.update(status='A4B_MIXED_OWN_STATE_OWNER_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
