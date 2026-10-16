"""One blocking CPU owner for appearance rank conservation and causal controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan=json.loads(args.plan.read_text())
    assert plan['combined_registered_feature_upper_bytes']<plan['global_temporary_feature_budget_bytes']
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='REGISTERED_APPEARANCE_RANK_STUDY',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        completed=[],new_GPU_or_optimizer_updates=0,full_goal_complete=False)
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'external/OneJev'))
    def save():
        temp=args.out/'owner_state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.out/'owner_state.json')
        print(json.dumps(state),flush=True)
    def run(stage,command):
        state.update(status=stage,command=command);save()
        with (args.out/(stage+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=child.wait()
        assert code==0,f'{stage} exited{code}; retain first failure'
        state.pop('child_pid',None);state['completed'].append(stage);save()
    save()
    try:
        run('REAL_ACTION_PROBABILITY_CONSERVATION',[python,'tools/groundjev/validate_appearance_ranks.py',
            '--plan',str(args.plan),'--out',str(args.out/'regression')])
        proof=json.loads((args.out/'regression/validation.json').read_text())
        assert proof['original_action_values_and_full_probability_multisets_exact']
        run('ALL_FIVE_CAUSAL_CONTROLS',[python,'tools/groundjev/run_fixed_affinity_controls.py',
            '--plan',str(args.plan),'--out',str(args.out/'controls')])
        result=json.loads((args.out/'controls/owner_state.json').read_text())
        assert result['status']=='ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(result['completed'])==20
        state.update(status='APPEARANCE_RANK_CONSERVATION_AND_ALL_FIVE_OFFICIAL_CONTROLS_COMPLETE');save()
    except Exception as error:
        state.update(status='APPEARANCE_RANK_STUDY_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
