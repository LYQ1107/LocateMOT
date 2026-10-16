"""Single blocking CPU owner: real parity, matched prefixes and paired full studies."""
import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    args.out.mkdir(parents=True,exist_ok=False)
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'external/OneJev'))
    plans={a:ROOT/f'configs/groundjev/compact_physical_{a}_semantic8000_v1.json' for a in ('coupled','canonical')}
    configs={a:json.loads(p.read_text()) for a,p in plans.items()}
    differences={k for k in set(configs['coupled'])|set(configs['canonical'])
        if configs['coupled'].get(k)!=configs['canonical'].get(k)}
    assert differences=={'name','canonical_physical_core','training_output','training_prefix_validation'}
    state=dict(status='REGISTERED_MATCHED_PHYSICAL_FACTORIZATION_PAIR',owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        plans={a:str(p) for a,p in plans.items()},registered_differences=sorted(differences),
        CPU_only=True,physical_GPUs=[],same_original_training_and_evaluation_jobs_untouched=True,
        completed=[],full_goal_complete=False)
    (args.out/'registration.json').write_text(json.dumps(dict(configs=configs,**state),indent=2)+'\n')
    def save():
        temporary=args.out/'owner_state.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(args.out/'owner_state.json');print(json.dumps(state),flush=True)
    def run(name,command):
        state.update(status=name,command=command);save()
        with (args.out/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(name);save()
    save()
    try:
        run('REAL_NATIVE_INVARIANCE',[python,'tools/groundjev/validate_canonical_physical_core.py',
            '--plan',str(plans['canonical']),'--out',str(args.out/'initial')])
        run('FULL_DT1203_INITIAL_PARITY',[python,'tools/groundjev/diagnose_compact_frozen_prefix.py',
            '--plan','configs/groundjev/canonical_physical_initial_DT1203_v1.json'])
        parity=json.loads((ROOT/'outputs/groundjev/diagnostics/canonical_physical_initial_DT1203_v1/summary.json').read_text())
        assert parity['original_every_output_state_and_assignment_exact'] and parity['all_declared_untrained_model_output_equalities_exact']
        for arm,plan in plans.items():
            run(arm+'_PREFIX',[python,'tools/groundjev/validate_compact_question_training.py',
                '--plan',str(plan),'--out','/data3/user/LocateMOT-checkpoints/physical_factorization_'+arm+'_prefix_v1'])
        # Independent matched studies use eight CPU threads combined, no GPU allocation.
        children={};logs={};multiplexer=selectors.DefaultSelector()
        state.update(status='PAIRED_MATCHED_STUDIES_RUNNING');save()
        for arm,plan in plans.items():
            command=[python,'tools/groundjev/run_compact_question_study.py','--plan',str(plan),
                '--out',str(args.out/(arm+'_study'))]
            if arm=='canonical':command+=['--initial-parity-proof',str(args.out/'initial/validation.json')]
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,bufsize=1)
            children[arm]=child;logs[arm]=(args.out/(arm+'_study.log')).open('w')
            multiplexer.register(child.stdout,selectors.EVENT_READ,data=arm)
        state['children']={a:c.pid for a,c in children.items()};save()
        while multiplexer.get_map():
            for key,_ in multiplexer.select():
                line=key.fileobj.readline();arm=key.data
                if not line:
                    multiplexer.unregister(key.fileobj);key.fileobj.close();continue
                logs[arm].write(line);logs[arm].flush()
                print(arm+': '+line,end='',flush=True)
        codes={a:c.wait() for a,c in children.items()}
        for log in logs.values():log.close()
        state['exit_codes']=codes;save();assert all(c==0 for c in codes.values()),codes
        for arm in plans:
            report=json.loads((args.out/(arm+'_study')/'owner_state.json').read_text())
            assert report['status']=='BROADER_COMPACT_SHARED8000_AND_ALL_FIVE_CAUSAL_CONTROLS_VALIDATED'
        state.update(status='MATCHED_PHYSICAL_FACTORIZATION_TRAINING_AND_ALL_CONTROLS_COMPLETE');save()
    except Exception as error:
        state.update(status='PHYSICAL_FACTORIZATION_PAIR_FAILED',error=repr(error));save();raise


if __name__=='__main__':main()
