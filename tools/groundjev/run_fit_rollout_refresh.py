"""One blocking GPU4 handoff, then a bounded current-actor native fit rollout."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import select
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    proof = json.loads((ROOT/plan['CPU_validation']).read_text())
    assert proof['status']=='ACTUAL48_NATIVE_REPLAY_AND_BOUNDED_REFRESH_CONTRACT_VALIDATED'
    assert plan['combined_registered_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    args.out.mkdir(parents=True,exist_ok=False)
    state = {'status':'WAITING_ORIGINAL_GPU4_FULL1203_OWNER_EXIT','owner_pid':os.getpid(),
        'physical_gpu':plan['physical_gpu'],'upstream':plan['upstream_owner'],
        'owner_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'new_training_steps':0,'new_DLM_requests':False,'full_goal_complete':False}

    def save():
        temp = args.out/'owner_state.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.out/'owner_state.json')
        print(json.dumps({'status':state['status'],'physical_gpu':plan['physical_gpu']}),flush=True)

    save()
    try:
        pid,start = plan['upstream_owner']['identity']
        proc = Path('/proc')/str(pid)
        try:
            fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
            if fields[19]==start and fields[0]!='Z':
                assert (proc/'cwd').resolve()==ROOT
                assert 'tools/groundjev/run_unified600_controls.py' in (proc/'cmdline').read_bytes().decode().split('\0')
                descriptor = os.pidfd_open(pid)
                try:
                    waiting = select.poll();waiting.register(descriptor,select.POLLIN);waiting.poll()
                finally:
                    os.close(descriptor)
        except (FileNotFoundError,ProcessLookupError):
            pass
        upstream = json.loads((ROOT/plan['upstream_owner']['directory']/'owner_state.json').read_text())
        assert upstream['status']=='SAME_SEMANTIC_POLICY_THREE_ARM_CONTROLS_COMPLETE'
        assert len(upstream['completed'])==3 and not upstream['failed']
        state['upstream_all_three_successfully_completed'] = True
        output = args.out/'rollout'
        env = os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        state['child_launch_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        command = ['python3','tools/groundjev/run_with_resources.py','--gpus',str(plan['physical_gpu']),
            '--',str(ROOT/'.venv-onejev-py310/bin/python'),'tools/groundjev/prepare_student_rollout.py',
            '--out',str(output),'--checkpoint',str(ROOT/plan['checkpoint']),
            '--candidate-source','native_ground','--native-candidate-replay',str(ROOT/plan['native_candidates']),
            '--output-byte-budget',str(plan['output_byte_budget'])]
        state['status']='CURRENT_NATIVE600_ACTOR_REAL48_FIT_ROLLOUT';save()
        with (args.out/'rollout.log').open('w') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            state['child_pid']=child.pid;save()
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code = child.wait()
        state['rollout_exit_code']=code
        if code:raise RuntimeError(f'Current-actor rollout exited {code}; preserve valid causal prefix and log')
        summary = json.loads((output/'summary.json').read_text())
        assert Path(summary['checkpoint']).resolve()==(ROOT/plan['checkpoint']).resolve()
        assert summary['counts']['replayed_native_frames']==plan['registered_frames']
        assert summary['counts']['actual_native_candidates']==plan['registered_candidates']
        from tools.groundjev.prepare_student_rollout import registered_native_candidates
        original = registered_native_candidates(ROOT/plan['native_candidates'])
        actions,seen = Counter(),set()
        for line in (output/'native_rollout.jsonl').open():
            row = json.loads(line);key=row['sequence'],row['frame']
            assert key not in seen;seen.add(key)
            assert row['candidates']==original[key]['candidates'] and row['perception']==original[key]['perception']
            for event in row['state']['events']:
                if 'candidate' in event:actions[event['action']]+=1
        assert seen==set(original)
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        assert size<=plan['output_byte_budget']
        state.update(status='CURRENT_NATIVE600_FIT48_REAL_ROLLOUT_COMPLETE',actual_action_counts=dict(actions),
            exported_bytes=size,summary=summary,next_step='Audit actual assigned-identity support; no new training or accuracy improvement claimed.')
        save()
    except Exception as error:
        state.update(status='CURRENT_ACTOR_FIT_ROLLOUT_REFRESH_FAILED',error=repr(error));save();raise


if __name__=='__main__':
    main()
