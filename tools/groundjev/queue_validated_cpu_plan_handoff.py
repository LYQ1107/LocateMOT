"""Wait for a complete equivalent plan; replace only the unfinished CPU stage."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def process(pid, tick=None):
    proc=Path('/proc')/str(pid)
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    assert fields[0]!='Z' and (tick is None or fields[19]==str(tick))
    assert (proc/'cwd').resolve()==ROOT
    return dict(pid=pid,starttick=fields[19],state=fields[0],
        command=(proc/'cmdline').read_bytes().decode().split('\0'))


def wait_exit(pid):
    try:descriptor=os.pidfd_open(pid)
    except ProcessLookupError:return
    try:
        waiter=select.poll();waiter.register(descriptor,select.POLLIN)
        while not waiter.poll(20000):pass
    finally:os.close(descriptor)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    out=ROOT/plan['output'];out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(plan,indent=2)+'\n')
    state=dict(status='WAITING_COMPLETED_FULL_PARALLEL_NATIVE_CPU_PLAN',
        owner_pid=os.getpid(),owner_starttick=process(os.getpid())['starttick'],
        existing_GPU_training_or_evaluators_stopped=False,actual_optimizer_updates=0,full_goal_complete=False)
    def save(status,**values):
        state.update(status=status,**values)
        tmp=out/'owner_state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'owner_state.json')
        print(json.dumps(state),flush=True)
    parallel=ROOT/plan['parallel_plan_output']
    old=ROOT/plan['original_owner_output']
    new=ROOT/plan['replacement_owner_output']
    training=Path(plan['training_output'])
    python=str(ROOT/'.venv-onejev-py310/bin/python')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    save(state['status'])
    try:
        parallel_state=json.loads(parallel.with_name(parallel.name+'_owner.json').read_text())
        assert [parallel_state['owner_pid'],parallel_state['owner_starttick']]==plan['parallel_owner_identity']
        if parallel_state['status']!='FULL_NATIVE_PARALLEL_CPU_PLAN_COMPLETE':
            actual=process(*plan['parallel_owner_identity'])
            assert 'python3' in actual['command'][0] and '-u' in actual['command']
            wait_exit(actual['pid'])
        parallel_state=json.loads(parallel.with_name(parallel.name+'_owner.json').read_text())
        assert parallel_state['status']=='FULL_NATIVE_PARALLEL_CPU_PLAN_COMPLETE' and parallel_state['exit_code']==0
        report=parallel/'CPU_plan_validation.json'
        measured=json.loads(report.read_text())
        assert measured['status']=='FULL_SOURCE_ALL_NATIVE_MM_ROWS_AND_FOUR_RANK_GLOBAL16_PLAN_VALIDATED'
        assert measured['rows']==459027 and measured['dropped_rows']==0 and measured['planned_world_size']==4
        assert measured['CPU_workers']==4 and measured['actual_optimizer_updates']==0
        assert measured['exact_serial_parallel_probe_rows']==192 and not measured['cuda_initialized'] and not measured['model_loaded']
        old_state=json.loads((old/'owner_state.json').read_text())
        assert old_state['status']!='FULL_SOURCE_JOINT_OWNER_FAILED',old_state.get('error')
        if old_state['status']!='FULL_NATIVE_CPU_PLAN' or training.exists():
            save('ORIGINAL_ALREADY_ADVANCED_NO_HANDOFF',original_status=old_state['status']);return 0
        assert [old_state['owner_pid'],old_state['owner_starttick']]==plan['original_owner_identity']
        save('REVALIDATING_COMPLETED_NATIVE_PLAN_BEFORE_CPU_HANDOFF')
        command=[python,str(ROOT/'tools/groundjev/run_full_joint_source_training.py'),'--cpu-plan',
            str(old/'CPU_plan_config.json'),'--report',str(out/'CPU_reuse_preflight.json'),
            '--validated-native-lengths',str(report)]
        with (out/'CPU_reuse_preflight.log').open('w') as log:
            code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert code==0, 'Completed plan reuse validation failed; original owner retained'
        reused=json.loads((out/'CPU_reuse_preflight.json').read_text())
        for key in ('rows','dropped_rows','maximum_tokens','planned_tokens','planned_updates',
                    'question_families','planned_unique_first_media_source_paths'):
            assert reused[key]==measured[key],key
        pid,tick=plan['original_owner_identity'];parent=process(pid,tick)
        assert 'tools/groundjev/run_full_joint_source_training.py' in parent['command']
        # Freeze the parent first so it cannot cross into GPU allocation between checks.
        os.kill(pid,signal.SIGSTOP)
        stopped_child=None
        try:
            while process(pid,tick)['state']!='T':time.sleep(.01)
            old_state=json.loads((old/'owner_state.json').read_text())
            if old_state['status']!='FULL_NATIVE_CPU_PLAN' or training.exists():
                save('ORIGINAL_ALREADY_ADVANCED_NO_HANDOFF',original_status=old_state['status']);return 0
            child=process(old_state['child_pid'])
            expected=[python,str(ROOT/'tools/groundjev/run_full_joint_source_training.py'),'--cpu-plan',
                str(old/'CPU_plan_config.json'),'--report',str(old/'CPU_plan_validation.json')]
            assert child['command'][:-1]==expected
            environment=dict(p.split('=',1) for p in (Path('/proc')/str(child['pid'])/'environ').read_bytes().decode().split('\0') if '=' in p)
            assert environment['CUDA_VISIBLE_DEVICES']==''
            os.kill(child['pid'],signal.SIGSTOP);stopped_child=child['pid']
            process(child['pid'],child['starttick'])
            receipt=dict(status='EXPLICIT_COMPLETED_EQUIVALENT_CPU_PLAN_HANDOFF_INTENT',
                reason='All459027 native rows now fully measured by the same function; completed-input reuse preflight passed. Replace only the zero-update CPU unit.',
                original_parent=parent,original_CPU_child=child,original_owner_state=old_state,
                completed_parallel_report=str(report),CPU_reuse_preflight=str(out/'CPU_reuse_preflight.json'),
                replacement_owner=str(new),training_output_absent=True,actual_optimizer_updates_before_handoff=0,
                GPU_training_or_inference_processes_signaled=False,
                expected_original_terminal='CPU child SIGTERM (-15), original owner records assertion/exit1; retained as explicit superseded CPU stage')
            (out/'handoff_intent.json').write_text(json.dumps(receipt,indent=2)+'\n')
            (old/'completed_parallel_CPU_handoff.json').write_text(json.dumps(receipt,indent=2)+'\n')
            os.kill(child['pid'],signal.SIGTERM);os.kill(child['pid'],signal.SIGCONT);stopped_child=None
            wait_exit(child['pid'])
        finally:
            if stopped_child is not None:os.kill(stopped_child,signal.SIGCONT)
            os.kill(pid,signal.SIGCONT)
        wait_exit(pid)
        old_state=json.loads((old/'owner_state.json').read_text())
        assert old_state['status']=='FULL_SOURCE_JOINT_OWNER_FAILED' and old_state['last_exit_code']==-15
        assert not training.exists()
        save('SUPERSEDED_CPU_OWNER_TERMINAL_RETAINED_STARTING_SAME_REGISTERED_EPOCH',
            superseded_original_state=old_state,completed_native_report=str(report))
        command=[python,'-u','tools/groundjev/run_full_joint_source_training.py','--plan',
            str(ROOT/plan['training_plan']),'--out',str(new),'--validated-native-lengths',str(report)]
        child=subprocess.Popen(command,cwd=ROOT,env=env)
        save('REPLACEMENT_SAME_EPOCH_OWNER_STARTED',replacement_child_pid=child.pid,replacement_command=command)
        descriptor=os.pidfd_open(child.pid);waiter=select.poll();waiter.register(descriptor,select.POLLIN)
        try:
            while not (new/'owner_state.json').exists():
                assert not waiter.poll(1000), 'Replacement owner exited before registration'
        finally:os.close(descriptor)
        observer_out=ROOT/plan['first5_observer_output']
        observer=subprocess.Popen([python,'-u','tools/groundjev/freeze_full_joint_first_updates.py',
            '--owner',str(new),'--out',str(observer_out),
            '--resource-registration',str(ROOT/plan['resource_registration'])],cwd=ROOT,env=env)
        save('SAME_EPOCH_RUNNING_WITH_FIRST5_ACTUAL_UPDATE_OBSERVER',replacement_child_pid=child.pid,observer_pid=observer.pid)
        code=child.wait()
        observer_code=observer.wait()
        assert code==observer_code==0
        final=json.loads((new/'owner_state.json').read_text())
        assert final['status']=='FULL_SOURCE_JOINT_EPOCH_AND_ALL_FIXED_BEFORE_DEFAULT_FINAL_ID_CONTROLS_VALIDATED'
        save('SAME_REGISTERED_EPOCH_AND_CONTROLS_COMPLETED_AFTER_CPU_ONLY_HANDOFF',replacement_completion=final)
    except Exception as error:
        save('CPU_PLAN_HANDOFF_COORDINATOR_FAILED',error=repr(error));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
