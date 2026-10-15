"""After paired E2E releases GPU5, run the existing two independent studies serially."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.groundjev.run_unified600_controls import alive
from tools.groundjev.service_process_group import members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    proof = json.loads((ROOT/plan['registration_validation']).read_text())
    assert proof['status'] == 'GPU5_FOLLOWUPS_SCIENTIFIC_PLANS_AND_COMPLETED_HANDOFF_VALIDATED'
    for relative in plan['superseded_waiter_receipts']:
        handoff = json.loads((ROOT/relative).read_text())
        assert handoff['status'] == 'UNSTARTED_CPU_WAITER_SUPERSEDED_WITHOUT_GPU_JOB'
        assert not alive(handoff['old_owner_identity'])
    upstream = ROOT/plan['upstream_e2e']
    prior = json.loads((upstream/'owner_state.json').read_text())
    proc = Path('/proc')/str(prior['owner_pid'])
    identity = None
    if proc.exists():
        assert (proc/'cwd').resolve() == ROOT
        assert 'tools/groundjev/run_paired_tracking_e2e.py' in (proc/'cmdline').read_bytes().decode().split('\0')
        identity = [prior['owner_pid'],(proc/'stat').read_text().rsplit(')',1)[1].split()[19]]
    state = {'status':'WAITING_ORIGINAL_PAIRED_E2E_GPU5_RELEASE','owner_pid':os.getpid(),
             'physical_gpu':5,'upstream_owner_identity':identity,'completed':{},
             'native_retention_jobs_unchanged':True,'full_goal_complete':False}
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''

    def save():
        temporary = args.out/'owner_state.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(args.out/'owner_state.json')
        print(state['status'],flush=True)

    save()
    try:
        while alive(identity):
            time.sleep(20)
        prior = json.loads((upstream/'owner_state.json').read_text())
        assert prior['status'] == 'SAME800_SAME_GPU_FRESH_FULL_COMPACT_E2E_AND_DETECTOR_PAIRING_VALIDATED', prior
        assert prior['physical_gpu'] == 5 and len(prior['completed']) == 2
        groups = set(prior['current_owned_service_groups'])
        for record in prior['completed'].values():
            groups.update(int(g) for g in record['sampled_resources']['sampled_group_peaks_MiB'])
        assert all(not members(group) for group in groups)
        release = {'status':'VERIFIED_COMPLETED_GPU_OWNER_RELEASE','physical_gpu':5,
            'upstream_owner_identity':identity,'upstream_owner_output':str(upstream),
            'upstream_status':prior['status'],'owned_process_groups':sorted(groups),
            'all_owned_groups_gone':True,
            'hardware_at_release':subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=uuid,name,memory.total,memory.free',
                '--format=csv,noheader,nounits'],text=True).strip()}
        (ROOT/plan['gpu_handoff_receipt']).write_text(json.dumps(release,indent=2)+'\n')
        state['actual_GPU5_release'] = release
        for name, script, configuration, output in [
            ('LOCALIZATION200','run_ground_localization_study.py',plan['localization_plan'],plan['localization_output']),
            ('NATURAL_MOT20','run_natural_memory_study.py',plan['natural_plan'],plan['natural_output'])]:
            state['status'] = name
            with (args.out/(name+'.log')).open('w') as log:
                child = subprocess.Popen(['python3','tools/groundjev/'+script,'--plan',configuration,'--out',output],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state['current_child'] = [child.pid,(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]]
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f'{name} exited {code}; preserve its latest checkpoint and unfinished unit')
            completed = json.loads((ROOT/output/'owner_state.json').read_text())
            expected = 'LOCALIZATION200_AND_CONTROLLED_VISUAL_ONE_COMPOSITION_COMPLETE' if name == 'LOCALIZATION200' else 'NATURAL_MOT20_SCOPE_AND_FROZEN_MEMORY_CONTROLS_COMPLETE'
            assert completed['status'] == expected
            state['completed'][name] = {'run':output,'status':completed['status']}
            state.pop('current_child',None)
            save()
        state['status'] = 'GPU5_LOCALIZATION_AND_NATURAL_MEMORY_STUDIES_COMPLETE'
        save()
    except Exception as error:
        state.update(status='GPU5_FOLLOWUP_QUEUE_FAILED',error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
