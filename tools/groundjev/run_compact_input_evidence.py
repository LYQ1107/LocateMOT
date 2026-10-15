"""One serial CPU owner: matched evidence retraining, then full-video controls."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    regression = json.loads((ROOT/plan['regression']).read_text())
    assert regression['status'] == 'REAL_THREE_TASK_INPUT_EVIDENCE_ISOLATION_AND_DEFAULT_EXACT'
    scheduler = json.loads((ROOT/plan['scheduler_regression']).read_text())
    assert scheduler['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE'
    assert len(scheduler['completed']) == 2
    assert all(row['decoded_frames'] == 21 for row in scheduler['completed'].values())
    previous = json.loads((ROOT/plan['upstream_owner']/'owner_state.json').read_text())
    proc = Path('/proc')/str(previous['owner_pid'])
    registered = None
    if proc.exists():
        assert (proc/'cwd').resolve() == ROOT
        assert 'tools/groundjev/run_fixed_affinity_controls.py' in (proc/'cmdline').read_bytes().decode().split('\0')
        registered = [previous['owner_pid'], (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]]
    state = {'status': 'WAITING_REGISTERED_HYBRID_CPU_OWNER_EXIT', 'owner_pid': os.getpid(),
             'upstream_registered': registered, 'device': 'cpu', 'additional_GPUs': 0,
             'completed_training': {}, 'new_feature_cache_files': False, 'full_goal_complete': False}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '4'
    python = str(ROOT/'.venv-onejev-py310/bin/python')

    def save():
        temporary = args.out/'owner_state.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.out/'owner_state.json')
        print(state['status'], flush=True)

    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, check=True)

    save()
    try:
        while alive(registered):
            time.sleep(20)
        previous = json.loads((ROOT/plan['upstream_owner']/'owner_state.json').read_text())
        assert previous['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE', previous
        import torch
        torch.set_num_threads(4)
        baseline = json.loads((ROOT/plan['baseline_training']/'owner_state.json').read_text())
        initial = ROOT/plan['initialization']
        assert baseline['steps'] == plan['steps'] == 1200
        assert baseline['selected_tasks'] == ['mot', 'rmot', 'ovmot']
        baseline_initial = ROOT/baseline['initialization']
        left = torch.load(initial, map_location='cpu', weights_only=True)['state_dict']
        right = torch.load(baseline_initial, map_location='cpu', weights_only=True)['state_dict']
        assert left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)
        assert Path(baseline['source_pack']).resolve() == (ROOT/plan['source_pack']).resolve()
        for mode in ('geometry_motion', 'geometry_appearance'):
            training = args.out/(mode+'_training')
            run(mode+'_TRAINING', [python, 'tools/groundjev/train_compact_joint.py',
                '--pack', str(ROOT/plan['source_pack']), '--out', str(training), '--checkpoint', str(initial),
                '--steps', str(plan['steps']), '--device', 'cpu', '--per-task-sampling', '--input-evidence', mode])
            trained = json.loads((training/'owner_state.json').read_text())
            assert trained['status'] == 'SHARED_COMPACT_JOINT1200_TRAINING_COMPLETE'
            assert trained['parameters'] == baseline['parameters'] == 1146511
            assert trained['sampled_example_ids'] == baseline['sampled_example_ids']
            assert all(v == 0 for v in trained['three_task_checkpoint_reload_max_difference'].values())
            state['completed_training'][mode] = {k:v for k,v in trained.items() if k != 'sampled_example_ids'}
            state['actual_all1200_sample_IDs_and_initial_tensors_exact'] = True
            save()
        run('ALL_THREE_INPUT_EVIDENCE_FULL_VIDEO_CONTROLS',
            ['python3', 'tools/groundjev/run_fixed_affinity_controls.py', '--plan', plan['controls'],
             '--out', str(args.out/'controls')])
        controls = json.loads((args.out/'controls/owner_state.json').read_text())
        assert controls['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(controls['completed']) == 12
        state.update(status='MATCHED_THREE_INPUT_EVIDENCE_TRAINING_AND12_CONTROLS_COMPLETE',
                     completed_controls=controls['completed'])
        save()
    except Exception as error:
        state.update(status='INPUT_EVIDENCE_STUDY_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
