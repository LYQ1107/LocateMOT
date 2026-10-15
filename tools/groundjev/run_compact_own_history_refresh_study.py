"""One CPU owner for matched history-pack preparation, training and five scopes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'REGISTERED_COMPACT_OWN_HISTORY_REFRESH_CPU_STUDY', 'owner_pid': os.getpid(),
        'owner_starttick': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'plan': str(args.plan), 'physical_GPUs': [], 'completed': {}, 'full_goal_complete': False}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    python = str(ROOT/'.venv-onejev-py310/bin/python')

    def save():
        (out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command):
        state['status'] = stage; save()
        with (out/(stage+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            state['current_child_pid'] = child.pid; save()
            for line in child.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            code = child.wait()
        state.pop('current_child_pid', None)
        assert code == 0, (stage, code)

    save()
    try:
        run('PREPARE_REAL_PAIRED_HISTORY_INPUTS', [python, 'tools/groundjev/prepare_compact_own_history_refresh.py', '--plan', str(args.plan)])
        packs = json.loads((ROOT/plan['pack_output']/'summary.json').read_text())
        assert packs['status'] == 'PAIRED_COMPACT_PHYSICAL_HISTORY_PACKS_AND_REAL_INPUT_CAPTURE_VALIDATED'
        state['completed']['packs'] = packs
        trained = {}
        for arm in ('old', 'refreshed'):
            training = out/(arm+'_training')
            run(arm+'_CPU1200_TRAINING', [python, 'tools/groundjev/train_compact_joint.py',
                '--pack', packs['packs'][arm], '--out', str(training), '--checkpoint', str(ROOT/plan['checkpoint']),
                '--steps', str(plan['steps_each']), '--device', 'cpu', '--per-task-sampling'])
            trained[arm] = json.loads((training/'owner_state.json').read_text())
            assert trained[arm]['steps'] == 1200 and trained[arm]['device'] == 'cpu'
            assert all(delta == 0 for delta in trained[arm]['three_task_checkpoint_reload_max_difference'].values())
            assert all(len(ids) == 400 for ids in trained[arm]['sampled_example_ids'].values())
            state['completed'][arm+'_training'] = {'checkpoint': trained[arm]['checkpoint'],
                'steps': 1200, 'sampled_groups_each_task': 400, 'three_task_reload_max_difference': 0,
                'parameters': trained[arm]['parameters']}; save()
        assert trained['old']['sampled_example_ids'] == trained['refreshed']['sampled_example_ids']
        state['exact_per_step_group_ID_sampling_matched'] = True
        references = out/'reference_runs'; references.mkdir()
        for scope, source in plan['before_CPU_reference_runs'].items():
            runtime = json.loads((ROOT/source/'result.json').read_text())
            assert runtime['decision_device'] == 'cpu' and runtime['frames'] == plan['scopes'][scope]['frames']
            assert Path(runtime['checkpoint']).resolve() == (ROOT/plan['checkpoint']).resolve()
            (references/(scope+'_before')).symlink_to((ROOT/source).resolve(), target_is_directory=True)
        policies = {'association': 'learned', 'auxiliary': 'learned', 'memory_policy': 'gated',
            'feature_view': 'all', 'lost_seconds': 3., 'shortlist': 32}
        controls = {'regression': plan['control_regression'], 'checkpoint': plan['checkpoint'],
            'reference_root': str(references), 'scopes': plan['scopes'],
            'arms': {'before': dict(policies, reference_arm='before'),
                     **{arm: dict(policies, checkpoint=trained[arm]['checkpoint']) for arm in trained}},
            'comparison_limit': 'Paired1200-step history-refresh continuation, identical sampled group IDs, unchanged semantic examples; not isolated age/confidence or token/FLOP matched. Same final checkpoint across five scopes. Existing physical fit coverage48frames; no full-dataset/ordinary-model success assumed.',
            'memory_storage': 'Existing lossless binary final memory and all decisions; no new input feature archives.',
            'cpu_blas_threads': 4}
        path = out/'resolved_controls_plan.json'; path.write_text(json.dumps(controls, indent=2)+'\n')
        run('FIVE_SCOPE_OFFICIAL_CONTROLS', [python, 'tools/groundjev/run_fixed_affinity_controls.py',
            '--plan', str(path), '--out', str(ROOT/plan['controls_output'])])
        controls_state = json.loads((ROOT/plan['controls_output']/'owner_state.json').read_text())
        assert controls_state['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(controls_state['completed']) == 15
        state['completed']['controls'] = controls_state['completed']
        state['status'] = 'PAIRED_COMPACT_HISTORY1200_AND_ALL_FIVE_SCOPE_OFFICIAL_CONTROLS_VALIDATED'; save()
    except Exception as error:
        state.update(status='COMPACT_OWN_HISTORY_REFRESH_STUDY_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
