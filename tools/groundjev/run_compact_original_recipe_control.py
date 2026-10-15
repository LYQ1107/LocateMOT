"""One CPU owner for the missing unchanged-recipe continuation control."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT' and plan['device'] == 'cpu'
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    pair = json.loads((ROOT/plan['paired_history_plan']).read_text())
    source = ROOT/plan['pack']; pack = torch.load(source/'examples.pt', weights_only=True)
    groups = Counter(e['task_index'] for e in pack['examples'])
    withholding = sum(bool(e.get('derived_withholding')) for e in pack['examples'] if e['task_index'] == 0)
    assert groups == {0: 149, 1: 325, 2: 62} and withholding == 69
    teacher = Counter(str(e['task_index']) for e in pack['examples'] if e['native_teacher'])
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'REGISTERED_ORIGINAL_RECIPE_CPU_CONTINUATION_CONTROL',
             'owner_pid': os.getpid(), 'owner_starttick': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
             'owner_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'plan': str(args.plan), 'GPUs': [], 'source_groups': dict(groups),
             'source_withholding_groups': withholding, 'source_native_teacher_groups': dict(teacher),
             'additional_feature_cache_bytes': 0, 'completed': {}, 'full_goal_complete': False}
    del pack
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    python = str(ROOT/'.venv-onejev-py310/bin/python')

    def save():
        temp = out/'owner_state.tmp'; temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(out/'owner_state.json')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command):
        state.update(status=stage, command=command); save()
        with (out/(stage+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            state['child_pid'] = child.pid; save()
            for line in child.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            code = child.wait()
        state.pop('child_pid', None)
        assert code == 0, f'{stage} exited{code}; preserve first failure and valid checkpoint/prefix'

    save()
    try:
        training = out/'training'
        run('ORIGINAL_RECIPE_SHARED1200_TRAINING', [python, 'tools/groundjev/train_compact_joint.py',
            '--pack', str(source), '--out', str(training), '--checkpoint', str(ROOT/plan['checkpoint']),
            '--steps', '1200', '--device', 'cpu', '--per-task-sampling'])
        trained = json.loads((training/'owner_state.json').read_text())
        assert trained['steps'] == 1200 and trained['device'] == 'cpu'
        assert all(len(ids) == 400 for ids in trained['sampled_example_ids'].values())
        assert all(delta == 0 for delta in trained['three_task_checkpoint_reload_max_difference'].values())
        old = json.loads((ROOT/pair['output']/'old_training/owner_state.json').read_text())
        assert all(trained['sampled_example_ids'][k] == old['sampled_example_ids'][k] for k in ('1', '2'))
        checkpoint = torch.load(training/'checkpoint.pt', map_location='cpu', weights_only=True)
        latest = torch.load(training/'latest_training_state.pt', map_location='cpu', weights_only=False)
        assert latest['step'] == 1200 and checkpoint['state_dict'].keys() == latest['state_dict'].keys()
        assert all(torch.equal(t, latest['state_dict'][k]) and torch.isfinite(t).all()
                   for k, t in checkpoint['state_dict'].items())
        state['completed']['training'] = {'steps': 1200, 'sampled_groups_each_task': 400,
            'RMOT_OVMOT_sample_streams_exactly_equal_completed_pair': True,
            'three_task_reload_max_difference': 0, 'full_state_equals_checkpoint_and_finite': True,
            'checkpoint': trained['checkpoint'], 'parameters': trained['parameters'],
            'actual_loop_seconds': trained['training_log'][-1]['elapsed_seconds']}
        save()
        references = out/'reference_runs'; references.mkdir()
        for scope, path in pair['before_CPU_reference_runs'].items():
            (references/(scope+'_before')).symlink_to((ROOT/path).resolve(), target_is_directory=True)
        policies = {'association': 'learned', 'auxiliary': 'learned', 'memory_policy': 'gated',
                    'feature_view': 'all', 'lost_seconds': 3., 'shortlist': 32}
        controls = {'regression': pair['control_regression'], 'checkpoint': plan['checkpoint'],
                    'reference_root': str(references), 'scopes': pair['scopes'], 'cpu_blas_threads': 4,
                    'arms': {'before': dict(policies, reference_arm='before'),
                             'continuation': dict(policies, checkpoint=trained['checkpoint'])},
                    'comparison_limit': plan['comparison_limit'],
                    'memory_storage': 'Original binary final memory and all candidate/decision records; no new input feature pack.'}
        resolved = out/'resolved_controls_plan.json'; resolved.write_text(json.dumps(controls, indent=2)+'\n')
        run('FIVE_SCOPE_OFFICIAL_CONTROLS', [python, 'tools/groundjev/run_fixed_affinity_controls.py',
            '--plan', str(resolved), '--out', str(out/'controls')])
        result = json.loads((out/'controls/owner_state.json').read_text())
        assert result['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(result['completed']) == 10
        state['completed']['controls'] = result['completed']
        state['status'] = 'ORIGINAL_RECIPE_CONTINUATION1200_AND_FIVE_SCOPE_CONTROLS_VALIDATED'; save()
    except Exception as error:
        state.update(status='ORIGINAL_RECIPE_CONTINUATION_CONTROL_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
