"""Owned GPU handoff, native localization200, and controlled visual/JEV composition."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tools.groundjev.run_unified600_controls import alive
    from tools.groundjev.reattach_native_full9 import alive as client_alive
    from tools.groundjev.service_process_group import members
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validated = json.loads((ROOT/plan['contract_validation']).read_text())
    assert validated['status'] == 'REAL_LOCALIZATION200_ROWS_AND_ACTUAL_TRACKING200_SCHEDULE_MATCH_VALIDATED'
    args.out.mkdir(parents=True, exist_ok=False)
    identity = None
    if not plan.get('gpu_handoff_receipt'):
        retention = ROOT/plan['retention_owner']
        prior = json.loads((retention/'owner_state.json').read_text())
        proc = Path('/proc')/str(prior['owner_pid'])
        if proc.exists():
            assert (proc/'cwd').resolve() == ROOT
            assert 'tools/groundjev/reattach_native_full9.py' in (proc/'cmdline').read_bytes().decode().split('\0')
            identity = [prior['owner_pid'], (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]]
        clients = json.loads((ROOT/plan['existing_client_contract']).read_text())
        clients = [c for c in clients if ':8102' in c['port_or_url']]
        assert clients
    state = {'status': 'VERIFYING_COMPLETED_GPU_HANDOFF' if plan.get('gpu_handoff_receipt') else 'WAITING_EXISTING_ADAPTED200_GPU6_RELEASE', 'owner_pid': os.getpid(),
        'physical_gpu': plan['physical_gpu'], 'registered_retention_owner': identity, 'completed': {},
        'new_feature_cache_files': False, 'foreign_or_current_retention_jobs_stopped': False,
        'composed_adapted_weights_not_joint_gradient_training': True, 'full_goal_complete': False}
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)

    def save():
        temporary = args.out/'owner_state.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.out/'owner_state.json')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command):
        state['status'] = stage
        with (args.out/(stage+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state['current_child'] = [child.pid, (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]]
            save()
            code = child.wait()
            if code:
                raise RuntimeError(f'{stage} exited {code}; inspect its retained log and unfinished unit')
        state.pop('current_child', None)

    save()
    try:
        if plan.get('gpu_handoff_receipt'):
            from tools.groundjev.completed_gpu_handoff import validate_release
            state['upstream_GPU_release'] = validate_release(ROOT/plan['gpu_handoff_receipt'], plan['physical_gpu'])
        while not plan.get('gpu_handoff_receipt'):
            current = json.loads((retention/'owner_state.json').read_text())
            release = current['completed'].get('adapted200')
            if release and not any(client_alive(c) for c in clients):
                assert release['all_group_members_gone']
                assert not members(release['owned_service_group_stopped'])
                state['upstream_GPU6_retention_release'] = release
                # Missing native retention tasks remain failures of that study;
                # a released GPU can still run this independent training control.
                break
            if not alive(identity):
                raise RuntimeError('Retention owner ended without verified GPU6 client/service release; do not duplicate or stop its jobs')
            time.sleep(20)
        train = ROOT/plan['training_output']
        python = str(ROOT/'.venv-ground-train-py312/bin/python')
        run('LOCALIZATION200_TRAINING', ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(plan['physical_gpu']), '--',
            python, 'tools/groundjev/train_native_ground_tracking200.py', '--plan', str(ROOT/plan['training_plan']),
            '--out', str(train), '--static-flex-shapes', '--debug-kernel-sync', '--checkpoint-every', '25'])
        trained = json.loads((train/'owner_state.json').read_text())
        assert trained['status'] == 'NATIVE_GROUND_LOCALIZATION200_TRAINING_COMPLETE' and trained['steps'] == 200
        assert trained['trainable_parameters'] == 27267584 and trained['checkpoint_reload_max_difference'] == 0
        assert trained['changed_BF16_elements'] > 0 and trained['all_registered_pairs_visited']
        assert trained['identity_loss_coefficients'] == {'teacher': 0., 'native_proposals': 0.}
        logs = [json.loads(line) for line in (train/'log.jsonl').open()]
        assert len(logs) == 200 and all(r['total_loss'] == r['native_loss'] for r in logs)
        state['completed']['training'] = {'run': str(train), 'steps': 200, 'parameters': trained['trainable_parameters'],
            'same_registered112_rows_200_pair_draws': True, 'identity_loss_coefficients_zero': True,
            'reload_max_difference': 0, 'actual_native_loss_equals_total_loss_every_step': True}
        save()
        receipt = args.out/'bundle_export.json'
        run('LOCALIZATION200_EXPORT', [python, 'tools/groundjev/export_native_projector_delta.py',
            '--delta', str(train/'projector_delta.pt'), '--out', str(ROOT/plan['bundle']), '--receipt', str(receipt)])
        exported = json.loads(receipt.read_text())
        assert exported['native_tensor_count'] == 569 and exported['all_other_native_tensors_unmodified']
        state['completed']['export'] = exported
        save()
        run('ORDERED_NATIVE_FRONTEND_CONTROLS', ['python3', 'tools/groundjev/compare_ground_checkpoints_isolated.py',
            '--plan', str(ROOT/plan['frontend_plan']), '--out', str(ROOT/plan['frontend_output'])])
        front = json.loads((ROOT/plan['frontend_output']/'owner_state.json').read_text())
        assert front['status'] == 'ISOLATED_ORDERED_GROUND_CONTROLS_COMPLETE' and len(front['completed']) == 12
        state['completed']['native_frontends'] = {'run': plan['frontend_output'], 'native_frames': 96,
            'scope_metrics': {k: r['metrics'] for k, r in front['completed'].items()},
            'actual_localization_repeat': front['repeat_checks']}
        save()
        run('GROUND_ONE_FACTORIAL', ['python3', 'tools/groundjev/run_unified600_controls.py',
            '--plan', str(ROOT/plan['factorial_plan']), '--out', str(ROOT/plan['factorial_output'])])
        full = json.loads((ROOT/plan['factorial_output']/'owner_state.json').read_text())
        assert full['status'] == 'SAME_POLICY_GROUND_ONE_FACTORIAL_CONTROLS_COMPLETE' and len(full['completed']) == 18
        assert not full['failed']
        state['completed']['factorial'] = {'run': plan['factorial_output'], 'controls': 18,
            'scope_metrics': {k: r['official']['metrics'] for k, r in full['completed'].items()},
            'scope': 'shortMOT4/RMO8/RMO12, threeGroundweights x originalOne/native600; not a full-dataset result'}
        state['status'] = 'LOCALIZATION200_AND_CONTROLLED_VISUAL_ONE_COMPOSITION_COMPLETE'
        save()
    except Exception as error:
        state.update(status='LOCALIZATION_STUDY_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
