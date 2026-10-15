"""One blocking owner: source handoff, real CPU plan, shared SFT, fixed controls."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def cpu_plan(config_path, report_path):
    import torch
    from tools.groundjev.paired_sft_sampler import index_shuffle_plan
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    from train import sft
    torch.set_num_threads(4)
    stateful = configure_group_reference_sft()
    sft.plan_epoch = index_shuffle_plan(sft.plan_epoch)
    original, epoch = sft.plan_only, sft.plan_epoch

    def capture(bundle, rows, args, world):
        batches, dropped, lengths = epoch(bundle, rows, args, 0, 0, world)
        assert world == 1 and len(rows) == 24000 and dropped == 0
        assert sorted(i for batch in batches for i in batch) == list(range(24000))
        steps = sft.plan_steps(batches, lengths, world, args.accum_rows)
        assert len(steps) == 6000
        os.environ['PLAN_WORLD'] = str(world)
        sft.plan_epoch = lambda *unused, **kwargs: (batches, dropped, lengths)
        try:
            code = original(bundle, rows, args, world)
        finally:
            sft.plan_epoch = epoch
        assert code == 0 and not torch.cuda.is_initialized()
        report = {'status': 'ALL24000_NATIVE_MM_CPU_ROWS_AND6000_PAIRED_SHUFFLE_UPDATES_VALIDATED',
                  'rows': len(rows), 'dropped_rows': dropped, 'maximum_tokens': max(lengths),
                  'planned_tokens': sum(lengths), 'planned_optimizer_steps': len(steps),
                  'actual_optimizer_steps': 0, 'world': world,
                  'input_io_protocol': stateful.protocol(args)['input_io_protocol'], 'full_goal_complete': False}
        report_path.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        return code

    sft.plan_only = capture
    return sft.main(['--config', str(config_path), '--plan-only'])


def main():
    from tools.groundjev.run_history_refresh_controls import wait_dependency, validate_run
    from tools.groundjev.service_process_group import members
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path)
    parser.add_argument('--out', type=Path); parser.add_argument('--cpu-plan', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.cpu_plan:
        return cpu_plan(args.cpu_plan, args.report)
    plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['training_backend'] == 'reference' and plan['physical_gpu'] == 7
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'REGISTERED_WAITING_ORIGINAL_NATIVE_SOURCE_AND_GROUP_EXPORT',
             'owner_pid': os.getpid(), 'owner_starttick': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
             'owner_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'physical_gpu_only_after_dependencies': 7, 'new_optimizer_steps': 0,
             'plan': str(args.plan), 'completed': {}, 'full_goal_complete': False}
    python = str(ROOT/'.venv-onejev-py310/bin/python')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    gpu = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '7', '--', python]

    def save():
        temp = args.out/'owner_state.tmp'; temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(args.out/'owner_state.json')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command):
        state.update(status=stage, command=command); save()
        with (args.out/(stage+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            state['child_pid'] = child.pid; save()
            for line in child.stdout:
                log.write(line); log.flush(); print(line, end='', flush=True)
            code = child.wait()
        state.pop('child_pid', None); state['last_exit_code'] = code
        assert code == 0, f'{stage} exited{code}; retain complete units and latest full state/prefix'

    save()
    try:
        dependencies = []
        for key, script, terminal in (
                ('native_groups', 'tools/groundjev/export_broader_native_full_groups.py',
                 'ALL43_NATIVE_TRAIN_CLIPS_FULL_V3_GROUP_REFS_AND_CPU_PLANS_VALIDATED'),
                ('native_source', 'tools/groundjev/collect_broader_native_MOT.py',
                 'BROADER_NATIVE_TRAIN43_VIDEO1056_FRAME_COLLECTION_COMPLETE')):
            registered = json.loads((ROOT/plan[key]/'owner_state.json').read_text())
            dependencies.append({'directory': plan[key], 'script': script,
                                 'identity': [registered['owner_pid'], registered['owner_starttick']],
                                 'required_terminal_status': terminal})
        state['dependencies'] = dependencies; save()
        for dependency in dependencies:
            result = wait_dependency(dependency)
            state['completed'][dependency['directory'].rsplit('/', 1)[1]] = {
                'terminal_status': result['status'], 'owner_exited': True}
            save()
        for group in result['owned_service_groups']:
            assert not members(group), f'Original collector service group remains: {group}'
        state['owned_source_service_groups_gone'] = result['owned_service_groups']; save()
        run('PREPARE_BALANCED_THREE_TASK_CURRICULUM',
            [python, 'tools/groundjev/prepare_broader_shared_curriculum.py', '--plan', str(args.plan)])
        curriculum = ROOT/plan['output']
        summary = json.loads((curriculum/'summary.json').read_text())
        assert summary['MOT_physical_frames'] == 1056 and summary['RMOT_expressions'] == 4350
        config = json.loads((curriculum/'training_config.json').read_text())
        cpu = dict(config, name='broader6000_native_CPU_plan', runs_dir=str(args.out/'CPU_plan'), device='cpu')
        cpu_config = args.out/'CPU_plan_config.json'; cpu_config.write_text(json.dumps(cpu, indent=2)+'\n')
        run('ALL_NATIVE_CPU_ROWS_AND_SAMPLING_PLAN',
            [python, str(Path(__file__).resolve()), '--cpu-plan', str(cpu_config), '--report', str(args.out/'CPU_plan_validation.json')])
        state['completed']['curriculum_and_CPU_plan'] = summary; save()
        assert shutil.disk_usage(ROOT).free >= 40*2**30, 'Need40GiB for new checkpoint series; do not delete prior results'
        training = Path(config['runs_dir'])/config['name']
        assert not training.exists(), 'Resume latest valid full state explicitly; do not overwrite'
        run('SHARED_ONEJEV6000_TRAINING', gpu+['tools/groundjev/run_group_reference_sft.py',
            '--config', str(curriculum/'training_config.json'), '--paired-index-shuffle',
            '--initial-training-state', str(ROOT/plan['initial_training_state']), '--expected-initial-step', '600'])
        records = [json.loads(line) for line in (training/'log.jsonl').open()]
        done = next(r for r in records if r.get('event') == 'done')
        assert done['steps'] == 6000 and done['examples'] == 24000
        assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in records if 'loss' in r)
        latest = json.loads((training/'latest_training_state.json').read_text())
        assert latest['step'] == 6000 and latest['seen'] == 24000
        state['new_optimizer_steps'] = 6000
        state['completed']['training'] = {'done': done, 'latest_full_state': latest}; save()
        checkpoint = training/'ckpt-step6000'
        run('NATIVE6000_MM_RELOAD', gpu+['tools/groundjev/check_onejev125_checkpoint.py', '--step', '6000',
            '--checkpoint', str(checkpoint), '--training-root', str(training), '--reference-checkpoint', config['model']])
        state['completed']['native_MM_reload'] = json.loads((training/'checkpoint6000_validation.json').read_text()); save()
        controls = json.loads((ROOT/'configs/groundjev/full_history_refresh600_controls_v1.json').read_text())
        results = args.out/'controls'; results.mkdir()
        for scope, job in controls['scopes'].items():
            before = ROOT/job['baseline_run'] if job['baseline_run'] else ROOT/controls['output_root']/('before_'+scope)
            for arm in ('before', 'default', 'assigned_identity'):
                ckpt = Path(config['model']) if arm == 'before' else checkpoint
                output = before if arm == 'before' and (before/'official_metrics.json').exists() else results/(arm+'_'+scope)
                if output.is_relative_to(results):
                    manifest = json.loads((ROOT/job['manifest']).read_text())
                    command = gpu+['tools/groundjev/run_video.py', '--task', manifest['task'],
                        '--video', str(ROOT/job['manifest']), '--backend', 'full', '--checkpoint', str(ckpt),
                        '--evidence-format', 'v3_crop_motion', '--adaptive-context-groups',
                        '--candidate-replay', str(ROOT/job['candidates']), '--omit-replay-feature-vectors',
                        '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0', '--out', str(output)]
                    if manifest['task'] == 'rmot': command += ['--native-rmot-membership']
                    if manifest['task'] == 'ovmot': command += ['--category-policy', 'native_hierarchical']
                    if arm == 'assigned_identity': command += ['--assigned-identity-updates']
                    name = arm+'_'+scope
                    run(name+'_INFERENCE', command)
                    scorer = 'evaluate_video.py' if manifest['task'] == 'mot' else 'evaluate_multitask.py'
                    run(name+'_OFFICIAL', ['python3', 'tools/groundjev/'+scorer, '--video', str(ROOT/job['manifest']), '--run', str(output)])
                    run(name+'_EVENTS', ['python3', 'tools/groundjev/analyze_tracking_events.py', '--video', str(ROOT/job['manifest']), '--run', str(output)])
                runtime = json.loads((output/'result.json').read_text())
                assert bool(runtime.get('assigned_identity_updates', False)) == (arm == 'assigned_identity')
                state['completed'][arm+'_'+scope] = validate_run(output, job, ckpt); save()
        state['status'] = 'SHARED6000_AND_ALL_FIXED_BEFORE_DEFAULT_FINAL_ID_CONTROLS_VALIDATED'; save()
    except Exception as error:
        state.update(status='BROADER_SHARED_ONEJEV_OWNER_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
