"""Block on registered training children, then compare unchanged semantic policies."""
import argparse
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def identity(pid):
    proc = Path('/proc')/str(pid)
    try:
        argv = (proc/'cmdline').read_bytes().decode().split('\0')
        assert (proc/'cwd').resolve() == ROOT
        assert 'tools/groundjev/run_with_resources.py' in argv
        assert 'tools/groundjev/onejev_resumable_sft.py' in argv
        return [pid, (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]]
    except FileNotFoundError:
        return None


def alive(registered):
    if registered is None:
        return False
    try:
        fields = (Path('/proc')/str(registered[0])/'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == registered[1]
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    training_owner = json.loads((ROOT/plan['training_owner']/'owner_state.json').read_text())
    registered = {arm: identity(training_owner[arm+'_child_pid']) for arm in ('teacher', 'native')}
    state = {'status': 'WAITING_REGISTERED_TEACHER_GPU4_HANDOFF', 'owner_pid': os.getpid(),
             'registered_training_children': registered, 'physical_gpu': plan['physical_gpu'],
             'completed': {}, 'failed': {}, 'same_checkpoint_all_three_tasks_each_arm': True,
             'native_semantic_policy_identical_before_after': True, 'E2E_FPS_claim': False,
             'training_or_scorers_or_thresholds_changed': False}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed']),
                          'failed': list(state['failed'])}), flush=True)

    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def wait_training(arm):
        state['status'] = 'WAITING_REGISTERED_'+arm.upper()+'_600_CHILD_EXIT'
        save()
        while alive(registered[arm]):
            time.sleep(20)
        training = ROOT/plan['arms'][arm]['training']
        records = [json.loads(line) for line in (training/'log.jsonl').open()]
        done = next(row for row in records if row.get('event') == 'done')
        master = json.loads((training/'latest_training_state.json').read_text())
        assert done['steps'] == master['step'] == 600 and done['examples'] == 2400
        assert all(math.isfinite(row['loss']) for row in records if 'loss' in row)
        state[arm+'_terminal_training_evidence'] = {'done': done, 'master': master}
        save()

    prefix = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(plan['physical_gpu']),
              '--', str(ROOT/'.venv-onejev-py310/bin/python')]
    save()
    try:
        if plan.get('upstream_GPU_owner'):
            upstream = ROOT/plan['upstream_GPU_owner']
            previous = json.loads((upstream/'owner_state.json').read_text())
            proc = Path('/proc')/str(previous['owner_pid'])
            registered = None
            if proc.exists():
                assert (proc/'cwd').resolve() == ROOT
                assert 'tools/groundjev/run_unified600_controls.py' in (proc/'cmdline').read_bytes().decode().split('\0')
                registered = [previous['owner_pid'], (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]]
            state.update(status='WAITING_REGISTERED_GPU_OWNER_EXIT', upstream_GPU_owner=registered)
            save()
            while alive(registered):
                time.sleep(20)
            assert json.loads((upstream/'owner_state.json').read_text())['status'] == 'SAME_SEMANTIC_POLICY_THREE_ARM_CONTROLS_COMPLETE'
        wait_training('teacher')
        for arm, recipe in plan['arms'].items():
            if arm == 'native':
                wait_training('native')
            checkpoint = ROOT/recipe['checkpoint']
            if arm != 'before500':
                if plan.get('reuse_native_MM_validation'):
                    validated = json.loads((ROOT/recipe['training']/'checkpoint600_validation.json').read_text())
                    assert validated['status'] == 'REAL_ONEJEV600_NATIVE_MM_RELOAD_VALIDATED'
                    assert validated['repeated_reload_probability_max_difference'] == 0
                    assert Path(validated['checkpoint']).resolve() == checkpoint.resolve()
                else:
                    run(arm+'_NATIVE_MM_RELOAD', prefix+['tools/groundjev/check_onejev125_checkpoint.py',
                    '--step', '600', '--checkpoint', str(checkpoint), '--training-root', str(ROOT/recipe['training']),
                    '--reference-checkpoint', str(ROOT/plan['arms']['before500']['checkpoint'])])
            for scope, job in plan['scopes'].items():
                name = arm+'_'+scope
                try:
                    manifest_path = ROOT/job['manifest']
                    manifest = json.loads(manifest_path.read_text())
                    output = args.out/name
                    # Original500 MOT already uses this exact policy and candidate source.
                    if arm == 'before500' and job.get('reuse_original500'):
                        output = ROOT/job['reuse_original500']
                    else:
                        command = prefix+['tools/groundjev/run_video.py', '--task', manifest['task'],
                            '--video', str(manifest_path), '--backend', 'full', '--checkpoint', str(checkpoint),
                            '--evidence-format', 'v3_crop_motion', '--adaptive-context-groups',
                            '--candidate-replay', str(ROOT/job['candidates']), '--out', str(output)]
                        if plan.get('omit_feature_vectors'):
                            command += ['--omit-replay-feature-vectors']
                        if plan.get('memory_checkpoint_format'):
                            command += ['--memory-checkpoint-format', plan['memory_checkpoint_format'],
                                '--memory-export-interval', str(plan['memory_export_interval'])]
                        if manifest['task'] == 'rmot':
                            command += ['--native-rmot-membership']
                        elif manifest['task'] == 'ovmot':
                            command += ['--category-policy', 'native_hierarchical']
                        run(name+'_INFERENCE', command)
                    scorer = 'evaluate_video.py' if manifest['task'] == 'mot' else 'evaluate_multitask.py'
                    if output.parent == args.out:
                        run(name+'_OFFICIAL', ['python3', 'tools/groundjev/'+scorer,
                            '--video', str(manifest_path), '--run', str(output)])
                        run(name+'_EVENTS', ['python3', 'tools/groundjev/analyze_tracking_events.py',
                            '--video', str(manifest_path), '--run', str(output)])
                    rows = [json.loads(line) for line in (output/'frame_decisions.jsonl').open()]
                    assert [r['frame'] for r in rows] == manifest['frame_numbers']
                    with (ROOT/job['candidates']).open() as source:
                        for actual, line in zip_longest(rows, source):
                            assert actual is not None and line is not None
                            original = json.loads(line)
                            expected = original['candidates']
                            if plan.get('omit_feature_vectors'):
                                expected = [{k: v for k, v in c.items() if k not in
                                    ('native_region_feature', 'temporal_region_feature')} for c in expected]
                            assert actual['candidates'] == expected
                            assert actual['frame'] == original['frame'] and actual['image_path'] == original['image_path']
                    result = json.loads((output/'result.json').read_text())
                    assert result['all_input_frames_processed']
                    assert not result['gt_read_by_runtime'] and not result['future_frames_read_by_runtime']
                    decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of',
                        'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
                    assert decoded == len(rows)
                    state['completed'][name] = {'run': str(output), 'decoded_frames': decoded,
                        'all_exact_native_candidates_retained': True,
                        'reused_identical_completed_MOT_control': output.parent != args.out,
                        'official': json.loads((output/'official_metrics.json').read_text())}
                    save()
                except Exception as error:
                    state['failed'][name] = repr(error)
                    save()
                    # A failure in one registered task does not discard the other task results.
        state['status'] = ('SAME_SEMANTIC_POLICY_THREE_ARM_CONTROLS_COMPLETE' if not state['failed']
                           else 'SAME_POLICY_CONTROLS_FINISHED_WITH_RECORDED_FAILURES')
        save()
        if state['failed']:
            raise RuntimeError(state['failed'])
    except Exception as error:
        state.update(status='UNIFIED600_CONTROLS_OWNER_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
