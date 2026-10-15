"""One CPU owner for the four registered task mixtures and official video controls."""
import argparse
from itertools import zip_longest
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
    import torch
    from locatemot.groundjev.compact import CompactCore
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-study', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = CompactCore(semantic=True)
    initialization = ((args.resume_study or args.out)/'shared_MOT300_initialization.pt')
    initial = torch.load(initialization if args.resume_study else ROOT/plan['initialization'], map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(initial['state_dict'], strict=False)
    assert not unexpected and all(k.startswith(('text.', 'category.', 'membership.', 'class_scale')) for k in missing)
    if not args.resume_study:
        torch.save({'architecture': 'set', 'semantic': True, 'state_dict': model.state_dict(),
            'source': plan['initialization'], 'semantic_initialization_seed': 0,
            'prior_RMOT_or_OVMOT_training': False}, initialization)
    expected_parameters = sum(p.numel() for p in model.parameters())
    del model, initial
    upstream = ROOT/plan['full_RMOT_owner']
    pid = json.loads((upstream/'owner_state.json').read_text())['owner_pid']
    proc = Path('/proc')/str(pid)
    registered = None
    if proc.exists():
        argv = (proc/'cmdline').read_bytes().decode().split('\0')
        assert 'tools/groundjev/run_full_rmot_study.py' in argv and (proc/'cwd').resolve() == ROOT
        registered = [pid, (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]]
    state = {'status': 'STARTING_REGISTERED_CPU_TASK_SUBSETS', 'owner_pid': os.getpid(),
        'device': 'cpu', 'additional_GPUs': 0, 'completed_training': {}, 'completed_controls': {}, 'failed': {},
        'shared_initialization': str(initialization), 'identical_parameter_count': expected_parameters,
        'full_RMOT_source_owner': registered, 'new_feature_cache_created': False,
        'full_goal_complete': False, 'fixed_total_updates_not_per_task_exposure': True}
    training_root = args.resume_study or args.out
    state['resume_study'] = str(args.resume_study) if args.resume_study else None
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    python = str(ROOT/'.venv-onejev-py310/bin/python')

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'trained': list(state['completed_training']),
            'controls': list(state['completed_controls']), 'failed': list(state['failed'])}), flush=True)

    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def control(arm, scope, job):
        name = arm+'_'+scope
        try:
            manifest = ROOT/job['manifest']
            m = json.loads(manifest.read_text())
            source = ROOT/job['candidates']
            output = args.out/name
            old = args.resume_study/name if args.resume_study else None
            if old is not None and (old/'result.json').is_file():
                output = old
            else:
                command = [python, 'tools/groundjev/run_video.py', '--task', m['task'], '--video', str(manifest),
                '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint',
                str(training_root/(arm+'_training')/'checkpoint.pt'), '--candidate-replay', str(source),
                '--omit-replay-feature-vectors', '--out', str(output)]
                if old is not None and (old/'frame_decisions.jsonl').is_file():
                    command += ['--resume-from', str(old)]
                run(name+'_INFERENCE', command)
            scorer = 'evaluate_video.py' if m['task'] == 'mot' else 'evaluate_multitask.py'
            run(name+'_OFFICIAL', ['python3', 'tools/groundjev/'+scorer, '--video', str(manifest), '--run', str(output)])
            run(name+'_EVENTS', ['python3', 'tools/groundjev/analyze_tracking_events.py', '--video', str(manifest), '--run', str(output)])
            count = 0
            with source.open() as native, (output/'frame_decisions.jsonl').open() as predictions:
                for old, new in zip_longest(native, predictions):
                    assert old is not None and new is not None
                    old, new = json.loads(old), json.loads(new)
                    geometry = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')}
                                for c in old['candidates']]
                    assert new['candidates'] == geometry and new['frame'] == old['frame'] == m['frame_numbers'][count]
                    assert new['image_path'] == old['image_path'] == m['images'][count]
                    count += 1
            assert count == len(m['images'])
            decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
            assert decoded == count
            result = json.loads((output/'result.json').read_text())
            assert result['end_to_end_fps'] is None and result['all_input_frames_processed']
            assert not result['gt_read_by_runtime'] and not result['future_frames_read_by_runtime']
            official = json.loads((output/'official_metrics.json').read_text())
            events = json.loads((output/'association_events.json').read_text())
            state['completed_controls'][name] = {'run': str(output), 'decoded_frames': decoded,
                'metrics': official['metrics'], 'base_novel': official.get('base_novel'),
                'all_exact_candidate_geometry_order_and_frames_retained': True,
                'correct_option_coverage': events.get('correct_option_coverage'),
                'existing_association_accuracy': events.get('existing_association_accuracy')}
            save()
        except Exception as error:
            state['failed'][name] = repr(error)
            save()

    save()
    try:
        if args.resume_study:
            handoff = json.loads((args.resume_study/'manager_handoff.json').read_text())
            state['status'] = 'WAITING_PRESERVED_PREVIOUS_OWNER_WORKERS'
            state['manager_handoff'] = handoff
            save()
            while members(handoff['old_owner_process_group']):
                time.sleep(5)
        training_records = {}
        for arm, tasks in plan['arms'].items():
            training = training_root/(arm+'_training')
            if not args.resume_study:
                run(arm+'_TRAINING', [python, 'tools/groundjev/train_compact_joint.py', '--pack', str(ROOT/plan['source_pack']),
                '--out', str(training), '--checkpoint', str(initialization), '--steps', str(plan['steps_each']),
                '--device', 'cpu', '--tasks', *tasks, '--per-task-sampling'])
            trained = json.loads((training/'owner_state.json').read_text())
            assert trained['status'] == 'SHARED_COMPACT_JOINT1200_TRAINING_COMPLETE'
            assert trained['steps'] == 1200 and trained['parameters'] == expected_parameters
            assert all(value == 0 for value in trained['three_task_checkpoint_reload_max_difference'].values())
            assert trained['changed_elements'] > 0
            training_records[arm] = trained
            state['completed_training'][arm] = {k: v for k, v in trained.items() if k != 'sampled_example_ids'}
            save()
        common = {}
        for task in ('0', '1', '2'):
            draws = [r['sampled_example_ids'][task] for r in training_records.values() if task in r['sampled_example_ids']]
            length = min(map(len, draws))
            assert all(draw[:length] == draws[0][:length] for draw in draws)
            common[task] = length
        state['common_per_task_actual_example_ID_prefixes_exact'] = common
        save()
        for scope, job in plan['known_controls'].items():
            for arm in plan['arms']:
                control(arm, scope, job)
        # This wait belongs to the sole blocking owner. A completed source is
        # sufficient even while the independent GPU5 Full controls keep running.
        for scope, job in plan['future_RMOT_controls'].items():
            state['status'] = 'WAITING_VERIFIED_FRESH_'+scope.upper()+'_SOURCE'
            save()
            result_path = ROOT/job['fresh_compact_result']
            while not result_path.is_file():
                if not alive(registered):
                    raise RuntimeError('Full RMOT source owner exited without the registered complete source; preserve completed subset controls')
                time.sleep(20)
            source_result = json.loads(result_path.read_text())
            assert source_result['all_input_frames_processed'] and source_result['frames'] == 800
            assert source_result['resumed_completed_frames'] == 0 and source_result['native_candidate_archive']
            for arm in plan['arms']:
                control(arm, scope, job)
        state['status'] = ('ALL_FOUR_TASK_SUBSETS_AND_REGISTERED_OFFICIAL_CONTROLS_COMPLETE' if not state['failed']
                           else 'TASK_SUBSETS_FINISHED_WITH_RECORDED_FAILED_UNITS')
        save()
        if state['failed']:
            raise RuntimeError(state['failed'])
    except Exception as error:
        state.update(status='COMPACT_TASK_SUBSET_OWNER_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
