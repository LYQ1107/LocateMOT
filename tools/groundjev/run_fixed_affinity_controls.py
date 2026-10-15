"""Frozen association/auxiliary interventions with identical native and text inputs."""
import argparse
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    regression = json.loads((ROOT/plan['regression']) .read_text())
    assert regression['status'] == 'REAL_AFFINITY_SEMANTICS_AUXILIARY_AND_LOSSLESS_MEMORY_VALIDATED'
    args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'STARTING_FIXED_AFFINITY_CONTROLS', 'owner_pid': os.getpid(),
        'plan': str(args.plan), 'device': 'cpu', 'additional_GPUs': 0,
        'new_feature_cache_files': False, 'completed': {}, 'full_goal_complete': False,
        'comparison': plan['comparison_limit'], 'memory_storage': plan['memory_storage']}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    python = str(ROOT/'.venv-onejev-py310/bin/python')

    def save():
        temporary = args.out/'owner_state.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.out/'owner_state.json')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, check=True)

    save()
    try:
        for scope, job in plan['scopes'].items():
            manifest, source = ROOT/job['manifest'], ROOT/job['candidates']
            m = json.loads(manifest.read_text())
            assert len(m['images']) == job['frames']
            for arm, policies in plan['arms'].items():
                name = scope+'_'+arm
                output = args.out/name
                run(name+'_INFERENCE', [python, 'tools/groundjev/run_video.py', '--task', m['task'],
                    '--video', str(manifest), '--backend', 'compact', '--compact-device', 'cpu',
                    '--checkpoint', str(ROOT/plan['checkpoint']), '--candidate-replay', str(source),
                    '--compact-association-policy', policies['association'],
                    '--compact-auxiliary-policy', policies['auxiliary'], '--omit-replay-feature-vectors',
                    '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0', '--out', str(output)])
                scorer = 'evaluate_video.py' if m['task'] == 'mot' else 'evaluate_multitask.py'
                run(name+'_OFFICIAL', ['python3', 'tools/groundjev/'+scorer, '--video', str(manifest), '--run', str(output)])
                run(name+'_EVENTS', ['python3', 'tools/groundjev/analyze_tracking_events.py',
                    '--video', str(manifest), '--run', str(output)])
                count, candidates, rejected = 0, 0, []
                with source.open() as native, (output/'frame_decisions.jsonl').open() as predictions:
                    for old, new in zip_longest(native, predictions):
                        assert old is not None and new is not None
                        old, new = json.loads(old), json.loads(new)
                        geometry = [{k: v for k, v in c.items() if k not in
                            ('native_region_feature', 'temporal_region_feature')} for c in old['candidates']]
                        assert new['candidates'] == geometry
                        assert new['frame'] == old['frame'] == m['frame_numbers'][count]
                        assert new['image_path'] == old['image_path'] == m['images'][count]
                        expected_rejection = bool(old['perception'].get('model_response_rejected', False))
                        assert bool(new['perception'].get('model_response_rejected', False)) == expected_rejection
                        if expected_rejection:
                            rejected.append(new['frame'])
                        assert all(len(c['native_region_feature']) == 2560 for c in old['candidates'])
                        candidates += len(geometry)
                        count += 1
                decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of',
                    'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
                assert count == decoded == job['frames'] and candidates == job['candidates_count']
                report = json.loads((output/'result.json').read_text())
                assert report['end_to_end_fps'] is None and report['all_input_frames_processed']
                assert not report['gt_read_by_runtime'] and not report['future_frames_read_by_runtime']
                assert report['memory_checkpoint_format'] == 'binary' and report['memory_export_interval'] == 0
                assert report['compact_association_policy'] == policies['association']
                assert report['compact_auxiliary_policy'] == policies['auxiliary']
                assert Path(report['final_identity_memory_path']).is_file()
                official = json.loads((output/'official_metrics.json').read_text())
                events = json.loads((output/'association_events.json').read_text())
                state['completed'][name] = {'run': str(output), 'policies': policies,
                    'decoded_frames': decoded, 'native_candidates': candidates, 'native_rejected_frames': rejected,
                    'all_native_geometry_labels_scores_order_and_frames_retained': True,
                    'metrics': official['metrics'], 'base_novel': official.get('base_novel'),
                    'correct_option_coverage': events.get('correct_option_coverage'),
                    'existing_association_accuracy': events.get('existing_association_accuracy'),
                    'speed_comparison_or_E2E_claim': False}
                save()
        state['status'] = 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE'
        save()
    except Exception as error:
        state.update(status='FIXED_AFFINITY_CONTROL_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
