"""Fresh Full/Compact on the same complete video and GPU; never estimate E2E by replay."""
import argparse
import csv
import io
from itertools import zip_longest
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.groundjev.run_unified600_controls import alive
from tools.groundjev.service_process_group import members, stop_owned_service


def canonical(row):
    return {'frame': row['frame'], 'image_path': row['image_path'],
        'candidates': [{k: c.get(k) for k in ('box', 'label', 'native_score', 'score_kind')}
                       for c in row['candidates']],
        'parser_rejected': bool(row['perception'].get('model_response_rejected', False))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    upstream = ROOT/plan['upstream']
    prior = json.loads((upstream/'owner_state.json').read_text())
    pid = prior['owner_pid']
    proc = Path('/proc')/str(pid)
    registered = None
    if proc.exists():
        assert (proc/'cwd').resolve() == ROOT
        assert 'tools/groundjev/run_full_rmot_study.py' in (proc/'cmdline').read_bytes().decode().split('\0')
        registered = [pid, (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]]
    manifest = ROOT/plan['manifest']
    video = json.loads(manifest.read_text())
    assert video['task'] == 'rmot' and len(video['images']) == plan['frames'] == 800
    state = {'status': 'WAITING_VERIFIED_GPU5_FULL_RMOT_OWNER_EXIT', 'owner_pid': os.getpid(),
        'upstream_owner': registered, 'physical_gpu': plan['physical_gpu'], 'completed': {},
        'candidate_comparison': None, 'new_feature_cache_budget_bytes': plan['feature_budget_bytes'],
        'all_input_frames_including_empty_and_rejected_retained': True, 'full_goal_complete': False}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    python = str(ROOT/'.venv-onejev-py310/bin/python')
    prefix = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(plan['physical_gpu']), '--']

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command, groups=None):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if groups is not None:
                groups.add(child.pid)
            code = child.wait()
        if code:
            state['failed_stage'] = stage
            raise RuntimeError(f'{stage} failed with exit code {code}; preserve the real prefix')

    save()
    try:
        while alive(registered):
            time.sleep(20)
        prior = json.loads((upstream/'owner_state.json').read_text())
        assert prior['status'] == 'FULL_TWO800_RMOT_FRESH_COMPACT_AND_ALL_MATCHED_FULL_CONTROLS_COMPLETE'
        assert prior['owned_native_service_groups_stopped']
        assert all(not members(group) for group in prior['owned_service_groups'])
        used = 0
        for arm, recipe in plan['arms'].items():
            directory = args.out/arm
            directory.mkdir()
            groups, services, logs = set(), [], []
            stop = threading.Event()
            resource = {'scope': arm, 'sampling_seconds': 5, 'sampled_owned_GPU_sum_peak_MiB': 0,
                'sampled_group_peaks_MiB': {}, 'errors': [], 'peak_kind': '5second sample lower bound, not exact instantaneous maximum'}
            def monitor():
                while not stop.is_set():
                    try:
                        raw = subprocess.check_output(['nvidia-smi', '-i', str(plan['physical_gpu']),
                            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
                        totals = {}
                        for current, memory in csv.reader(io.StringIO(raw)):
                            try:
                                group = os.getpgid(int(current))
                                if group not in groups:
                                    continue
                                totals[group] = totals.get(group, 0)+int(memory)
                            except (ProcessLookupError, ValueError):
                                continue
                        resource['sampled_owned_GPU_sum_peak_MiB'] = max(
                            resource['sampled_owned_GPU_sum_peak_MiB'], sum(totals.values()))
                        for group, total in totals.items():
                            key = str(group)
                            resource['sampled_group_peaks_MiB'][key] = max(resource['sampled_group_peaks_MiB'].get(key, 0), total)
                        (directory/'resources.json').write_text(json.dumps(resource, indent=2)+'\n')
                    except Exception as error:
                        resource['errors'].append(repr(error))
                        return
                    stop.wait(5)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                hardware = subprocess.check_output(['nvidia-smi', '-i', str(plan['physical_gpu']),
                    '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
                ports = [plan['ground_port']]+([plan['feature_port']] if recipe['backend'] == 'compact' else [])
                for port in ports:
                    with socket.socket() as probe:
                        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        probe.bind(('127.0.0.1', port))
                commands = [('ground', ['tools/groundjev/serve_groundanything.py', '--port', str(plan['ground_port']),
                    '--mem-fraction-static', '.4', '--out', str(directory/'ground_service')])]
                if recipe['backend'] == 'compact':
                    commands.append(('features', ['tools/groundjev/serve_native_features.py', '--port', str(plan['feature_port']),
                        '--receipt', str(directory/'feature_service.json')]))
                service_start = time.perf_counter()
                for name, command in commands:
                    log = (directory/(name+'_server.log')).open('w')
                    logs.append(log)
                    child = subprocess.Popen(prefix+[str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), *command],
                        cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    services.append(child)
                    groups.add(child.pid)
                state['current_owned_service_groups'] = [s.pid for s in services]
                state['status'] = 'WAITING_PRIVATE_FRESH_'+arm.upper()+'_SERVICES'
                save()
                ground_url = f'http://127.0.0.1:{plan["ground_port"]}/v1'
                deadline = time.monotonic()+600
                while True:
                    if any(s.poll() is not None for s in services):
                        raise RuntimeError('Private E2E native service exited before readiness')
                    try:
                        with urlopen(ground_url+'/models', timeout=5) as response:
                            ready = any(item['id'] == 'groundinganything' for item in json.load(response)['data'])
                        features = recipe['backend'] != 'compact' or (directory/'feature_service.json').is_file()
                        if ready and features:
                            break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError('Private E2E service readiness timeout')
                    time.sleep(2)
                startup = time.perf_counter()-service_start
                output = directory/'video'
                command = prefix+[python, 'tools/groundjev/run_video.py', '--task', 'rmot', '--video', str(manifest),
                    '--backend', recipe['backend'], '--checkpoint', str(ROOT/recipe['checkpoint']),
                    '--grounding-url', ground_url, '--grounding-parser', 'official_scoring',
                    '--grounding-decoder-policy', 'tracking_dense', '--perception-failure-policy', 'empty_and_record',
                    '--record-native-candidates', '--omit-replay-feature-vectors',
                    '--native-candidate-byte-budget', str(plan['feature_budget_bytes']-used), '--out', str(output)]
                if recipe['backend'] == 'full':
                    command += ['--evidence-format', 'v3_crop_motion', '--adaptive-context-groups', '--native-rmot-membership']
                else:
                    command += ['--native-feature-url', f'http://127.0.0.1:{plan["feature_port"]}']
                run(arm+'_ACTUAL_FRESH800', command, groups)
                report = json.loads((output/'result.json').read_text())
                assert report['frames'] == 800 and report['all_input_frames_processed']
                assert report['resumed_completed_frames'] == 0 and report['association_replay_fps'] is None
                assert report['end_to_end_fps'] == 800/report['complete_video_wall_seconds']
                assert not report['gt_read_by_runtime'] and not report['future_frames_read_by_runtime']
                used += report['native_candidate_archive_bytes']
                assert used <= plan['feature_budget_bytes']
                state['new_feature_cache_bytes'] = used
                # Stop all owned model services before CPU scoring and before
                # constructing the next arm's independently initialized service.
                for service in services:
                    stop_owned_service(service)
                services.clear()
                run(arm+'_OFFICIAL', ['python3', 'tools/groundjev/evaluate_multitask.py', '--video', str(manifest), '--run', str(output)])
                rows = [json.loads(line) for line in (output/'frame_decisions.jsonl').open()]
                assert [r['frame'] for r in rows] == video['frame_numbers']
                assert [r['image_path'] for r in rows] == video['images']
                decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                    '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
                assert decoded == 800
                state['completed'][arm] = {'hardware_at_launch': hardware, 'run': str(output), 'decoded_frames': decoded,
                    'actual_end_to_end_fps': report['end_to_end_fps'], 'pipeline_wall_seconds': report['complete_video_wall_seconds'],
                    'preloaded_services_startup_seconds_excluded': startup,
                    'core_initialization_seconds_excluded': report['model_initialization_seconds'],
                    'core_Torch_allocator_peak_bytes': report['peak_decision_gpu_memory_bytes'],
                    'core_process_peak_RSS_bytes': report['peak_decision_process_rss_bytes'],
                    'trainable_compact_core_parameters': report.get('trainable_compact_core_parameters'),
                    'frozen_CPU_text_table': report.get('frozen_CPU_text_table'),
                    'native_ROI_service': json.loads((directory/'feature_service.json').read_text()) if recipe['backend']=='compact' else None,
                    'official': json.loads((output/'official_metrics.json').read_text())}
                save()
            finally:
                for service in services:
                    stop_owned_service(service)
                for log in logs:
                    log.close()
                stop.set()
                thread.join()
                (directory/'resources.json').write_text(json.dumps(resource, indent=2)+'\n')
                if arm in state['completed']:
                    state['completed'][arm]['sampled_resources'] = resource
                state['current_owned_service_groups_stopped'] = True
                save()
        left = args.out/'full_native600/video/frame_candidates.jsonl'
        right = args.out/'compact/video/frame_candidates.jsonl'
        differences = []
        with left.open() as a, right.open() as b:
            count = 0
            for first, second in zip_longest(a, b):
                assert first is not None and second is not None
                first, second = json.loads(first), json.loads(second)
                if canonical(first) != canonical(second):
                    differences.append(first['frame'])
                count += 1
        assert count == 800
        state['candidate_comparison'] = {'frames': count, 'different_frames': differences,
            'canonical_fields': 'frame/image/box/label/native_score/score_kind/parserrejection; ROImetadata intentionally excluded',
            'same_actual_detector_candidates': not differences}
        assert len({item['hardware_at_launch'].split(',')[0].strip() for item in state['completed'].values()}) == 1
        assert all(not item['sampled_resources']['errors'] and item['sampled_resources']['sampled_owned_GPU_sum_peak_MiB'] > 0
                   for item in state['completed'].values())
        if differences:
            raise RuntimeError('Strict fresh detector pairing failed; preserve both complete runs, do not claim controlled association comparison')
        state['status'] = 'SAME800_SAME_GPU_FRESH_FULL_COMPACT_E2E_AND_DETECTOR_PAIRING_VALIDATED'
        save()
    except Exception as error:
        state.update(status='PAIRED_TRACKING_E2E_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
