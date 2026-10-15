"""One blocking GPU5 owner: fresh Compact RMOT, then identical-candidate controls."""
import argparse
import csv
import io
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
from tools.groundjev.run_unified600_controls import identity, alive
from tools.groundjev.service_process_group import stop_owned_service


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    archival = json.loads((ROOT/plan['archive_preflight_report']).read_text())
    assert archival['status'] == 'ACTUAL_NATIVE_ROI_ARCHIVE_REPLAY4_VALIDATED'
    assert archival['all_outputs_memory_state_geometry_exact']
    upstream = json.loads((ROOT/plan['training_owner']/'owner_state.json').read_text())
    registered = {arm: identity(upstream[arm+'_child_pid']) for arm in ('teacher', 'native')}
    state = {'status': 'WAITING_VERIFIED_SHARED600_GPU5_HANDOFF', 'owner_pid': os.getpid(),
        'training_children': registered, 'physical_gpu': plan['physical_gpu'], 'completed': {},
        'fresh_native_compact_not_replay': True, 'full_Goal_complete': False,
        'total_new_feature_cache_budget_bytes': plan['new_feature_cache_budget_bytes'],
        'GT_or_future_frames_read_by_runtime': False, 'old_U2_queue_restarted': False}
    state['actual_native_ROI_archive_replay_regression'] = archival
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    services, logs, process_groups = [], [], set()
    stop = threading.Event()
    resource = {'owned_process_group_peaks_MiB': {}, 'owned_GPU5_process_sum_peak_MiB': 0,
                'sampling_seconds': 5, 'errors': []}
    monitor_thread = None

    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command):
        state['status'] = stage
        save()
        with (args.out/(stage+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process_groups.add(child.pid)
            code = child.wait()
        if code:
            state['failed_stage'] = stage
            raise RuntimeError(f'{stage} failed with exit code {code}; preserve the completed causal prefix')

    def monitor():
        while not stop.is_set():
            try:
                text = subprocess.check_output(['nvidia-smi', '-i', str(plan['physical_gpu']),
                    '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
                total = 0
                groups = {}
                for pid, memory in csv.reader(io.StringIO(text)):
                    try:
                        group = os.getpgid(int(pid))
                        if group not in process_groups:
                            continue
                        used = int(memory)
                    except (ProcessLookupError, ValueError):
                        continue
                    key = str(group)
                    groups[key] = groups.get(key, 0)+used
                    total += used
                for key, used in groups.items():
                    resource['owned_process_group_peaks_MiB'][key] = max(
                        resource['owned_process_group_peaks_MiB'].get(key, 0), used)
                resource['owned_GPU5_process_sum_peak_MiB'] = max(resource['owned_GPU5_process_sum_peak_MiB'], total)
                (args.out/'owned_resources.json').write_text(json.dumps(resource, indent=2)+'\n')
            except Exception as error:
                resource['errors'].append(repr(error))
                return
            stop.wait(5)

    def score(name, manifest, output, source):
        run(name+'_OFFICIAL', ['python3', 'tools/groundjev/evaluate_multitask.py', '--video', str(manifest), '--run', str(output)])
        run(name+'_POSTHOC', ['python3', 'tools/groundjev/analyze_tracking_events.py', '--video', str(manifest), '--run', str(output)])
        m = json.loads(manifest.read_text())
        rows = [json.loads(line) for line in (output/'frame_decisions.jsonl').open()]
        assert [r['frame'] for r in rows] == m['frame_numbers']
        with source.open() as native:
            count = 0
            for original, row in zip(map(json.loads, native), rows):
                expected = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')}
                            for c in original['candidates']]
                assert row['candidates'] == expected and row['image_path'] == original['image_path']
                count += 1
        assert count == len(rows) == 800
        decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
        assert decoded == 800
        report = json.loads((output/'result.json').read_text())
        assert not report['gt_read_by_runtime'] and not report['future_frames_read_by_runtime']
        state['completed'][name] = {'run': str(output), 'decoded_frames': decoded,
            'all_native_candidates_and_rejected_frames_retained': True,
            'official': json.loads((output/'official_metrics.json').read_text()),
            'native_end_to_end_fps': report['end_to_end_fps'],
            'detector_excluded_replay_fps': report['association_replay_fps']}
        save()

    prefix = ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(plan['physical_gpu']),
              '--', str(ROOT/'.venv-onejev-py310/bin/python')]
    save()
    try:
        while any(alive(record) for record in registered.values()):
            time.sleep(20)
        for arm in ('teacher', 'native'):
            training = ROOT/plan['training'][arm]
            records = [json.loads(line) for line in (training/'log.jsonl').open()]
            done = next(row for row in records if row.get('event') == 'done')
            assert done['steps'] == 600 and done['examples'] == 2400
            assert json.loads((training/'latest_training_state.json').read_text())['step'] == 600
            run(arm+'_REAL600_MM_RELOAD_GPU5', prefix+['tools/groundjev/check_onejev125_checkpoint.py',
                '--step', '600', '--training-root', str(training), '--checkpoint', str(training/'ckpt-step600'),
                '--reference-checkpoint', str(ROOT/plan['full_checkpoints']['before500'])])
        state['GPU5_at_handoff'] = subprocess.check_output(['nvidia-smi', '-i', str(plan['physical_gpu']),
            '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
        for port in (plan['ground_port'], plan['feature_port']):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('127.0.0.1', port))
        commands = [('ground', ['tools/groundjev/serve_groundanything.py', '--port', str(plan['ground_port']),
                        '--mem-fraction-static', '.4', '--out', str(args.out/'ground_service')]),
                    ('features', ['tools/groundjev/serve_native_features.py', '--port', str(plan['feature_port']),
                        '--receipt', str(args.out/'feature_service.json')])]
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        for name, command in commands:
            log = (args.out/(name+'_server.log')).open('w')
            logs.append(log)
            child = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(plan['physical_gpu']),
                '--', str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), *command], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            services.append(child)
            process_groups.add(child.pid)
        state['owned_service_groups'] = [s.pid for s in services]
        state['status'] = 'WAITING_OWNED_NATIVE_RMOT_SERVICES'
        save()
        ground_url = f'http://127.0.0.1:{plan["ground_port"]}/v1'
        feature_url = f'http://127.0.0.1:{plan["feature_port"]}'
        deadline = time.monotonic()+600
        while True:
            if any(s.poll() is not None for s in services):
                raise RuntimeError('Owned RMOT native service exited before readiness')
            try:
                with urlopen(ground_url+'/models', timeout=5) as response:
                    ready = any(item['id'] == 'groundinganything' for item in json.load(response)['data'])
                if ready and (args.out/'feature_service.json').exists():
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Owned RMOT native service readiness timeout')
            time.sleep(2)
        used = 0
        sources = {}
        for scope, relative in plan['manifests'].items():
            manifest = ROOT/relative
            output = args.out/(scope+'_shared_compact_fresh')
            run(scope+'_FRESH_COMPACT_NATIVE800', prefix+['tools/groundjev/run_video.py', '--task', 'rmot',
                '--video', str(manifest), '--backend', 'compact', '--checkpoint', str(ROOT/plan['compact_checkpoint']),
                '--grounding-url', ground_url, '--grounding-parser', 'official_scoring',
                '--grounding-decoder-policy', 'tracking_dense', '--perception-failure-policy', 'empty_and_record',
                '--native-feature-url', feature_url, '--record-native-candidates', '--omit-replay-feature-vectors',
                '--native-candidate-byte-budget', str(plan['new_feature_cache_budget_bytes']-used), '--out', str(output)])
            source = output/'frame_candidates.jsonl'
            used += source.stat().st_size
            assert used <= plan['new_feature_cache_budget_bytes']
            sources[scope] = source
            state['new_feature_cache_bytes'] = used
            score(scope+'_shared_compact_fresh', manifest, output, source)
            report = json.loads((output/'result.json').read_text())
            assert report['end_to_end_fps'] is not None and report['association_replay_fps'] is None
            assert report['resumed_completed_frames'] == 0
        for service in services:
            stop_owned_service(service)
        services.clear()
        state['owned_native_services_stopped_before_replay_controls'] = True
        save()
        for scope, relative in plan['manifests'].items():
            manifest = ROOT/relative
            for arm, checkpoint in [('simple', None), *plan['full_checkpoints'].items()]:
                output = args.out/(scope+'_'+arm)
                command = prefix+['tools/groundjev/run_video.py', '--task', 'rmot', '--video', str(manifest),
                    '--backend', 'simple' if arm == 'simple' else 'full', '--candidate-replay', str(sources[scope]),
                    '--omit-replay-feature-vectors', '--out', str(output)]
                if arm != 'simple':
                    command += ['--checkpoint', str(ROOT/checkpoint), '--evidence-format', 'v3_crop_motion',
                                '--adaptive-context-groups', '--native-rmot-membership']
                run(scope+'_'+arm+'_MATCHED800', command)
                score(scope+'_'+arm, manifest, output, sources[scope])
        state['status'] = 'FULL_TWO800_RMOT_FRESH_COMPACT_AND_ALL_MATCHED_FULL_CONTROLS_COMPLETE'
        save()
    except Exception as error:
        state.update(status='FULL_RMOT_STUDY_FAILED', error=repr(error))
        save()
        raise
    finally:
        for service in services:
            stop_owned_service(service)
        for log in logs:
            log.close()
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join()
        (args.out/'owned_resources.json').write_text(json.dumps(resource, indent=2)+'\n')
        state['owned_native_service_groups_stopped'] = True
        save()


if __name__ == '__main__':
    main()
