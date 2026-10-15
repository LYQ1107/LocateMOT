"""Owned GPU7 native training clips, bounded exact ROI archives, no train GT input."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tools.groundjev.completed_gpu_handoff import validate_release
    from tools.groundjev.service_process_group import stop_owned_service
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT' and plan['physical_gpu'] == 7
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    release = validate_release(ROOT/plan['gpu_release'], 7)
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'PREPARING_BROADER_TRAIN_ONLY_NATIVE_CLIPS', 'owner_pid': os.getpid(),
        'owner_starttick': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        'physical_gpu': 7, 'verified_GPU7_release': release, 'completed': {},
        'actual_new_training_updates': 0, 'full_goal_complete': False}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    services, service_logs = [], []

    def save():
        temporary = out/'owner_state.tmp'; temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(out/'owner_state.json')
        print(json.dumps({'status': state['status'], 'completed_clips': len(state['completed']),
            'completed_frames': sum(r['frames'] for r in state['completed'].values()),
            'archive_bytes': sum(r['archive_bytes'] for r in state['completed'].values())}), flush=True)

    save()
    try:
        manifests = out/'manifests'; manifests.mkdir()
        clips = []
        for dataset, filename in plan['protocols'].items():
            protocol = json.loads((PROTOCOL/filename).read_text())
            for sequence in protocol['sequences']:
                if not sequence['train_frames']:
                    continue
                name = sequence['sequence']
                length = plan['longer_training_clips'].get(name, plan['frames_per_clip'][dataset])
                allowed = sorted(sequence['train_frames'], key=lambda f: int(Path(f).stem))
                assert len(allowed) >= length
                start = (len(allowed)-length)//2; selected = allowed[start:start+length]
                images = [str(Path(sequence['image_dir'])/f) for f in selected]
                assert all(Path(path).is_file() for path in images)
                frame_numbers = [int(Path(f).stem) for f in selected]
                assert all(b > a for a, b in zip(frame_numbers, frame_numbers[1:]))
                manifest = {'task': 'mot', 'dataset': dataset, 'sequence': name,
                    'scope': 'registered_train_only_native_source_not_benchmark_evaluation',
                    'vocabulary': ['person'], 'expression': None,
                    'fps': sequence['fps'], 'images': images, 'frame_numbers': frame_numbers,
                    'gt_path': sequence['gt_path'], 'source_train_protocol': str(PROTOCOL/filename),
                    'frame_selection': plan['selection'], 'physical_video_fps_verified': True}
                path = manifests/(name+'.json'); path.write_text(json.dumps(manifest, indent=2)+'\n')
                clips.append((name, path, manifest))
        assert len(clips) == plan['expected_videos']
        assert sum(len(m['images']) for _, _, m in clips) == plan['expected_frames']
        state.update(status='TRAIN_ONLY_43_CLIPS_1056_IMAGES_VALIDATED',
            planned_clips=len(clips), planned_frames=plan['expected_frames'], GT_read_to_select_frames=False)
        save()
        for port in (plan['ground_port'], plan['feature_port']):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); probe.bind(('127.0.0.1', port))
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '7', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if free >= 45*1024:
                break
            time.sleep(20)
        for name, command in [('ground', ['tools/groundjev/serve_groundanything.py', '--port', str(plan['ground_port']),
                '--mem-fraction-static', '.4', '--out', str(out/'ground_service')]),
                ('features', ['tools/groundjev/serve_native_features.py', '--port', str(plan['feature_port']),
                '--receipt', str(out/'feature_service.json')])]:
            log = (out/(name+'_service.log')).open('w'); service_logs.append(log)
            services.append(subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '7', '--',
                str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), *command], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        state.update(status='WAITING_OWNED_ORIGINAL_GROUND_AND_ROI_SERVICES', owned_service_groups=[s.pid for s in services]); save()
        ground_url, feature_url = f'http://127.0.0.1:{plan["ground_port"]}/v1', f'http://127.0.0.1:{plan["feature_port"]}'
        deadline = time.monotonic()+600
        while True:
            if any(s.poll() is not None for s in services):
                raise RuntimeError('Owned broad-fit service exited before readiness; retain launch logs')
            try:
                with urlopen(ground_url+'/models', timeout=5) as response:
                    ready = any(m['id'] == 'groundinganything' for m in json.load(response)['data'])
                if ready and (out/'feature_service.json').is_file():
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Owned broad-fit services did not become ready')
            time.sleep(2)
        python = str(ROOT/'.venv-onejev-py310/bin/python')
        for name, manifest_path, manifest in clips:
            output = out/'clips'/name; output.parent.mkdir(exist_ok=True)
            used = sum(r['archive_bytes'] for r in state['completed'].values())
            remaining = plan['native_candidate_archive_budget_bytes']-used
            assert remaining > 0
            command = [python, 'tools/groundjev/run_video.py', '--task', 'mot', '--video', str(manifest_path),
                '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint', str(ROOT/plan['checkpoint']),
                '--grounding-url', ground_url, '--grounding-parser', 'official_scoring',
                '--grounding-decoder-policy', 'tracking_dense', '--perception-failure-policy', 'empty_and_record',
                '--native-feature-url', feature_url, '--record-native-candidates', '--native-candidate-format', 'packed_npz_f32',
                '--native-candidate-byte-budget', str(remaining), '--omit-replay-feature-vectors',
                '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0', '--out', str(output)]
            state.update(status='COLLECTING_'+name, current_clip=name); save()
            with (out/(name+'.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                state['current_child_pid'] = child.pid; save()
                for line in child.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                assert child.wait() == 0, f'{name} failed; preserve causal prefix and original frames'
            state.pop('current_child_pid', None)
            runtime = json.loads((output/'result.json').read_text())
            assert runtime['frames'] == len(manifest['images']) and runtime['all_input_frames_processed']
            assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
            assert runtime['decision_device'] == 'cpu' and runtime['resumed_completed_frames'] == 0
            assert runtime['native_candidate_archive_format'] == 'packed_npz_f32'
            assert runtime['native_candidate_archive_bytes'] <= remaining
            state['completed'][name] = {'run': str(output), 'frames': runtime['frames'],
                'archive_bytes': runtime['native_candidate_archive_bytes'],
                'native_candidates': sum(len(json.loads(line)['candidates']) for line in (output/'frame_candidates.jsonl').open()),
                'rejected_frames': len(runtime['perception_rejected_frames'])}
            save()
        assert len(state['completed']) == 43 and sum(r['frames'] for r in state['completed'].values()) == 1056
        state.update(status='BROADER_NATIVE_TRAIN43_VIDEO1056_FRAME_COLLECTION_COMPLETE',
            combined_registered_feature_upper_bytes=plan['combined_registered_feature_upper_bytes'])
        state.pop('current_clip', None); save()
    except Exception as error:
        state.update(status='BROADER_NATIVE_TRAIN_COLLECTION_FAILED', error=repr(error)); save()
        raise
    finally:
        for service in services:
            stop_owned_service(service)
        for log in service_logs:
            log.close()


if __name__ == '__main__':
    main()
