"""GPU7 retention handoff, bounded native challenge archive, then CPU memory controls."""
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


def main():
    from tools.groundjev.run_unified600_controls import alive
    from tools.groundjev.reattach_native_full9 import alive as client_alive
    from tools.groundjev.service_process_group import members, stop_owned_service
    from tools.groundjev.run_video import ReplayFrames
    from tools.groundjev.analyze_natural_recovery_case import analyze
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    proof = json.loads((ROOT/plan['archive_validation']).read_text())
    assert proof['status'] == 'REAL21_PACKED_NATIVE_ROI_ARCHIVE_REPLAY_AND_NEXT_FRAME_EXACT'
    smoke = json.loads((ROOT/plan['scheduler_validation']/'owner_state.json').read_text())
    assert smoke['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(smoke['completed']) == 2
    posthoc = json.loads((ROOT/plan['natural_posthoc_validation']).read_text())
    assert posthoc['status'] == 'ACTUAL_COMPLETED800_NATURAL_GAP_POSTHOC_FUNCTIONAL_VALIDATED'
    args.out.mkdir(parents=True, exist_ok=False)
    retention = ROOT/plan['retention_owner']
    upstream = json.loads((retention/'owner_state.json').read_text())
    proc = Path('/proc')/str(upstream['owner_pid'])
    identity = None
    if proc.exists():
        assert (proc/'cwd').resolve() == ROOT
        assert 'tools/groundjev/reattach_native_full9.py' in (proc/'cmdline').read_bytes().decode().split('\0')
        identity = [upstream['owner_pid'], (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]]
    clients = [c for c in json.loads((ROOT/plan['client_contract']).read_text()) if ':8101' in c['port_or_url']]
    assert clients
    state = {'status': 'WAITING_ORIGINAL_NATIVE_RETENTION_GPU7_RELEASE', 'owner_pid': os.getpid(),
        'physical_gpu': 7, 'registered_retention_owner': identity, 'completed': {},
        'new_archive_cap_bytes': plan['archive_cap_bytes'], 'old_U2_queue_restarted': False,
        'GT_read_by_runtime': False, 'artificial_occlusion_or_visibility_filter': False,
        'full_goal_complete': False}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    services, service_logs, groups = [], [], set()
    stop = threading.Event()
    resource = {'sampling_seconds': 5, 'owned_group_peaks_MiB': {}, 'owned_GPU7_sum_peak_MiB': 0, 'errors': [],
        'scope': 'fresh native source only; five-second samples are a lower bound on instantaneous peak'}

    def save():
        temporary = args.out/'owner_state.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(args.out/'owner_state.json')
        print(json.dumps({'status': state['status'], 'completed': list(state['completed'])}), flush=True)

    def run(stage, command):
        state['status'] = stage
        with (args.out/(stage+'.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            groups.add(child.pid)
            state['current_child'] = [child.pid, (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]]
            save()
            code = child.wait()
            if code:
                raise RuntimeError(f'{stage} exited {code}; retain its causal prefix and log')
        state.pop('current_child', None)

    def monitor():
        while not stop.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi', '-i', '7', '--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'], text=True)
                total, by_group = 0, {}
                for pid, memory in csv.reader(io.StringIO(raw)):
                    try:
                        group = os.getpgid(int(pid)); used = int(memory)
                    except (ProcessLookupError, ValueError):
                        continue
                    if group in groups:
                        by_group[str(group)] = by_group.get(str(group), 0)+used
                        total += used
                for group, used in by_group.items():
                    resource['owned_group_peaks_MiB'][group] = max(resource['owned_group_peaks_MiB'].get(group, 0), used)
                resource['owned_GPU7_sum_peak_MiB'] = max(resource['owned_GPU7_sum_peak_MiB'], total)
                (args.out/'fresh_source_resources.json').write_text(json.dumps(resource, indent=2)+'\n')
            except Exception as error:
                resource['errors'].append(repr(error))
                return
            stop.wait(5)

    save()
    thread = None
    try:
        while True:
            current = json.loads((retention/'owner_state.json').read_text())
            release = current['completed'].get('original')
            if release and not any(client_alive(c) for c in clients):
                assert release['all_group_members_gone'] and not members(release['owned_service_group_stopped'])
                state['upstream_GPU7_retention_release'] = release
                break
            if not alive(identity):
                raise RuntimeError('Retention owner ended without verified original GPU7 release; do not stop or duplicate its jobs')
            time.sleep(20)
        state['GPU7_at_handoff'] = subprocess.check_output(['nvidia-smi', '-i', '7',
            '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True).strip()
        for port in (plan['ground_port'], plan['feature_port']):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); probe.bind(('127.0.0.1', port))
        for name, command in [('ground', ['tools/groundjev/serve_groundanything.py', '--port', str(plan['ground_port']),
                '--mem-fraction-static', '.4', '--out', str(args.out/'ground_service')]),
                ('features', ['tools/groundjev/serve_native_features.py', '--port', str(plan['feature_port']),
                '--receipt', str(args.out/'feature_service.json')])]:
            log = (args.out/(name+'_service.log')).open('w'); service_logs.append(log)
            child = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '7', '--',
                str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), *command], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            services.append(child); groups.add(child.pid)
        state.update(status='WAITING_OWNED_NATIVE_CHALLENGE_SERVICES', owned_service_groups=[s.pid for s in services]); save()
        ground_url, feature_url = f'http://127.0.0.1:{plan["ground_port"]}/v1', f'http://127.0.0.1:{plan["feature_port"]}'
        deadline = time.monotonic()+600
        while True:
            if any(s.poll() is not None for s in services):raise RuntimeError('Owned challenge service exited before readiness')
            try:
                with urlopen(ground_url+'/models', timeout=5) as response:
                    ready = any(m['id'] == 'groundinganything' for m in json.load(response)['data'])
                if ready and (args.out/'feature_service.json').is_file():break
            except OSError:
                pass
            if time.monotonic() > deadline:raise RuntimeError('Owned challenge service readiness timeout')
            time.sleep(2)
        manifest_path = ROOT/plan['manifest']
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest['images']) == 1657 and manifest['frame_numbers'][0] == 1659 and manifest['frame_numbers'][-1] == 3315
        output = args.out/'fresh_compact_gated3'
        thread = threading.Thread(target=monitor, daemon=True); thread.start()
        run('FRESH_NATIVE_MOT20_1657', ['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '7', '--',
            str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py', '--task', 'mot',
            '--video', str(manifest_path), '--backend', 'compact', '--checkpoint', str(ROOT/plan['checkpoint']),
            '--grounding-url', ground_url, '--grounding-parser', 'official_scoring', '--grounding-decoder-policy', 'tracking_dense',
            '--perception-failure-policy', 'empty_and_record', '--native-feature-url', feature_url,
            '--record-native-candidates', '--native-candidate-format', 'packed_npz_f32',
            '--native-candidate-byte-budget', str(plan['archive_cap_bytes']), '--omit-replay-feature-vectors',
            '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0', '--out', str(output)])
        stop.set(); thread.join()
        for service in services:stop_owned_service(service)
        services.clear()
        runtime = json.loads((output/'result.json').read_text())
        assert runtime['frames'] == 1657 and runtime['resumed_completed_frames'] == 0 and runtime['end_to_end_fps'] is not None
        assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert runtime['native_candidate_archive_format'] == 'packed_npz_f32' and runtime['native_candidate_archive_feature_files'] == 1657
        source = Path(runtime['native_candidate_archive'])
        total = source.stat().st_size+sum(p.stat().st_size for p in (source.parent/'native_region_features').glob('*.npz'))
        assert total == runtime['native_candidate_archive_bytes'] <= plan['archive_cap_bytes']
        replay, count = ReplayFrames(source), 0
        with (output/'frame_decisions.jsonl').open() as trace:
            for index, line in enumerate(trace):
                row = json.loads(line); raw = replay.frame(row['frame'])
                expected = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')} for c in raw['candidates']]
                assert row['frame'] == manifest['frame_numbers'][index] and row['image_path'] == manifest['images'][index]
                assert row['candidates'] == expected
                count += len(raw['candidates'])
            assert index+1 == 1657
        replay.close()
        decoded = int(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(output/'tracks.mp4')], text=True))
        assert decoded == 1657
        run('FRESH_OFFICIAL', ['python3', 'tools/groundjev/evaluate_video.py', '--video', str(manifest_path), '--run', str(output)])
        run('FRESH_POSTHOC', ['python3', 'tools/groundjev/analyze_tracking_events.py', '--video', str(manifest_path), '--run', str(output)])
        selected = analyze(manifest, output)
        state['completed']['fresh_source'] = {'run': str(output), 'decoded_frames': decoded, 'native_candidates': count, 'archive_bytes': total,
            'metrics': json.loads((output/'official_metrics.json').read_text())['metrics'],
            'native_pipeline_FPS': runtime['end_to_end_fps'], 'resources': resource,
            'natural_case_counts': selected['counts']}
        save()
        controls = json.loads((ROOT/plan['controls_blueprint']).read_text())
        controls['scopes']['mot20_challenge']['candidates_count'] = count
        controls['scopes']['mot20_challenge']['candidates'] = str(source.resolve())
        resolved = args.out/'resolved_controls_plan.json'; resolved.write_text(json.dumps(controls, indent=2)+'\n')
        run('CPU_FROZEN_MEMORY_CONTROLS', ['python3', 'tools/groundjev/run_fixed_affinity_controls.py',
            '--plan', str(resolved), '--out', str(args.out/'controls')])
        complete = json.loads((args.out/'controls/owner_state.json').read_text())
        assert complete['status'] == 'ALL_FIXED_AFFINITY_CONTROLS_COMPLETE' and len(complete['completed']) == len(controls['arms'])
        summaries = {}
        for name, record in complete['completed'].items():
            recovery = analyze(manifest, ROOT/record['run'])
            summaries[name] = dict(record, natural_recovery=recovery)
        state['completed']['memory_controls'] = summaries
        state['CPU_gated3_accuracy_matches_fresh_GPU7'] = summaries['mot20_challenge_reference_gated3']['metrics'] == state['completed']['fresh_source']['metrics']
        state.update(status='NATURAL_MOT20_SCOPE_AND_FROZEN_MEMORY_CONTROLS_COMPLETE',
            interpretation='Report actual candidate-gap support and all official metrics; registered low-visibility interval alone is not proof of missing candidates or successful recovery.')
        save()
    except Exception as error:
        state.update(status='NATURAL_MEMORY_STUDY_FAILED', error=repr(error)); save(); raise
    finally:
        stop.set()
        if thread:thread.join()
        for service in services:stop_owned_service(service)
        for log in service_logs:log.close()


if __name__ == '__main__':
    main()
