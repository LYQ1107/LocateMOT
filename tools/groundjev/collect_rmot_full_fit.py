"""One blocking owner: explicitly allocate native query collection on allowed GPUs."""
import argparse
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tools.groundjev.run_unified600_controls import alive
    from tools.groundjev.service_process_group import members, stop_owned_service

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    gpu = plan['physical_gpu']
    assert plan['project_root'] == '/data2/user/LocateMOT' and gpu in (6, 7)
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    scope = ROOT/plan['scope_output']
    validation = json.loads((scope/'validation.json').read_text())
    assert validation['status'] == 'ALL_FIT1814_PHYSICAL_FRAMES_18_NATIVE_QUERY_SCOPES_VALIDATED_NOT_COLLECTED'
    assert validation['counts']['physical_frames'] == plan['expected_physical_frames']
    assert validation['counts']['query_frame_contexts'] == plan['expected_query_frame_contexts']
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    shared = plan.get('co_location')
    state = dict(status='VALIDATING_REGISTERED_SHARED_GPU7_ALLOCATION' if shared else 'CPU_WAITING_REGISTERED_GPU6_RETENTION_OWNER', owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        physical_gpu=gpu, upstream_owner_identity=plan.get('upstream_owner_identity'),
        actual_new_training_updates=0, completed={}, full_goal_complete=False)

    def save():
        temporary = out/'owner_state.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'], completed_rollouts=len(state['completed']),
            completed_query_frame_contexts=sum(r['frames'] for r in state['completed'].values()),
            archive_bytes=sum(r['archive_bytes'] for r in state['completed'].values()))), flush=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    services, service_logs = [], []
    save()
    try:
        if shared:
            assert gpu == 7 and plan['ground_mem_fraction_static'] == .3
            registered = shared['training_owner_identity']
            assert alive(registered)
            proc = Path('/proc')/str(registered[0])
            assert (proc/'cwd').resolve() == ROOT
            assert shared['training_owner_script'] in (proc/'cmdline').read_bytes().decode().split('\0')
            training = json.loads((ROOT/shared['training_owner']/'owner_state.json').read_text())
            assert training['status'] == 'SHARED_ONEJEV6000_TRAINING'
            child = training['child_pid']; child_proc = Path('/proc')/str(child)
            child_fields = (child_proc/'stat').read_text().rsplit(')', 1)[1].split()
            assert child_fields[0] != 'Z'
            child_args = (child_proc/'cmdline').read_bytes().decode().split('\0')
            assert 'tools/groundjev/run_fla_group_reference_sft.py' in child_args
            # run_with_resources is a wrapper; CUDA visibility is passed to
            # its training child, so verify the registered physical GPU argv.
            assert '--gpus' in child_args and child_args[child_args.index('--gpus')+1] == '7'
            allocation = dict(status='VERIFIED_SHARED_GPU7_ALLOCATION_WITH_EXISTING_TRAINING',
                physical_gpu=7, registered_training_owner=registered,
                registered_training_child=[child, child_fields[19]], new_physical_GPUs=0,
                ground_mem_fraction_static=plan['ground_mem_fraction_static'],
                tracking_or_training_protocols_changed=False, contention_timing_declared=True)
            (out/'shared_gpu7_allocation.json').write_text(json.dumps(allocation, indent=2)+'\n')
            state.update(status='SHARED_GPU7_TRAINING_ALLOCATION_VERIFIED', allocation=allocation); save()
        else:
            registered = plan['upstream_owner_identity']
            if alive(registered):
                proc = Path('/proc')/str(registered[0])
                assert (proc/'cwd').resolve() == ROOT
                assert plan['upstream_owner_script'] in (proc/'cmdline').read_bytes().decode().split('\0')
                fd = os.pidfd_open(registered[0])
                try:
                    if alive(registered):
                        select.select([fd], [], [])
                finally:
                    os.close(fd)
            assert not alive(registered)
            upstream = ROOT/plan['upstream_owner']
            terminal = json.loads((upstream/'owner_state.json').read_text())
            assert terminal['status'] == plan['upstream_required_terminal_status']
            completed = terminal['completed']['adapted200']
            assert completed['complete_all9'] and not completed['missing_tasks']
            assert completed['full_native_row_count'] == plan['upstream_required_all9_rows']
            group = plan['upstream_adapted_service_group']
            assert completed['owned_service_group_stopped'] == group and not members(group)
            summary = json.loads((upstream/'adapted200_native_summary.json').read_text())
            assert summary['completed_all9'] and summary['native_summarizer_unchanged']
            release = dict(status='VERIFIED_COMPLETED_GPU_OWNER_RELEASE', physical_gpu=6,
                upstream_owner_identity=registered, owned_process_groups=[group], all_owned_groups_gone=True,
                original_native_summary=str(upstream/'adapted200_native_summary.json'),
                native_runner_exit_code_unknown=summary['actual_runner_exit_code_unknown_after_parent_exit'])
            (out/'upstream_gpu6_release.json').write_text(json.dumps(release, indent=2)+'\n')
            state.update(status='UPSTREAM_GPU6_ALL9_RELEASE_VERIFIED', verified_upstream_release=release); save()
        for port in (plan['ground_port'], plan['feature_port']):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('127.0.0.1', port))
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if free >= plan.get('launch_min_free_MiB', 45*1024):
                break
            time.sleep(20)
        state.update(status=f'STARTING_OWNED_GPU{gpu}_NATIVE_DLM_AND_ROI', launch_free_memory_MiB=free); save()
        for name, command in [('ground', ['tools/groundjev/serve_groundanything.py',
                '--port', str(plan['ground_port']), '--mem-fraction-static', str(plan.get('ground_mem_fraction_static', .4)), '--out', str(out/'ground_service')]),
                ('features', ['tools/groundjev/serve_native_features.py', '--port', str(plan['feature_port']),
                '--receipt', str(out/'feature_service.json')])]:
            log = (out/(name+'_service.log')).open('w'); service_logs.append(log)
            service = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', str(gpu), '--',
                str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), *command], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            services.append(service)
        state.update(status=f'WAITING_OWNED_GPU{gpu}_NATIVE_SERVICES', owned_service_groups=[s.pid for s in services]); save()
        ground_url = f'http://127.0.0.1:{plan["ground_port"]}/v1'
        feature_url = f'http://127.0.0.1:{plan["feature_port"]}'
        deadline = time.monotonic()+600
        while True:
            assert all(s.poll() is None for s in services), 'Owned native service exited before readiness'
            try:
                with urlopen(ground_url+'/models', timeout=5) as response:
                    ready = any(m['id'] == 'groundinganything' for m in json.load(response)['data'])
                if ready and (out/'feature_service.json').exists():
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Owned RMOT native services did not become ready')
            time.sleep(2)
        for item in validation['scopes']:
            name = item['name']; manifest_path = ROOT/item['manifest']
            manifest = json.loads(manifest_path.read_text())
            output = out/'clips'/name; output.parent.mkdir(exist_ok=True)
            used = sum(r['archive_bytes'] for r in state['completed'].values())
            remaining = plan['native_candidate_archive_budget_bytes']-used
            assert remaining > 0
            command = [str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py',
                '--task', 'rmot', '--video', str(manifest_path), '--backend', 'compact', '--compact-device', 'cpu',
                '--checkpoint', str(ROOT/plan['checkpoint']), '--grounding-url', ground_url,
                '--grounding-parser', plan['parser'], '--grounding-decoder-policy', plan['decoder'],
                '--perception-failure-policy', plan['perception_failure_policy'], '--native-feature-url', feature_url,
                '--record-native-candidates', '--native-candidate-format', 'packed_npz_f32',
                '--native-candidate-byte-budget', str(remaining), '--omit-replay-feature-vectors',
                '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0', '--out', str(output)]
            state.update(status='COLLECTING_'+name, current_rollout=name); save()
            with (out/(name+'.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                state['current_child_pid'] = child.pid; save()
                for line in child.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True)
                assert child.wait() == 0, f'{name} failed; preserve all original frames and valid causal prefix'
            state.pop('current_child_pid', None)
            runtime = json.loads((output/'result.json').read_text())
            assert runtime['frames'] == len(manifest['images']) == item['frames'] and runtime['all_input_frames_processed']
            assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
            assert runtime['decision_device'] == 'cpu' and runtime['resumed_completed_frames'] == 0
            assert runtime['native_candidate_archive_format'] == 'packed_npz_f32'
            assert runtime['native_candidate_archive_bytes'] <= remaining
            with (output/'frame_candidates.jsonl').open() as records:
                candidates = sum(len(json.loads(line)['candidates']) for line in records)
            state['completed'][name] = dict(run=str(output), manifest=str(manifest_path), video=item['video'],
                query_role=item['query_role'], frames=runtime['frames'], archive_bytes=runtime['native_candidate_archive_bytes'],
                native_candidates=candidates, rejected_frames=len(runtime['perception_rejected_frames']))
            save()
        assert len(state['completed']) == plan['expected_query_rollouts']
        assert sum(r['frames'] for r in state['completed'].values()) == plan['expected_query_frame_contexts']
        state.pop('current_rollout', None)
        state.update(status='ALL1814_FIT_PHYSICAL_FRAMES_18_NATIVE_QUERY_ROLLOUTS_COMPLETE',
            unique_physical_frames=plan['expected_physical_frames'],
            combined_registered_feature_upper_bytes=plan['combined_registered_feature_upper_bytes'])
        save()
    except Exception as error:
        state.update(status='FULL_FIT_RMOT_COLLECTION_FAILED', error=repr(error)); save()
        raise
    finally:
        for service in services:
            stop_owned_service(service)
        for log in service_logs:
            log.close()


if __name__ == '__main__':
    main()
