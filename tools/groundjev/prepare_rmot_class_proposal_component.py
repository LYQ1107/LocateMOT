"""Keep the original referring predicate while enriching native proposal evidence."""
import argparse
from copy import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from locatemot.groundjev.compact import CompactDecision
    from tools.groundjev.export_broader_native_full_groups import export_clip
    from tools.groundjev.export_rmot_full_fit_groups import context_for_manifest
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    from tools.groundjev.service_process_group import stop_owned_service

    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT' and plan['physical_gpu'] == 4
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    validation = json.loads((ROOT/plan['source_prompt_control']).read_text())
    assert validation['status'] == 'ALL_THREE_FRESH_NATIVE_ARMS_EXIT0_RAW_PREDICTIONS_AND_RECALL_DENOMINATORS_VALIDATED'
    source = ROOT/plan['candidate_source']
    records = [json.loads(l) for l in source.open()]
    manifest = json.loads((ROOT/plan['manifest']).read_text())
    assert [r['frame'] for r in records] == manifest['frame_numbers']
    assert len(records) == plan['expected_frames'] == 154
    assert sum(len(r['candidates']) for r in records) == plan['expected_native_candidates'] == 2182
    assert all(r['arm'] == 'provided_class_proposals' and r['task'] == 'mot' for r in records)
    assert all(not r['GT_or_future_frames_in_native_model_input'] for r in records)
    manifest = dict(manifest, scope='fit_provided_class_proposals_original_query_component_not_benchmark',
        proposal_source=str(source), original_expression_preserved=True,
        proposal_conditioning='provided person/car classes; original predicate remains actor and JEV input')
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    manifest_path = out/'manifest.json'; manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    state = dict(status='VALIDATED_NATIVE_CLASS_PROPOSALS_KEEPING_ORIGINAL_QUERY', owner_pid=os.getpid(),
        owner_starttick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        physical_gpu_for_native_ROI_only=4, actual_new_DLM_requests=0, actual_new_training_updates=0,
        original_referring_expression=manifest['expression'], full_goal_complete=False)

    def save():
        tmp = out/'owner_state.tmp'; tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(out/'owner_state.json')
        print(json.dumps({k: v for k, v in state.items() if k not in ('command', 'export_summary')}), flush=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', PYTHONPATH=str(ROOT/'external/OneJev'), PYTHONUNBUFFERED='1')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    feature = None; log = None
    save()
    try:
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if free >= plan['feature_min_free_MiB']:
                break
            time.sleep(20)
        # HTTPServer owns and checks its fresh port. Never stop another service.
        log = (out/'native_feature_service.log').open('w')
        feature = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '4', '--',
            str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), 'tools/groundjev/serve_native_features.py',
            '--port', str(plan['feature_port']), '--receipt', str(out/'native_feature_service.json')],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='WAITING_OWNED_NATIVE_ROI_ONLY_SERVICE', launch_free_memory_MiB=free,
            owned_feature_service_group=feature.pid); save()
        deadline = time.monotonic()+300
        while not (out/'native_feature_service.json').exists():
            assert feature.poll() is None, 'Owned native feature service exited before readiness'
            if time.monotonic() > deadline:
                raise RuntimeError('Owned native feature service did not become ready')
            time.sleep(1)
        original = out/'source'
        command = [sys.executable, 'tools/groundjev/run_video.py', '--task', 'rmot', '--video', str(manifest_path),
            '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint', str(ROOT/plan['checkpoint']),
            '--candidate-replay', str(source), '--native-feature-url', f'http://127.0.0.1:{plan["feature_port"]}',
            '--record-native-candidates', '--native-candidate-format', 'packed_npz_f32',
            '--native-candidate-byte-budget', str(plan['native_candidate_archive_budget_bytes']),
            '--omit-replay-feature-vectors', '--memory-checkpoint-format', 'binary', '--memory-export-interval', '0',
            '--out', str(original)]
        state.update(status='COLLECTING_REAL_NATIVE_ROI_AND_UNCORRECTED_ORIGINAL_QUERY_ACTOR', command=command); save()
        with (out/'source.log').open('w') as source_log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            state['source_child_pid'] = child.pid; save()
            for line in child.stdout:
                source_log.write(line); source_log.flush(); print(line, end='', flush=True)
            assert child.wait() == 0, 'Preserve valid native prefix and identify the first source failure'
        state.pop('source_child_pid', None)
        runtime = json.loads((original/'result.json').read_text())
        assert runtime['frames'] == 154 and runtime['all_input_frames_processed']
        assert runtime['specification']['expression'] == manifest['expression']
        assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert len(runtime['perception_rejected_frames']) == plan['expected_rejected_frames'] == 6
        assert runtime['native_candidate_archive_bytes'] <= plan['native_candidate_archive_budget_bytes']
        stop_owned_service(feature); feature = None; log.close(); log = None
        state.update(status='SOURCE_COMPLETE_NATIVE_ROI_SERVICE_STOPPED_CPU_EXPORT',
            source_frames=154, archive_bytes=runtime['native_candidate_archive_bytes']); save()
        torch.set_num_threads(4); torch.manual_seed(0)
        stateful = configure_group_reference_sft()

        class Recorder(CompactDecision):
            def accepted_batch(self, group, ranked, inputs):
                self.captured.append((group, [copy(t) for t in ranked]))

        actor = Recorder(ROOT/plan['checkpoint'], device='cpu')
        context = context_for_manifest(manifest, original, plan['label_root'], 0)
        summary = export_clip('0000_class_proposals_original_query', out, out/'groups', plan,
            actor, stateful, context=context)
        assert summary['dropped_rows'] == 0 and summary['counts']['frames'] == 154
        assert summary['counts']['candidates'] == 2182 and summary['counts']['rejected_frames'] == 6
        assert summary['counts']['query_membership:yes'] > 0
        group_bytes = sum(p.stat().st_size for p in (out/'groups').rglob('*') if p.is_file())
        assert group_bytes < plan['group_export_budget_bytes']
        assert not torch.cuda.is_initialized()
        state.update(status='NATIVE_CLASS_PROPOSALS_ORIGINAL_QUERY_154_COMPONENT_ALL_INPUTS_AND_CPU_ROWS_VALIDATED',
            export_summary=summary, group_bytes=group_bytes, source_original_query_preserved=True,
            original_referring_source_or_current6000_modified=False, new_Physical_frames=0)
        save()
    except Exception as error:
        state.update(status='NATIVE_CLASS_PROPOSALS_ORIGINAL_QUERY_COMPONENT_FAILED', error=repr(error)); save()
        raise
    finally:
        if feature is not None:
            stop_owned_service(feature)
        if log is not None:
            log.close()


if __name__ == '__main__':
    main()
