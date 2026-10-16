"""Fresh native DLM all-frame prompt control; GT is only an external box audit."""
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
    from PIL import Image
    from locatemot.groundjev.perception import GroundAnythingAdapter, GroundingProtocolError
    from tools.groundjev.export_rmot_full_fit_groups import context_for_manifest
    from tools.groundjev.service_process_group import stop_owned_service

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT' and plan['physical_gpu'] == 4
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    manifest = json.loads((ROOT/plan['manifest']).read_text())
    original = json.loads(Path(plan['original_expression_file']).read_text())
    canonical = json.loads(Path(plan['canonical_expression_file']).read_text())
    assert original['label'] == canonical['label'] and not original.get('ignore') and not canonical.get('ignore')
    assert manifest['expression'] == original['sentence']
    assert len(manifest['images']) == plan['frames'] == 154
    source = ROOT/plan['source_run']
    runtime = json.loads((source/'result.json').read_text())
    assert runtime['all_input_frames_processed'] and runtime['frames'] == 154
    assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
    baseline = [json.loads(l) for l in (source/'frame_candidates.jsonl').open()]
    assert [r['frame'] for r in baseline] == manifest['frame_numbers']
    out = ROOT/plan['output']; out.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        previous = json.loads((out/'owner_state.json').read_text())
        assert previous['plan'] == str(args.plan)
    state = dict(status='WAITING_SHARED_GPU4_MEMORY_FOR_FRESH_PROMPT_CONTROL',
        owner_pid=os.getpid(), owner_starttick=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        plan=str(args.plan), physical_gpu=4, completed={}, model_training_updates=0,
        existing_jobs_or_metrics_changed=False, full_goal_complete=False)

    def save():
        tmp = out/'owner_state.tmp'; tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(out/'owner_state.json')
        print(json.dumps(dict(status=state['status'], completed=list(state['completed']))), flush=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    server = None
    log = None
    save()
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', plan['port']))
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if free >= plan['launch_min_free_MiB']:
                break
            time.sleep(20)
        state.update(status='STARTING_FRESH_NATIVE_GROUND_ON_SHARED_GPU4', launch_free_memory_MiB=free); save()
        log = (out/'server.log').open('a' if args.resume else 'w')
        service_out = out/('service_resume_'+str(os.getpid()) if args.resume else 'service')
        server = subprocess.Popen(['python3', 'tools/groundjev/run_with_resources.py', '--gpus', '4', '--',
            str(ROOT/'external/GroundAnything/.venv-serve/bin/python'), 'tools/groundjev/serve_groundanything.py',
            '--model', str(ROOT/plan['checkpoint']), '--port', str(plan['port']),
            '--mem-fraction-static', str(plan['ground_mem_fraction_static']), '--out', str(service_out)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='WAITING_OWNED_NATIVE_PROMPT_CONTROL_SERVICE', owned_service_group=server.pid); save()
        url = f'http://127.0.0.1:{plan["port"]}/v1'
        deadline = time.monotonic()+600
        while True:
            assert server.poll() is None, 'Owned fresh prompt-control service exited before readiness'
            try:
                with urlopen(url+'/models', timeout=5) as response:
                    if any(m['id'] == 'groundinganything' for m in json.load(response)['data']):
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Owned prompt-control service readiness timeout')
            time.sleep(2)
        frontend = GroundAnythingAdapter(url, parser_policy=plan['parser'], decoder_policy=plan['decoder'])
        assert frontend.decode_parameters['sampling_seed'] == plan['sampling_seed'] == 42
        collected = {'recorded_original': baseline}
        for arm in plan['arms']:
            name = arm['name']; path = out/(name+'.jsonl')
            rows = [json.loads(l) for l in path.open()] if args.resume and path.exists() else []
            assert [r['frame'] for r in rows] == manifest['frame_numbers'][:len(rows)]
            assert all(r['arm'] == name and r['expression'] == arm['expression'] for r in rows)
            state.update(status='COLLECTING_'+name, restored_completed_frames=len(rows)); save()
            with path.open('a' if rows else 'w') as records:
                for frame, image_path in list(zip(manifest['frame_numbers'], manifest['images']))[len(rows):]:
                    with Image.open(image_path) as source_image:
                        image = source_image.convert('RGB')
                    try:
                        candidates, metadata = frontend(image, arm['task'], plan['vocabulary'],
                            expression=arm['expression'], image_path=image_path)
                    except GroundingProtocolError as error:
                        candidates = []
                        metadata = dict(error.metadata, model_response_rejected=True,
                            rejected_output_policy='empty_and_record; frame retained, no GT proposals')
                    row = dict(arm=name, task=arm['task'], expression=arm['expression'], frame=frame,
                        image_path=image_path, candidates=candidates, perception=metadata,
                        GT_or_future_frames_in_native_model_input=False)
                    records.write(json.dumps(row)+'\n'); records.flush(); rows.append(row)
                    if len(rows) % 25 == 0 or len(rows) == 154:
                        print(json.dumps(dict(arm=name, frames=len(rows), candidates=sum(len(r['candidates']) for r in rows))), flush=True)
            assert len(rows) == 154
            collected[name] = rows
            state['completed'][name] = dict(frames=len(rows), candidates=sum(len(r['candidates']) for r in rows))
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < plan['diagnostic_data_budget_bytes']
            save()

        # Audit only after native collection: no GT-driven proposal selection.
        context = context_for_manifest(manifest, source, plan['label_root'], 0)

        def iou(a, b):
            intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
            return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)

        reports = {}
        for name, rows in collected.items():
            total, hit, candidates, rejected, positive_frames, covered_frames = 0, 0, 0, 0, 0, 0
            per_frame = []
            for row in rows:
                frame = row['frame']; truth = [g for g in context['truth'][frame] if g['is_query']]
                best = [max((iou(g['box'], c['box']) for c in row['candidates']), default=0.) for g in truth]
                count = sum(x >= .5 for x in best)
                total += len(truth); hit += count; candidates += len(row['candidates'])
                rejected += bool(row['perception'].get('model_response_rejected'))
                positive_frames += bool(truth); covered_frames += count > 0
                per_frame.append(dict(frame=frame, query_GT_boxes=len(truth), native_candidates=len(row['candidates']),
                    query_GT_best_candidate_IoU=best, query_GT_boxes_covered_at_half=count))
            reports[name] = dict(frames=len(rows), query_GT_box_occurrences=total, covered_query_GT_boxes=hit,
                query_GT_proposal_recall_IoU_half=hit/total, native_candidates=candidates,
                rejected_frames=rejected, query_positive_frames=positive_frames,
                query_positive_frames_with_any_covered_target=covered_frames, per_frame=per_frame)
        assert len({r['query_GT_box_occurrences'] for r in reports.values()}) == 1
        assert reports['recorded_original']['covered_query_GT_boxes'] == 0
        fresh = collected['fresh_original']
        def geometry(rows):
            return [{k: c[k] for k in ('box', 'label', 'native_score', 'score_kind')} for c in rows]
        repeated = dict(frames=154,
            exact_box_label_score_frames=sum(geometry(a['candidates']) == geometry(b['candidates']) for a, b in zip(baseline, fresh)),
            exact_raw_wire_frames=sum(a['perception']['result']['raw_output'] == b['perception']['result']['raw_output']
                for a, b in zip(baseline, fresh)),
            resource_allocation_and_physical_GPU_differ=True)
        result = dict(status='ALL154_FIT_FRAME_FRESH_NATIVE_PROMPT_CONTROLS_COMPLETE', reports=reports,
            recorded_vs_fresh_original=repeated, identical_original_and_canonical_target_maps=True,
            every_registered_frame_retained=True, physical_gpu=4, no_GT_input_to_native_model=True,
            new_training_updates=0, new_ROI_or_image_archive=False, original_source_unchanged=True,
            matched_weights_parser_decoder_seed_all_fresh_arms=True,
            inference_proposal_protocols_not_promoted=True, tracking_metric_or_gain_claim=False,
            component_proposal_recall_not_tracker_accuracy=True, concurrent_timing_not_uncontended_FPS=True,
            full_goal_complete=False)
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < plan['diagnostic_data_budget_bytes']
        state.update(status=result['status'], reports={k: {n: v for n, v in s.items() if n != 'per_frame'} for k, s in reports.items()}); save()
        print(json.dumps(dict(status=result['status'], reports=state['reports'], recorded_vs_fresh_original=repeated), indent=2), flush=True)
    except Exception as error:
        state.update(status='NATIVE_RMOT_PROMPT_CONTROL_FAILED', error=repr(error)); save()
        raise
    finally:
        if server is not None:
            stop_owned_service(server)
        if log is not None:
            log.close()


if __name__ == '__main__':
    main()
