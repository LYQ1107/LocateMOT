"""Compare shared training references with independent causal native requests."""
from copy import copy
import gzip
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.assigned_identity_update import build_assigned_update_request
    from locatemot.groundjev.decision import OneJevDecision
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.check_assigned_identity_runtime import RecordedDecision
    from tools.groundjev.group_reference_rows import REFERENCE_KEY, install
    from tools.groundjev.run_tracking_recipe_sft import configure_tracking_recipe_sft
    from tools.groundjev.run_video import ReplayFrames
    from train import common, mm

    torch.set_num_threads(4)
    configure_tracking_recipe_sft()
    original_load = install(ROOT)
    mm.set_root(ROOT)
    output = ROOT/'outputs/groundjev/training_data/broader_native_MOT1056_full_v3_groups_v1'
    source = ROOT/'outputs/groundjev/training_data/broader_native_MOT1056_gpu7_v3_complete_manifest'
    clip = 'MOT17-02-FRCNN'
    legacy_rows = 0
    for relative in ('native_ground_own_history_v1/train.jsonl',
                     'base175_streamed_component_v1/train.jsonl.gz',
                     'compact_actor_assigned_update_component_v1/train.jsonl.gz'):
        path = ROOT/'outputs/groundjev/training_data'/relative
        a, b = original_load(path, limit=2), common.load_rows(path, limit=2)
        assert a == b and len(a) == 2
        legacy_rows += len(a)
    rows = common.load_rows(output/'clips'/clip/'train_refs.jsonl.gz')
    cases = [next(r for r in rows if r['task'] == family and r['state']['historical_tracks'])
             for family in ('identity', 'reliability', 'assigned_identity_memory_update')]
    manifest = json.loads((source/'manifests'/(clip+'.json')).read_text())
    runtime = json.loads((source/'clips'/clip/'result.json').read_text())
    traces = [json.loads(s) for s in (source/'clips'/clip/'frame_decisions.jsonl').open()]
    builder = OneJevDecision.__new__(OneJevDecision)
    builder.evidence_format = 'v3_crop_motion'
    specification = {'task': 'mot', 'vocabulary': ['person'], 'expression': None}
    bundle = common.load_bundle(str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
                                device='cpu', train=True, multimodal=True, load_model=False)
    reports = []
    for row in cases:
        reference = row[REFERENCE_KEY]
        _, number, batch = re.fullmatch(r'(.+)_(\d+)_batch(\d+)',
                                       reference['group'].removesuffix('_assigned')).groups()
        number, batch = int(number), int(batch)
        saved = RecordedDecision()
        tracker = OnlineTracker(saved)
        # Each independent case starts at the source beginning: ReplayFrames is forward-only.
        replay = ReplayFrames(Path(runtime['native_candidate_archive']))
        prefix = 0
        try:
            for trace in traces:
                candidates = replay.frame(trace['frame'])['candidates']
                with Image.open(trace['image_path']) as raw:
                    image = raw.convert('RGB')
                timestamp = trace['frame']/manifest['fps']
                if trace['frame'] == number:
                    break
                saved.row = trace
                assert tracker.step(image, candidates, timestamp, specification) == (trace['outputs'], trace['state'])
                prefix += 1
            assert trace['frame'] == number
            events = {e['candidate']: e for e in trace['state']['events'] if 'candidate' in e}
            offset = batch*16
            actual_ids = events[offset]['decision']['shortlisted_ids']
            alias_ids = [t['id'] for t in row['state']['historical_tracks']]
            assert len(actual_ids) == len(alias_ids)
            aliases = dict(zip(actual_ids, alias_ids))
            ranked = []
            for actual, alias in aliases.items():
                track = copy(tracker.tracks[actual]); track.track_id = alias; ranked.append(track)
            group = candidates[offset:offset+len(row['state']['candidates'])]
            request, native_media = builder.build_request(image, group, ranked, timestamp, specification)
            if row['task'] == 'assigned_identity_memory_update':
                selected = {local: aliases[e['track_id']] for local in range(len(group))
                            if (e := events[offset+local]).get('track_id') is not None and e['action'] != 'NEW'}
                request = build_assigned_update_request(request.state, selected)
            assert request.state == row['state']
            assert request.questions[reference['question']].model_dump(exclude_none=True) == row['question']
            native_images = [m['image'] for m in native_media]
            actual_images, videos, metadata = mm.load_media(row['media'])
            assert not videos and not metadata and len(actual_images) == len(native_images)
            assert all(a.size == b.size and a.tobytes() == b.tobytes() for a, b in zip(actual_images, native_images))
            expanded = {k: v for k, v in row.items() if k != REFERENCE_KEY}
            prompt = common.render_example(row, bundle.template_for(row), 0, seed=0, augment=True).prompt
            assert prompt == common.render_example(expanded, bundle.template_for(expanded), 0, seed=0, augment=True).prompt
            actual_loader = mm.load_media
            mm.load_media = lambda media: (native_images, [], [])
            try:
                eager, eager_ends = mm.mm_inputs(bundle.processor, [prompt], [row['media']], 'cpu')
            finally:
                mm.load_media = actual_loader
            lazy, lazy_ends = mm.mm_inputs(bundle.processor, [prompt], [row['media']], 'cpu')
            assert eager_ends == lazy_ends and set(eager) == set(lazy)
            assert all(torch.equal(eager[k], lazy[k]) for k in eager)
            assert bundle.counter.length(prompt, row['media']) == int(lazy['attention_mask'].sum())
            reports.append({'row': row['id'], 'family': row['task'], 'causal_prefix_frames_exact': prefix,
                            'native_state_question_pixels_and_processor_tensors_exact': True,
                            'tokens': int(lazy['attention_mask'].sum()), 'processor_keys': sorted(lazy)})
        finally:
            replay.close()
    assert not torch.cuda.is_initialized()
    report = {'status': 'NATIVE_CAUSAL_REQUEST_AND_REAL_PROCESSOR_GROUP_REFERENCES_VALIDATED',
              'legacy_rows_unchanged': legacy_rows, 'cases': reports,
              'new_model_predictions': 0, 'new_optimizer_updates': 0, 'full_goal_complete': False}
    (output/'independent_native_input_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
