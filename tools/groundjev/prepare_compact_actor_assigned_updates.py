"""Explicit Compact-actor final-ID supervision with exact streamed Full media."""
import argparse
from collections import Counter, defaultdict
from copy import copy, deepcopy
import gzip
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.assigned_identity_update import build_assigned_update_request, PROTOCOL
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.decision import OneJevDecision
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL as SPLITS
    from tools.groundjev.run_tracking_recipe_sft import configure_tracking_recipe_sft
    from tools.groundjev.streamed_tracking_media import CANDIDATE_KEY, HISTORY_KEY, IO_PROTOCOL
    from qev.prompt import render_question

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['question_protocol'] == PROTOCOL and plan['media_protocol'] == IO_PROTOCOL
    assert plan['combined_registered_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    settings = json.loads((ROOT/plan['diagnosis_plan']).read_text())
    diagnosis, out = ROOT/plan['diagnosis'], ROOT/plan['output']
    validation = json.loads((diagnosis/'validation.json').read_text())
    assert validation['all1184_GT_matches_birth_anchors_eligibility_and_targets_recomputed']
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'COMPACT_ACTOR_FINAL_ID_COMPONENT_RUNNING', 'source_plan': str(args.plan),
        'new_training_steps': 0, 'new_GPU_jobs': 0, 'full_goal_complete': False}

    def save():
        (out/'owner_state.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(status), flush=True)

    save()
    try:
        torch.set_num_threads(settings['torch_threads']); torch.manual_seed(0)
        stateful = configure_tracking_recipe_sft()
        from train import common, mm, sft
        mm.set_root(ROOT)
        rows = [json.loads(line) for line in (ROOT/settings['source']).open()]
        references = [json.loads(line) for line in (diagnosis/'actor_frames.jsonl').open()]
        labels = {(r['sequence'], r['frame'], r['candidate']): r for r in
                  map(json.loads, (diagnosis/'posthoc_candidate_labels.jsonl').open())}
        assert len(rows) == len(references) == plan['expected_frames']
        assert len(labels) == plan['expected_candidates']
        pack = torch.load(ROOT/settings['existing_native_ROI_pack'], map_location='cpu', weights_only=True)
        features = defaultdict(list)
        for example in pack:
            if not example['derived_withholding']:
                features[example['source_sequence'], example['frame']].append(example)
        assert set(features) == {(r['sequence'], r['frame']) for r in rows}
        for row in rows:
            offset = 0
            for example in sorted(features[row['sequence'], row['frame']],
                                  key=lambda e: int(re.search(r'_batch(\d+)$', e['id'])[1])):
                vectors = example['candidates']; group = row['candidates'][offset:offset+len(vectors)]
                assert [c['box'] for c in group] == example['candidate_boxes']
                for candidate, vector in zip(group, vectors):
                    candidate['native_region_feature'] = vector.tolist()
                offset += len(group)
            assert offset == len(row['candidates'])
        sequences = {}
        for name, file in settings['sequence_protocols'].items():
            seq = next(s for s in json.loads((SPLITS/file).read_text())['sequences'] if s['sequence'] == name)
            sequences[name] = seq, {int(Path(f).stem): f for f in seq['train_frames']}
        actor = CompactDecision(ROOT/settings['checkpoint'], device='cpu',
            batch_candidates=settings['candidate_group_size'], shortlist=settings['shortlist_limit'],
            shortlist_policy=settings['shortlist_policy'])
        builder = OneJevDecision.__new__(OneJevDecision); builder.evidence_format = 'v3_crop_motion'
        counts, image_paths, processor_cases = Counter(), set(), {}
        previous, tracker, provenance, aliases, pool = None, None, {}, {}, []
        specification = {'task': 'mot', 'vocabulary': ['person'], 'expression': None}
        with gzip.open(out/'train.jsonl.gz', 'wt', encoding='utf-8') as training, (out/'all_candidate_registry.jsonl').open('w') as registry:
            for row, reference in zip(rows, references):
                name, frame, candidates = row['sequence'], row['frame'], row['candidates']
                seq, allowed = sequences[name]
                assert frame in allowed and Path(row['image_path']).resolve() == (Path(seq['image_dir'])/allowed[frame]).resolve()
                if name != previous:
                    previous, provenance, aliases = name, {}, {}
                    pool = list(range(1, 100001))
                    random.Random(list(sequences).index(name)*1009).shuffle(pool)
                    tracker = OnlineTracker(actor, lost_seconds=settings['lost_seconds'],
                        update_threshold=settings['update_threshold'], memory_policy=settings['memory_policy'])
                before = {tid: copy(t) for tid, t in tracker.tracks.items()}
                with Image.open(row['image_path']) as raw:
                    image = raw.convert('RGB')
                timestamp = frame/seq['fps']
                outputs, state = tracker.step(image, candidates, timestamp, specification)
                assert outputs == reference['outputs'] and state == reference['state']
                assert (name, frame) == (reference['sequence'], reference['frame'])
                events = {e['candidate']: e for e in state['events'] if 'candidate' in e}
                for batch, offset in enumerate(range(0, len(candidates), settings['candidate_group_size'])):
                    group = candidates[offset:offset+settings['candidate_group_size']]
                    ids = events[offset]['decision']['shortlisted_ids']
                    assert all(events[offset+i]['decision']['shortlisted_ids'] == ids for i in range(len(group)))
                    ranked = []
                    for tid in ids:
                        aliases.setdefault(tid, pool[len(aliases)])
                        track = copy(before[tid]); track.track_id = aliases[tid]; ranked.append(track)
                    base, native = builder.build_request(image, group, ranked, timestamp, specification)
                    selected = {i: aliases[e['track_id']] for i in range(len(group))
                                if (e := events[offset+i]).get('track_id') is not None and e['action'] != 'NEW'}
                    request = build_assigned_update_request(base.state, selected) if selected else None
                    media = [{'type': 'image', 'path': row['image_path'], 'w': image.width, 'h': image.height},
                        {'type': 'image', 'path': row['image_path'], 'w': native[1]['image'].width,
                         'h': native[1]['image'].height, CANDIDATE_KEY: [c['box'] for c in group]}]
                    if ranked:
                        history = [dict(id=aliases[tid], **deepcopy(provenance[tid])) for tid in ids]
                        assert all(r[key]['sequence'] == name and r[key]['frame'] <= frame
                                   for r in history for key in ('anchor', 'long', 'recent'))
                        media.append({'type': 'image', 'path': row['image_path'], 'w': native[2]['image'].width,
                            'h': native[2]['image'].height, HISTORY_KEY: history})
                    regenerated, videos, metas = mm.load_media(media)
                    assert not videos and not metas and len(regenerated) == len(native)
                    assert all(a.size == b['image'].size and a.tobytes() == b['image'].tobytes()
                               for a, b in zip(regenerated, native))
                    counts['native_media_groups_pixels_exact'] += 1
                    counts['native_sheets_pixels_exact'] += len(native)-1
                    counts['existing_assignments_in_questions'] += len(selected)
                    image_paths.add(row['image_path'])
                    for i in range(len(group)):
                        label = labels[name, frame, offset+i]
                        assert label['actual_actor_identity'] == events[offset+i].get('track_id')
                        registry.write(json.dumps(dict(label, selected_input_alias=selected.get(i),
                            registered_new_question=i in selected))+'\n')
                        counts['registered_candidates'] += 1
                        answer = label['potential_assigned_ID_target']
                        if answer is None:
                            continue
                        question = request.questions[f'assigned_update_{i}']
                        common.target_vector({answer: 1.}, render_question('q', question).labels)
                        sample = {'id': f'{name}_{frame}_batch{batch}_compact_assigned_update_{i}',
                            'split': 'train', 'source': name, 'task': 'assigned_identity_memory_update',
                            'state': request.state, 'question': question.model_dump(exclude_none=True),
                            'target': {answer: 1.}, 'media': media,
                            'supervision': {'tracking_task_bucket': 'mot', 'frame': frame,
                                'actor': 'fixed sharedCompact CPU actual causal history; not Full actor',
                                'actor_checkpoint': str(ROOT/settings['checkpoint']),
                                'question_protocol': PROTOCOL, 'GT_only_external_labels': True,
                                'target_rule': plan['target_rule'], 'actual_global_assignment_not_GT_correction': True}}
                        training.write(json.dumps(sample, ensure_ascii=False)+'\n')
                        counts['supervised:'+answer] += 1
                        if len(processor_cases) < 2 and len(ranked) not in processor_cases:
                            processor_cases[len(ranked)] = sample, [m['image'] for m in native]
                for i, event in events.items():
                    if event.get('track_id') is None:
                        continue
                    crop = {'path': row['image_path'], 'box': candidates[i]['box'], 'sequence': name, 'frame': frame}
                    tid = event['track_id']
                    if event['action'] == 'NEW':
                        provenance[tid] = {key: deepcopy(crop) for key in ('anchor', 'long', 'recent')}
                    else:
                        provenance[tid]['recent'] = deepcopy(crop)
                        if event['memory_action'] == 'UPDATE':
                            provenance[tid]['long'] = deepcopy(crop)
                counts['frames'] += 1
        assert counts['registered_candidates'] == plan['expected_candidates']
        assert counts['frames'] == plan['expected_frames']
        assert counts['existing_assignments_in_questions'] == plan['expected_existing_assignments']
        assert counts['supervised:yes'] == plan['expected_yes'] and counts['supervised:no'] == plan['expected_no']
        assert counts['existing_assignments_in_questions']-counts['supervised:yes']-counts['supervised:no'] == 6
        bundle = common.load_bundle(str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
            device='cpu', dtype='float32', train=True, multimodal=True, load_model=False)
        tensor_checks = []
        for history_size, (sample, native) in processor_cases.items():
            prompt = common.render_example(sample, bundle.template_for(sample), 0, seed=0, augment=True).prompt
            actual_loader = mm.load_media
            try:
                mm.load_media = lambda media: (native, [], [])
                eager, _ = mm.mm_inputs(bundle.processor, [prompt], [sample['media']], 'cpu')
            finally:
                mm.load_media = actual_loader
            lazy, _ = mm.mm_inputs(bundle.processor, [prompt], [sample['media']], 'cpu')
            assert eager.keys() == lazy.keys() and all(torch.equal(eager[k], lazy[k]) for k in eager)
            assert bundle.counter.length(prompt, sample['media']) == int(lazy['attention_mask'].sum())
            tensor_checks.append({'history_count': history_size, 'all_processor_tensors_exact': True,
                'tokens': int(lazy['attention_mask'].sum()), 'keys': list(lazy)})
        summary = dict(status='COMPACT_ACTOR_FINAL_ID_COMPONENT_PIXELS_AND_INPUTS_VALIDATED_NOT_TRAINED',
            counts=dict(counts), rows=plan['expected_rows'], yes=plan['expected_yes'], no=plan['expected_no'],
            all_actual_existing_assignments_in_input_including_six_unknown=True,
            actor_outputs_and_state_exact_to_completed_CPU_diagnosis=True,
            question_protocol=PROTOCOL, input_io_protocol=IO_PROTOCOL, processor_tensor_checks=tensor_checks,
            unique_physical_fit_frames=48, original_images=len(image_paths),
            native_ROI_pack_reused=True, new_PNG_or_feature_archives=0,
            source_actor='actual fixed sharedCompact CPU; explicitly cross-actor for Full',
            Full_inference_or_quality_gain_proven=False, new_optimizer_steps=0, new_GPU_jobs=0,
            combined_registered_upper_bytes=plan['combined_registered_upper_bytes'], full_goal_complete=False)
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        config = json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
        config.update(model=str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600'),
            train=str((out/'train.jsonl.gz').resolve()), device='cpu', plan_only=True,
            name='compact_actor_assigned_update_CPU_plan', runs_dir=str((out/'CPU_plan').resolve()), max_steps=0)
        path = out/'official_CPU_plan_config.json'; path.write_text(json.dumps(config, indent=2)+'\n')
        native_plan, native_epoch = sft.plan_only, sft.plan_epoch

        def capture(bundle, data, settings, world):
            started = time.monotonic(); cached = native_epoch(bundle, data, settings, 0, 0, 1)
            elapsed = time.monotonic()-started
            batches, dropped, lengths = cached
            assert len(data) == plan['expected_rows'] and dropped == 0 and len(lengths) == len(data)
            assert Counter(next(iter(row['target'])) for row in data) == {'yes': 377, 'no': 345}
            assert stateful.protocol(settings)['input_io_protocol'] == IO_PROTOCOL
            os.environ['PLAN_WORLD'] = str(world)
            sft.plan_epoch = lambda *unused, **kwargs: cached
            try:
                code = native_plan(bundle, data, settings, world)
            finally:
                sft.plan_epoch = native_epoch
            assert code == 0 and not torch.cuda.is_initialized()
            summary.update(status='COMPACT_ACTOR_FINAL_ID_INPUTS_PIXELS_AND_FULL_CPU_PLAN_VALIDATED_NOT_TRAINED',
                overlength_dropped_rows=dropped, max_actual_tokens=max(lengths), total_actual_tokens=sum(lengths),
                actual_full_native_plan_seconds=elapsed, actual_CPU_world_size=world,
                hypothetical_4row_epoch_updates=len(sft.plan_steps(batches, lengths, 1, settings.accum_rows)))
            (out/'validated_component_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            return code

        sft.plan_only = capture
        try:
            assert sft.main(['--config', str(path), '--plan-only']) == 0
        finally:
            sft.plan_only = native_plan
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        assert size < plan['new_export_budget_bytes']
        status.update(status=summary['status'], counts=dict(counts), real_component_bytes=size)
        save()
    except Exception as error:
        status.update(status='COMPACT_ACTOR_FINAL_ID_COMPONENT_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
