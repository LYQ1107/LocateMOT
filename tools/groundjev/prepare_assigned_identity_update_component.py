"""Prepare explicit final-ID write supervision from an existing uncorrected actor."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.assigned_identity_update import PROTOCOL as QUESTION_PROTOCOL, build_assigned_update_request
    from locatemot.groundjev.decision import overlap
    from tools.groundjev.prepare_tracking_decisions import PROTOCOL, annotations
    from qev.prompt import render_question
    from train import common

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['combined_registered_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    source, out = ROOT/plan['source'], ROOT/plan['output']
    audit = json.loads((ROOT/plan['source_audit']).read_text())
    assert audit['status'] == 'FULL_NATIVE_FIT_OBSERVATION_Q3_VERSUS_ASSIGNED_IDENTITY_AUDITED'
    assert audit['actual_existing_identity_assignments'] == 658
    out.mkdir(parents=True, exist_ok=False)
    updates, groups = {}, defaultdict(dict)
    for row in map(json.loads, (source/'train.jsonl').open()):
        if row['task'] != 'memory_update':
            continue
        match = re.fullmatch(r'(.+)_(\d+)_batch(\d+)_update_(\d+)', row['id'])
        name, frame, batch, local = match.groups()
        key = name, int(frame), int(batch), int(local)
        assert key not in updates
        updates[key] = row; groups[key[:2]][key[2]] = row['state']['candidates']
    protocols = {}
    for file, name in [('mot17_half.json', 'MOT17-02-FRCNN'), ('dancetrack.json', 'dancetrack0001'), ('mot20_half.json', 'MOT20-01')]:
        protocol = json.loads((PROTOCOL/file).read_text())
        seq = next(s for s in protocol['sequences'] if s['sequence'] == name)
        allowed = {int(Path(f).stem) for f in seq['train_frames']}
        protocols[name] = allowed, annotations(seq, allowed)
    counts, seen, media_paths = Counter(), set(), set()
    previous, anchors, aliases = None, {}, {}
    with gzip.open(out/'train.jsonl.gz', 'wt', encoding='utf-8') as training, (out/'all_candidate_registry.jsonl').open('w') as registry:
        for row in map(json.loads, (source/'native_rollout.jsonl').open()):
            name, frame = row['sequence'], row['frame']
            allowed, truth = protocols[name]; assert frame in allowed
            if previous != name:
                previous, anchors, aliases = name, {}, {}
            before = dict(anchors)
            gt, candidates = truth[frame], row['candidates']
            matched = {}
            if gt and candidates:
                scores = np.asarray([[overlap(c['box'], g['box']) for g in gt] for c in candidates])
                rr, cc = linear_sum_assignment(-scores)
                matched = {int(i): gt[int(j)] for i, j in zip(rr, cc) if scores[i, j] >= .5}
            events = {e['candidate']: e for e in row['state']['events'] if 'candidate' in e}
            offset = 0
            for batch, group in sorted(groups[name, frame].items()):
                assert [c['box'] for c in group] == [c['box'] for c in candidates[offset:offset+len(group)]]
                old = updates[name, frame, batch, 0]
                state = old['state']
                actor_ranked = events[offset]['decision']['shortlisted_ids']
                shadow_ranked = [t['id'] for t in state['historical_tracks']]
                assert len(actor_ranked) == len(shadow_ranked)
                for actor_id, shadow_id in zip(actor_ranked, shadow_ranked):
                    assert aliases.setdefault(actor_id, shadow_id) == shadow_id
                assert state['geometry_columns_track_ids'] == shadow_ranked
                selected = {local: aliases[e['track_id']] for local in range(len(group))
                            if (e := events[offset+local]).get('track_id') is not None and e['action'] != 'NEW'}
                request = build_assigned_update_request(state, selected) if selected else None
                if request:
                    assert {k: v for k, v in request.state.items() if k != 'final_assignment'} == state
                    assert len(request.questions) == len(selected)
                for local in range(len(group)):
                    key = name, frame, batch, local
                    assert key not in seen; seen.add(key)
                    original, event, target = updates[key], events[offset+local], matched.get(offset+local)
                    assert original['state'] == state and original['media'] == old['media']
                    gid = target['gt_id'] if target else None
                    assigned = event.get('track_id'); answer = None
                    if assigned is None:
                        category = 'no_final_identity'
                    elif event['action'] == 'NEW':
                        category = 'NEW_initialization'
                    elif gid is None:
                        category = 'unmatched_not_universal_negative_truth'
                    elif before.get(assigned) is None:
                        category = 'unknown_birth_anchor'
                    elif before[assigned] == gid:
                        category = 'known_consistent_existing_identity'
                        answer = 'yes' if target['visibility'] >= plan['visibility_threshold_from_existing_observation_proxy'] else 'no'
                    else:
                        category = 'known_conflicting_existing_identity'; answer = 'no'
                    counts[category] += 1
                    record = {'sequence': name, 'frame': frame, 'candidate': offset+local,
                        'original_training_row': original['id'], 'final_action': event['action'],
                        'final_actor_identity': assigned, 'selected_input_alias': selected.get(local),
                        'posthoc_candidate_GT_identity': gid, 'posthoc_immutable_birth_GT_identity': before.get(assigned),
                        'eligibility': category, 'new_supervised_target': answer,
                        'original_candidate_Q3_target': original['target'],
                        'registered_new_question': local in selected, 'GT_fields_outside_model_input': True}
                    registry.write(json.dumps(record)+'\n')
                    if answer is None:
                        counts['registered_without_new_target'] += 1
                        continue
                    question = request.questions[f'assigned_update_{local}']
                    common.target_vector({answer: 1.}, render_question('q', question).labels)
                    sample = {'id': original['id'].replace('_update_', '_assigned_update_'),
                        'split': 'train', 'source': name, 'task': 'assigned_identity_memory_update',
                        'state': request.state, 'question': question.model_dump(exclude_none=True),
                        'target': {answer: 1.}, 'media': original['media'],
                        'supervision': dict(original['supervision'], tracking_task_bucket='mot',
                            question_protocol=QUESTION_PROTOCOL, original_row=original['id'],
                            target_rule='known immutable birth ID equality and visibility>=.6',
                            final_assignment_is_actual_model_global_matching=True,
                            unknown_anchor_and_unmatched_not_negative=True)}
                    training.write(json.dumps(sample, ensure_ascii=False)+'\n')
                    counts['supervised:'+answer] += 1
                    counts['original_Q3_yes_new_no'] += answer == 'no' and original['target'] == {'yes': 1.}
                    media_paths.update(m['path'] for m in original['media'])
                offset += len(group)
            assert offset == len(candidates)
            for i, event in events.items():
                if event['action'] == 'NEW':
                    anchors[event['track_id']] = matched[i]['gt_id'] if i in matched else None
            counts['frames'] += 1
    assert seen == updates.keys() and len(seen) == 1184 and counts['frames'] == 48
    for category in ('known_consistent_existing_identity', 'known_conflicting_existing_identity',
                     'unknown_birth_anchor', 'unmatched_not_universal_negative_truth', 'NEW_initialization', 'no_final_identity'):
        assert counts[category] == audit['counts'][category]
    assert counts['known_consistent_existing_identity'] == plan['known_consistent_existing_assignments']
    assert counts['known_conflicting_existing_identity'] == plan['known_conflicting_existing_assignments']
    assert counts['supervised:yes'] == 29 and counts['supervised:no'] == 584
    assert counts['registered_without_new_target'] == 571
    assert all((ROOT/path).is_file() for path in media_paths)
    assert not torch.cuda.is_initialized()
    summary = {'status': 'FINAL_ASSIGNED_ID_UPDATE_COMPONENT_GENERATED_NOT_TRAINED_OR_ENABLED',
        'question_protocol': QUESTION_PROTOCOL, 'counts': dict(counts), 'supervised_rows': 613,
        'all_original_candidates_registered': len(seen), 'actual_existing_assignments': 658,
        'unknown_or_unmatched_existing_assignments_registered_without_target': 45,
        'all_actual_existing_assignments_in_input_including_unknown': True,
        'alias_mapping_from_actual_shortlist_to_saved_request_history_verified': True,
        'input_state_only_adds_model_final_assignment_not_GT_labels': True,
        'old_observation_questions_and_targets_unchanged': True,
        'existing_media_paths_reused': len(media_paths), 'new_image_or_feature_artifacts': 0,
        'source': str(source), 'GT_access': 'registered MOT train frames only; no actor correction or eval GT',
        'current_tracker_or_running_training_changed': False, 'new_GPU_or_optimizer_loaded': False,
        'class_imbalance_requires_balanced_curriculum': True,
        'needs_future_same_builder_post_global_matching_before_any_memory_write_runtime': True,
        'not_tracking_accuracy_or_runtime_training_equivalence_proof': True,
        'combined_registered_upper_bytes': plan['combined_registered_upper_bytes'], 'full_goal_complete': False}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < plan['new_export_budget_bytes']
    print(json.dumps(summary), flush=True)

    # Plan the entire new question component using the unchanged native trainer.
    from tools.groundjev.run_streamed_media_sft import configure_streamed_sft
    configure_streamed_sft()
    from train import sft
    config = json.loads((ROOT/'configs/groundjev/onejev_unified600_native_v2.json').read_text())
    config.update(train=str((out/'train.jsonl.gz').resolve()), device='cpu', plan_only=True,
        name='assigned_update_CPU_plan', runs_dir=str((out/'CPU_plan').resolve()), max_steps=0)
    config_path = out/'official_CPU_plan_config.json'; config_path.write_text(json.dumps(config, indent=2)+'\n')
    native_epoch, native_plan = sft.plan_epoch, sft.plan_only
    def capture(bundle, rows, settings, world):
        start = time.monotonic(); cached = native_epoch(bundle, rows, settings, 0, 0, 1)
        elapsed = time.monotonic()-start
        batches, dropped, lengths = cached
        assert len(rows) == 613 and dropped == 0 and len(lengths) == len(rows)
        assert Counter(next(iter(r['target'])) for r in rows) == {'yes': 29, 'no': 584}
        os.environ['PLAN_WORLD'] = str(world)
        sft.plan_epoch = lambda *unused, **kwargs: cached
        try:
            code = native_plan(bundle, rows, settings, world)
        finally:
            sft.plan_epoch = native_epoch
        assert code == 0 and not torch.cuda.is_initialized()
        summary.update(status='FINAL_ASSIGNED_ID_UPDATE_INPUTS_AND_FULL_CPU_PLAN_VALIDATED_NOT_TRAINED_OR_ENABLED',
            actual_CPU_world_size=world, overlength_dropped_rows=dropped,
            max_actual_tokens=max(lengths), total_actual_tokens=sum(lengths),
            actual_full_native_plan_epoch_wall_seconds=elapsed, actual_new_training_updates=0,
            hypothetical_4row_epoch_updates=len(sft.plan_steps(batches, lengths, 1, settings.accum_rows)),
            all_component_file_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
        assert summary['all_component_file_bytes'] < plan['new_export_budget_bytes']
        (out/'validated_component_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)
        return code
    sft.plan_only = capture
    try:
        assert sft.main(['--config', str(config_path), '--plan-only']) == 0
    finally:
        sft.plan_only = native_plan


if __name__ == '__main__':
    main()
