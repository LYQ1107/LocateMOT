"""All legal fit expressions on frozen native frames, reusing exact existing media."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    from PIL import Image, ImageChops
    from locatemot.groundjev.decision import OneJevDecision, overlap
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_video

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--annotation-contract', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['project_root'] == '/data2/user/LocateMOT'
    assert plan['combined_registered_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    out = args.out or ROOT/plan['output']; out.mkdir(parents=True, exist_ok=False)
    native = {(r['sequence'], r['frame']): r for r in map(json.loads, (ROOT/plan['native_source']).open())}
    clips = json.loads((ROOT/plan['native_plan']).read_text())['clips']
    by_video = {c['video']: c for c in clips}
    assert set(by_video) == set(LEGAL_SPLITS['refer_kitti_v2']['fit'])
    reference = list(map(json.loads, (ROOT/plan['reference_rows']).open()))
    templates = defaultdict(list)
    for row in reference:
        if row['supervision']['query_index'] == 0:
            video = row['source'].removeprefix('refer_kitti_')
            templates[video, row['supervision']['frame']].append(row)
    assert templates.keys() == native.keys() and len(native) == plan['physical_frames']
    assert sum(len(r['candidates']) for r in native.values()) == plan['native_candidates']

    # Verify the reused model inputs against the actual native request builder,
    # including every saved crop's pixels. No new image artifact is produced.
    builder = OneJevDecision.__new__(OneJevDecision)
    builder.evidence_format, builder.rmot_membership = 'v3_crop_motion', True
    gt, reused_media, pixel_comparisons = {}, set(), 0
    for (video, frame), row in native.items():
        assert_legal_video(video, dataset='refer_kitti_v2')
        assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
        with Image.open(row['image_path']) as raw:
            image = raw.convert('RGB')
        groups = [row['candidates'][i:i+16] for i in range(0, len(row['candidates']), 16)] or [[]]
        base = templates[video, frame]
        group_templates = defaultdict(list)
        for item in base:
            batch = int(item['id'].split('_batch')[1].split('_')[0])
            group_templates[batch].append(item)
        assert len(group_templates) == len(groups)
        for batch, group in enumerate(groups):
            rows = group_templates[batch]
            request, media = builder.build_request(image, group, [], frame/by_video[video]['fps'], rows[0]['state']['task'])
            expected_keys = [f'membership_{i}' for i in range(len(group))] or ['empty']
            assert len(rows) == len(expected_keys)
            for item, key in zip(rows, expected_keys):
                assert item['state'] == request.state
                assert item['question'] == request.questions[key].model_dump(exclude_none=True)
                assert item['media'] == rows[0]['media']
            assert len(media) == len(rows[0]['media'])
            for actual, saved in zip(media, rows[0]['media']):
                path = ROOT/saved['path']; reused_media.add(str(path))
                with Image.open(path) as raw:
                    pixels = raw.convert('RGB')
                assert actual['image'].size == pixels.size == (saved['w'], saved['h'])
                assert ImageChops.difference(actual['image'], pixels).getbbox() is None
                pixel_comparisons += 1
        labels = {}
        path = Path(plan['label_root'])/video/f'{frame:06d}.txt'
        if path.exists():
            for line in path.read_text().splitlines():
                f = line.split(); x, y, w, h = map(float, f[2:6])
                labels[int(f[1])] = [x*image.width, y*image.height, (x+w)*image.width, (y+h)*image.height]
        gt[video, frame] = labels

    # A release expression can mention an ID without a box on this frame.
    # Audit the whole registered component before emitting any training row;
    # do not silently intersect IDs or turn unresolved targets into negatives.
    queries, missing = [], []
    for indexed in map(json.loads, (ROOT/plan['query_index']).open()):
        video = assert_legal_video(indexed['video'], dataset='refer_kitti_v2')
        assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
        query = json.loads(Path(indexed['expression_file']).read_text())
        assert not query.get('ignore') and query['sentence'] == indexed['sentence']
        queries.append((indexed, query))
        for frame in by_video[video]['frames']:
            absent = set(query['label'].get(str(frame), [])) - gt[video, frame].keys()
            if absent:
                missing.append({'video': video, 'frame': frame,
                    'expression_file': indexed['expression_file'], 'missing_ids': sorted(absent)})
    if missing and args.annotation_contract is None:
        unique = {(r['video'], r['frame'], identity) for r in missing for identity in r['missing_ids']}
        report = {'status': 'RMOT_QUERY_COMPONENT_HELD_MISSING_GT_TARGET_BOXES',
            'audited_expression_files': len(queries), 'audited_query_frame_contexts': plan['query_frame_contexts'],
            'missing_query_frame_contexts': len(missing),
            'affected_expression_files': len({r['expression_file'] for r in missing}),
            'unique_missing_video_frame_identity_refs': len(unique), 'unique_missing': sorted(unique),
            'affected_contexts': missing, 'no_targets_intersected_or_relabeled': True,
            'actual_native_vs_existing_media_pixel_comparisons': pixel_comparisons,
            'training_rows_written': 0, 'new_model_calls': False, 'new_GPU_training': False,
            'first_actionable_root_cause': 'release expression target IDs have no same-frame released box; determine visibility/annotation contract before supervision',
            'full_goal_complete': False}
        (out/'failure_audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'affected_contexts'}, indent=2), flush=True)
        raise SystemExit(2)
    if missing:
        contract = json.loads(args.annotation_contract.read_text())
        assert contract['status'] == 'PINNED_OFFICIAL_RMOT_RELEASE_BOX_PROJECTION_VALIDATED'
        assert contract['all_affected_contexts_validated'] == len(missing)
        unique = {(r['video'], r['frame'], identity) for r in missing for identity in r['missing_ids']}
        assert {tuple(r) for r in contract['unique_missing']} == unique
        (out/'unboxed_expression_targets.json').write_text(json.dumps({
            'status': 'ALL_CONTEXTS_RETAINED_USING_PINNED_OFFICIAL_RELEASED_BOX_PROJECTION',
            'contract_validation': str(args.annotation_contract), 'official_commit': contract['official_commit'],
            'rule': contract['rule'], 'affected_contexts': missing,
            'original_expression_ID_lists_preserved_in_source': True,
            'no_GT_box_interpolation_or_sample_exclusion': True}, indent=2)+'\n')

    # Legacy q0/q1 samples must still have their exact targets and native input
    # state. This checks the release's top-left x/y/width/height interpretation.
    legacy = {}
    compared_media = set()
    for item in reference:
        key = (item['source'].removeprefix('refer_kitti_'), item['supervision']['frame'],
               item['state']['task']['expression'], item['id'].split('_batch')[1])
        assert key not in legacy
        legacy[key] = item
        # Alternate expressions use the same visual evidence. Check the actual
        # legacy q1 files too, rather than merely comparing declared sizes.
        canonical = next(r for r in templates[key[:2]] if r['id'].split('_batch')[1] == key[3])
        for old, new in zip(item['media'], canonical['media']):
            pair = old['path'], new['path']
            if pair in compared_media:
                continue
            with Image.open(ROOT/old['path']) as a, Image.open(ROOT/new['path']) as b:
                a, b = a.convert('RGB'), b.convert('RGB')
                assert a.size == b.size and ImageChops.difference(a, b).getbbox() is None
            compared_media.add(pair)
    counts, per_video, coverage, matched_legacy = Counter(), defaultdict(Counter), [], set()
    with gzip.open(out/'train.jsonl.gz', 'wt', encoding='utf-8') as output:
        for query_index, (indexed, query) in enumerate(queries):
            video = indexed['video']
            current = Counter()
            for frame in by_video[video]['frames']:
                ids = set(query['label'].get(str(frame), []))
                unboxed = sorted(ids - gt[video, frame].keys())
                boxes = [box for identity, box in gt[video, frame].items() if identity in ids]
                assert bool(ids) == bool(boxes)
                current['positive_contexts' if ids else 'empty_contexts'] += 1
                for template in templates[video, frame]:
                    sample = deepcopy(template)
                    sample['state']['task']['expression'] = query['sentence']
                    suffix = template['id'].split('_batch')[1]
                    batch, question_key = suffix.split('_', 1)
                    group = sample['state']['candidates']
                    if group:
                        candidate = group[int(question_key.removeprefix('membership_'))]
                        positive = any(overlap(candidate['box'], box) >= .5 for box in boxes)
                    else:
                        positive = bool(ids)
                    target = 'yes' if positive else 'no'
                    sample['target'] = {target: 1.}
                    sample['id'] = f'rmot_diverse_{video}_{frame}_q{query_index}_batch{suffix}'
                    sample['supervision'].update(query_index=query_index, expression_file=indexed['expression_file'],
                        expanded_expression_coverage_only_not_new_physical_frame=True,
                        expression_IDs_without_same_frame_released_box=unboxed,
                        target_source='released same-frame boxes with ID in expression; candidate IoU>=.5')
                    key = video, frame, query['sentence'], suffix
                    if key in legacy:
                        old = legacy[key]
                        assert all(sample[k] == old[k] for k in ('state', 'question', 'target', 'task'))
                        assert [(m['type'], m['w'], m['h']) for m in sample['media']] == [(m['type'], m['w'], m['h']) for m in old['media']]
                        matched_legacy.add(key)
                    output.write(json.dumps(sample, ensure_ascii=False)+'\n')
                    current['rows'] += 1
                    current[sample['task']+':'+target] += 1
                    counts[sample['task']+':'+target] += 1
            stats = dict(current)
            stats.update(positive_contexts=current['positive_contexts'], empty_contexts=current['empty_contexts'])
            stats.update(video=video, expression_file=indexed['expression_file'], sentence=query['sentence'],
                native_frames=len(by_video[video]['frames']), has_positive_context=current['positive_contexts'] > 0)
            coverage.append(stats)
            for totals in (counts, per_video[video]):
                totals.update(expressions=1, expressions_with_positive_context=stats['has_positive_context'],
                              positive_contexts=current['positive_contexts'], empty_contexts=current['empty_contexts'], rows=current['rows'])
    assert len(matched_legacy) == len(reference) == 346
    assert counts['expressions'] == plan['expression_files']
    assert counts['expressions_with_positive_context'] == plan['expression_files_with_positive_sample']
    assert counts['positive_contexts'] == plan['positive_query_frame_contexts']
    assert counts['empty_contexts'] == plan['empty_query_frame_contexts']
    assert counts['positive_contexts'] + counts['empty_contexts'] == plan['query_frame_contexts']
    (out/'expression_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2)+'\n')
    byte_count = sum(p.stat().st_size for p in out.iterdir())
    assert byte_count < plan['export_budget_bytes']
    report = {'status': 'RMOT_ALL_FIT_EXPRESSION_COMPONENT_GENERATED_NOT_TRAINED',
        'counts': dict(counts), 'per_video': {v: dict(c) for v, c in per_video.items()},
        'physical_frames': len(native), 'native_candidates': plan['native_candidates'],
        'rejected_native_frames': sum(bool(r['perception'].get('model_response_rejected')) for r in native.values()),
        'reference_rows_exact_state_question_target_task': len(matched_legacy),
        'reference_native_input_state_and_question_checks': len(templates),
        'actual_native_vs_existing_media_pixel_comparisons': pixel_comparisons,
        'legacy_query_media_pixel_pairs_compared': len(compared_media),
        'unboxed_target_contexts_retained_with_official_projection': len(missing),
        'annotation_contract_validation': str(args.annotation_contract) if args.annotation_contract else None,
        'existing_unique_media_paths_reused': len(reused_media), 'new_image_or_feature_artifacts': 0,
        'all_fit_expressions_and_recorded_frames_retained': True,
        'proposals_still_conditioned_on_original_recorded_primary_expression': True,
        'counterfactual_membership_not_new_expression_conditioned_detector_rollouts': True,
        'no_evaluation_or_forbidden_GT_supervision': True, 'GT_IDs_in_model_input': False,
        'no_actor_history_or_temporal_expression_observability_claim': True,
        'negative_membership_is_unchanged_GT_IoU_proxy_not_calibrated_uncertainty': True,
        'requires_positive_negative_aware_mixed_task_curriculum': True,
        'compressed_train_bytes': (out/'train.jsonl.gz').stat().st_size,
        'export_bytes_before_summary': byte_count, 'export_budget_bytes': plan['export_budget_bytes'],
        'combined_registered_upper_bytes': plan['combined_registered_upper_bytes'],
        'new_model_calls': False, 'new_GPU_training': False,
        'full_goal_complete': False}
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
