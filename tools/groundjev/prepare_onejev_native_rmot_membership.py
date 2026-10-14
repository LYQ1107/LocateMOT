"""Instantaneous expression supervision on every recorded native fit candidate."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    from PIL import Image
    from locatemot.groundjev.decision import OneJevDecision, overlap
    from locatemot.unified.data.legal_scope import assert_legal_video, LEGAL_SPLITS
    from qev.prompt import render_question
    from train import common
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    builder = OneJevDecision.__new__(OneJevDecision)
    builder.evidence_format, builder.rmot_membership = 'v3_crop_motion', True
    plan = json.loads((ROOT/'configs/groundjev/rmot_native72_20261014.json').read_text())
    source = ROOT/'outputs/groundjev/compact/rmot_native72_v1/native_rollout.jsonl'
    rows = {(r['sequence'], r['frame']): r for r in map(json.loads, source.open())}
    counts, frame_scope = Counter(), []
    with (args.out/'train.jsonl').open('w') as output:
        for clip in plan['clips']:
            video = assert_legal_video(clip['video'], dataset='refer_kitti_v2')
            assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
            queries = [json.loads(Path(clip[k]).read_text()) for k in ('primary_expression_file', 'counter_expression_file')]
            for frame in clip['frames']:
                row = rows[video, frame]
                candidates = row['candidates']
                with Image.open(row['image_path']) as raw:
                    image = raw.convert('RGB')
                counts['frames'] += 1
                counts['native_candidates'] += len(candidates)
                counts['rejected_native_frames'] += bool(row['perception'].get('model_response_rejected'))
                registered = {'video': video, 'frame': frame, 'native_candidates': len(candidates),
                              'native_response_rejected': bool(row['perception'].get('model_response_rejected')), 'query_rows': []}
                for qi, query in enumerate(queries):
                    spec = {'task': 'rmot', 'vocabulary': ['person', 'car'], 'expression': query['sentence']}
                    target_ids = set(query['label'].get(str(frame), []))
                    # Supervision stays outside every native request state.
                    labels = {}
                    if target_ids:
                        release_root = Path(clip['primary_expression_file']).parents[2]
                        label_path = release_root/'labels_with_ids/image_02'/video/f'{frame:06d}.txt'
                        for line in label_path.read_text().splitlines():
                            f = line.split(); x, y, w, h = map(float, f[2:6])
                            labels[int(f[1])] = [x*image.width, y*image.height, (x+w)*image.width, (y+h)*image.height]
                        assert target_ids <= labels.keys()
                    boxes = [labels[i] for i in target_ids]
                    groups = [candidates[i:i+16] for i in range(0, len(candidates), 16)] or [[]]
                    query_rows = 0
                    for batch, group in enumerate(groups):
                        request, media = builder.build_request(image, group, [], frame/clip['fps'], spec)
                        files = [{'type': 'image', 'path': row['image_path'], 'w': image.width, 'h': image.height}]
                        for number, item in enumerate(media[1:], 2):
                            directory = args.out/'evidence'; directory.mkdir(exist_ok=True)
                            path = directory/f'{video}_{frame:06d}_q{qi}_batch{batch}_image{number}.png'
                            item['image'].save(path)
                            files.append({'type': 'image', 'path': str(path), 'w': item['image'].width, 'h': item['image'].height})
                        targets = [(f'membership_{i}', 'yes' if any(overlap(c['box'], b) >= .5 for b in boxes) else 'no')
                                   for i, c in enumerate(group)] or [('empty', 'yes' if target_ids else 'no')]
                        for key, target in targets:
                            question = request.questions[key]
                            common.target_vector({target: 1.}, render_question('q', question).labels)
                            kind = 'query_membership' if group else 'empty_query_presence'
                            sample = {'id': f'native_rmot_{video}_{frame}_q{qi}_batch{batch}_{key}',
                                'split': 'train', 'source': 'refer_kitti_'+video, 'task': kind,
                                'state': request.state, 'question': question.model_dump(exclude_none=True),
                                'target': {target: 1.}, 'media': files,
                                'supervision': {'frame': frame, 'query_index': qi, 'candidate_source': 'actual_native_GroundAnything_DLM',
                                    'instantaneous_query_supervision_no_OneJev_actor_history_claim': True,
                                    'GT_IDs_in_model_input': False, 'actor_state_corrected_by_GT': False,
                                    'target_source': 'fit release expression IDs and exact GT boxes at IoU>=.5',
                                    'unmatched_native_candidates': 'explicit negative membership proxy, not calibrated uncertainty truth',
                                    'evidence_format': 'v3_crop_motion', 'future_images_read': False}}
                            output.write(json.dumps(sample)+'\n'); query_rows += 1
                            counts[kind+':'+target] += 1
                    registered['query_rows'].append(query_rows)
                    counts['query_contexts'] += 1
                frame_scope.append(registered)
    assert counts['frames'] == len(rows) == 72 and counts['query_contexts'] == 144
    assert counts['native_candidates'] == 165 and sum(counts['query_membership:'+k] for k in ('yes', 'no')) == 330
    assert counts['query_membership:yes'] == 58
    summary = {'status': 'REAL_NATIVE_FIT_RMOT_QUERY_SUPERVISION_COMPLETE', 'counts': dict(counts),
        'rows': sum(sum(r['query_rows']) for r in frame_scope), 'registered_frames': frame_scope,
        'actual_recorded_native_source': str(source), 'all_native_candidates_retained_for_both_queries': True,
        'registered_empty_and_rejected_frames_retained': True, 'validation_or_test_supervision_read': False,
        'GT_in_runtime_state': False, 'OneJev500_own_history_rollout_claim': False,
        'instantaneous_expression_training_not_identity_training': True,
        'native_runtime_or_new_training_complete': False,
        'evidence_png_bytes': sum(p.stat().st_size for p in (args.out/'evidence').glob('*.png')),
        'OVMOT_Base_category_SFT_still_required': True}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'registered_frames'}, indent=2))


if __name__ == '__main__':
    main()
