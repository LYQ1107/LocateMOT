"""Registered native-grid dense/ref data and real decoded-proposal identity pairs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data2/user/LocateMOT/data')


def build_pairs(plan):
    import numpy as np
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    from grounding_anything.protocol import GroundingEntry, serialize_answer, build_dense_bbox_prompt, build_refer_bbox_prompt, validate_prompt_answer_pair, canonicalize_phrase
    from locatemot.groundjev.decision import overlap
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_video
    from tools.groundjev.prepare_tracking_decisions import annotations, PROTOCOL
    native = {}
    for filename in plan['native_proposal_source']:
        for line in (ROOT/filename).open():
            row = json.loads(line)
            native[(row['sequence'], row['frame'])] = row['candidates']
    pairs = []
    counts = {'rows': 0, 'dense_rows': 0, 'referring_rows': 0, 'negative_referring_rows': 0,
              'native_proposals': 0, 'native_aligned_regions': 0}
    def row(identifier, path, gt, selected, phrase, family, frame):
        with Image.open(path) as source:
            width, height = source.size
        boxes = [g['box'] for g in gt if g['gt_id'] in selected]
        assert len(boxes) == len(selected)
        assert all(0 <= b[0] < b[2] <= width-1 and 0 <= b[1] < b[3] <= height-1 for b in boxes)
        coordinates = tuple(tuple(round(x/(width-1 if i%2 == 0 else height-1)*999) for i, x in enumerate(b)) for b in boxes)
        answer = serialize_answer([GroundingEntry(canonicalize_phrase(phrase), coordinates or None)], 'bbox')
        prompt = build_dense_bbox_prompt([phrase]) if family == 'dense' else build_refer_bbox_prompt(phrase)
        validate_prompt_answer_pair(prompt, answer, 'dense_bbox' if family == 'dense' else 'refer_bbox')
        decoded = native[(identifier, frame)]
        matched = {}
        if decoded and gt:
            scores = np.asarray([[overlap(c['box'], g['box']) for g in gt] for c in decoded])
            rr, cc = linear_sum_assignment(-scores)
            matched = {gt[int(j)]['gt_id']: decoded[int(i)]['box'] for i, j in zip(rr, cc) if scores[i, j] >= .5}
        counts['rows'] += 1; counts[family+'_rows' if family == 'dense' else 'referring_rows'] += 1
        counts['negative_referring_rows'] += int(family == 'referring' and not boxes)
        counts['native_proposals'] += len(decoded); counts['native_aligned_regions'] += len(matched)
        return {'sft': {'id': identifier+f'_{frame}_{family}_{counts["rows"]}',
                       'images': [{'path': str(path), 'bytes': path.read_bytes()}],
                       'messages': [{'role': 'user', 'content': '<image>\n'+prompt}, {'role': 'assistant', 'content': answer}]},
                'image_path': str(path), 'frame': frame, 'gt': {g['gt_id']: g['box'] for g in gt},
                'native': matched, 'selected_target_count': len(boxes)}
    for clip in plan['dense_clips']:
        sequence = next(s for s in json.loads((PROTOCOL/clip['protocol']).read_text())['sequences'] if s['sequence'] == clip['sequence'])
        names = {int(Path(p).stem): p for p in sequence['train_frames']}
        assert set(clip['frames']) <= set(names)
        gt = annotations(sequence, set(clip['frames']))
        rows = [row(clip['sequence'], Path(sequence['image_dir'])/names[n], gt[n],
                    {g['gt_id'] for g in gt[n]}, 'person', 'dense', n) for n in clip['frames']]
        pairs.extend({'family': 'dense', 'sequence': clip['sequence'], 'rows': rows[i:i+2]} for i in range(0, len(rows), 2))
    referring = json.loads((ROOT/plan['referring_clips_source'].split(',')[0]).read_text())
    absent = {}
    if plan.get('absent_expression_registry'):
        registry = json.loads((ROOT/plan['absent_expression_registry']).read_text())
        absent = {entry['video']: entry for entry in registry['entries']}
    for clip in referring['clips']:
        if clip['video'] not in plan['referring_fit_videos']:
            continue
        video = assert_legal_video(clip['video'], dataset='refer_kitti_v2')
        assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
        gt = {}
        for number in clip['frames']:
            path = DATA/'kitti_tracking_training/image_02'/video/f'{number:06d}.png'
            with Image.open(path) as source:
                w, h = source.size
            labels = DATA/'refer_kitti_v2/labels_with_ids/image_02'/video/f'{number:06d}.txt'
            targets = []
            for line in labels.read_text().splitlines() if labels.exists() else []:
                f = line.split(); x, y, bw, bh = map(float, f[2:6])
                targets.append({'gt_id': int(f[1]), 'box': [x*w, y*h, (x+bw)*w, (y+bh)*h]})
            gt[number] = targets
        for key in ('primary_expression_file', 'counter_expression_file'):
            query = json.loads(Path(clip[key]).read_text()); assert not query.get('ignore')
            rows = [row(video, DATA/'kitti_tracking_training/image_02'/video/f'{n:06d}.png',
                        gt[n], set(query['label'].get(str(n), [])), query['sentence'], 'referring', n) for n in clip['frames']]
            pairs.extend({'family': 'referring', 'sequence': video, 'rows': rows[i:i+2]} for i in range(0, len(rows), 2))
        if video in absent:
            entry = absent[video]
            assert entry['frames'] == clip['frames']
            query = json.loads(Path(entry['expression_file']).read_text())
            assert query['sentence'] == entry['expression'] and not query.get('ignore')
            assert all(not query['label'].get(str(n), []) for n in clip['frames'])
            rows = [row(video, DATA/'kitti_tracking_training/image_02'/video/f'{n:06d}.png',
                        gt[n], set(), query['sentence'], 'referring', n) for n in clip['frames']]
            pairs.extend({'family': 'referring', 'sequence': video, 'absent_expression': True,
                          'rows': rows[i:i+2]} for i in range(0, len(rows), 2))
    assert len(pairs) == plan['two_image_pairs'] and counts['rows'] == plan['native_SFT_rows']
    counts.update(pairs=len(pairs), unique_images=len({r['image_path'] for p in pairs for r in p['rows']}),
                  native_pair_contrastive_support=sum(len(set(p['rows'][0]['native']) & set(p['rows'][1]['native'])) > 1 for p in pairs))
    assert counts['unique_images'] == plan['unique_images']
    if absent:
        assert counts['negative_referring_rows'] == plan['negative_referring_rows']
    return pairs, counts
