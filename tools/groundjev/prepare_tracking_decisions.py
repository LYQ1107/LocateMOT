"""Legal real-image clip supervision for the official OneJev multimodal SFT.

This first curriculum stage uses GT proposals and teacher identity states. It
is explicitly distinct from detector/model rollout and from benchmark output.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'external/OneJev')]
PROTOCOL = Path('/data2/user/LocateMOT/outputs/unified/u2/protocol')


def annotations(sequence, allowed):
    rows = defaultdict(list)
    with open(sequence['gt_path']) as source:
        for raw in csv.reader(source):
            frame, identity = int(raw[0]), int(raw[1])
            if frame not in allowed or int(float(raw[6])) != 1 or int(float(raw[7])) != 1:
                continue
            x, y, w, h = map(float, raw[2:6])
            rows[frame].append({'gt_id': identity, 'box': [x, y, x+w, y+h],
                                'visibility': float(raw[8]), 'label': 'person'})
    return rows


def mine_clips(frames, gt, count, length):
    from locatemot.groundjev.decision import overlap
    scored = []
    for stride in (1, 4):
        span = (length-1)*stride+1
        for start in range(0, len(frames)-span+1, max(4, length//2)):
            clip = frames[start:start+span:stride]
            score = 0.
            previous = set()
            for frame in clip:
                targets = gt[frame]
                score += sum(1-t['visibility'] for t in targets)
                score += sum(overlap(a['box'], b['box']) for i, a in enumerate(targets) for b in targets[i+1:])
                current = {t['gt_id'] for t in targets}
                score += .25*len(previous-current)
                previous = current
            scored.append((score, stride, clip))
    selected = []
    # Take both dense contiguous interactions and wider-time real observations.
    for stride in (1, 4):
        for score, s, clip in sorted(scored, key=lambda x: (-x[0], x[2][0])):
            if s != stride or any(len(set(clip)&set(c[2])) > length//2 for c in selected):
                continue
            selected.append((score, stride, clip))
            if sum(c[1] == stride for c in selected) >= count//2:
                break
    return selected


class TrackDecisionDataset:
    """Export same runtime question/state schema, with labels kept outside state."""
    def __init__(self, out, seed, candidates=6, history=16):
        from locatemot.groundjev.decision import OneJevDecision
        self.builder = OneJevDecision.__new__(OneJevDecision)
        self.out, self.seed = out, seed
        self.candidates, self.history = candidates, history
        self.counts, self.clips, self.rows = Counter(), [], []

    def emit(self, identifier, request, question, target, media, metadata):
        row = {'id': identifier, 'split': 'train', 'source': metadata['sequence'],
               'task': metadata['kind'], 'state': request.state,
               'question': question.model_dump(exclude_none=True), 'target': {str(target): 1.},
               'media': media, 'supervision': metadata}
        self.rows.append(row)
        self.counts[metadata['kind']+':'+str(target)] += 1

    def add_clip(self, sequence, gt, frame_names, score, stride, frames, clip_index):
        from PIL import Image
        from locatemot.groundjev.decision import overlap
        from locatemot.groundjev.tracker import Track
        rng = random.Random(self.seed+clip_index*1009)
        identities = sorted({t['gt_id'] for frame in frames for t in gt[frame]})
        aliases = dict(zip(identities, rng.sample(range(100, 9999), len(identities))))
        tracks, recoveries, close_pairs, low_visibility = {}, 0, 0, 0
        clip_name = sequence['sequence']+f'_clip{clip_index:03d}'
        for step, frame in enumerate(frames):
            path = Path(sequence['image_dir'])/frame_names[frame]
            with Image.open(path) as source:
                image = source.convert('RGB')
            timestamp = frame/sequence['fps']
            current = gt[frame]
            ordered = sorted(current, key=lambda c: (
                -max((overlap(c['box'], o['box']) for o in current if o['gt_id'] != c['gt_id']), default=0.),
                c['visibility'], c['gt_id']))
            group = ordered[:self.candidates]
            ranked = sorted(tracks.values(), key=lambda t: (
                -max((overlap(c['box'], t.predicted_box(timestamp)) for c in group), default=0.),
                t.lost_age, t.track_id))[:self.history]
            request, runtime_media = self.builder.build_request(image, group, ranked, timestamp,
                {'task': 'mot', 'vocabulary': ['person'], 'expression': None})
            media = [{'type': 'image', 'path': str(path), 'w': image.width, 'h': image.height}]
            if len(runtime_media) == 2:
                sheet = runtime_media[1]['image']
                sheet_path = self.out/'history'/f'{clip_name}_{frame:06d}.png'
                sheet_path.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(sheet_path)
                media.append({'type': 'image', 'path': str(sheet_path), 'w': sheet.width, 'h': sheet.height})
            info = {'sequence': sequence['sequence'], 'frame': frame, 'clip': clip_name,
                    'observed_clip_step': step, 'clip_length': len(frames), 'stride': stride,
                    'candidate_source': 'legal_training_gt_proposal', 'state_source': 'teacher_causal_gt_history',
                    'gt_used_for_supervision': True, 'model_rollout': False,
                    'identity_alias_permuted': True, 'future_observations_in_state': False}
            by_alias = {t.track_id for t in ranked}
            for i, candidate in enumerate(group):
                alias = aliases[candidate['gt_id']]
                target = str(alias) if alias in by_alias else 'NEW' if alias not in tracks else None
                if target is None:
                    # Real shortlist failure is retained as a separate question,
                    # never relabeled as a new identity.
                    q = request.questions[f'match_{i}'].model_copy(deep=True)
                    q.criteria['NO_MATCH'] = 'A previously observed identity has no matching option in this shortlist; retain it without creating a new ID'
                    target = 'NO_MATCH'
                    kind = 'missing_shortlist_identity'
                else:
                    q, kind = request.questions[f'match_{i}'], 'identity'
                self.emit(f'{clip_name}_{frame}_{i}_match', request, q, target, media, info|{'kind': kind})
                reliable = candidate['visibility'] >= .6
                self.emit(f'{clip_name}_{frame}_{i}_update', request, request.questions[f'update_{i}'],
                          'yes' if reliable else 'no', media, info|{'kind': 'memory_update'})
                level = '3' if candidate['visibility'] >= .85 else '2' if reliable else '1'
                self.emit(f'{clip_name}_{frame}_{i}_score', request, request.questions[f'reliability_{i}'],
                          level, media, info|{'kind': 'reliability', 'score_label_source': 'GT visibility ordinal auxiliary target'})
                if alias in by_alias and i == 0:
                    q = request.questions[f'match_{i}'].model_copy(deep=True)
                    del q.criteria[str(alias)]
                    q.criteria['NO_MATCH'] = 'The matching historical identity is absent from the listed options; hold the unresolved association'
                    self.emit(f'{clip_name}_{frame}_{i}_no_match', request, q, 'NO_MATCH', media,
                              info|{'kind': 'withheld_correct_option', 'option_withholding_is_training_augmentation': True})
                low_visibility += not reliable
            visible = {aliases[c['gt_id']] for c in current}
            for t in ranked:
                target = 'REACTIVATE' if t.track_id in visible and t.lost_age else 'ACTIVE' if t.track_id in visible else 'LOST'
                self.emit(f'{clip_name}_{frame}_life{t.track_id}', request, request.questions[f'life_{t.track_id}'],
                          target, media, info|{'kind': 'lifecycle'})
            close_pairs += sum(overlap(a['box'], b['box']) > .1 for i, a in enumerate(current) for b in current[i+1:])
            # Teacher observes this frame only after rendering its supervision.
            for c in current:
                alias = aliases[c['gt_id']]
                crop = image.crop(tuple(c['box'])).convert('RGB')
                crop.thumbnail((192, 192))
                reliable = c['visibility'] >= .6
                if alias not in tracks:
                    tracks[alias] = Track(alias, c['box'], 'person', crop.copy(), crop.copy(),
                        long_image=crop.copy(), timestamp=timestamp, last_reliable_timestamp=timestamp,
                        recent_boxes=[c['box']])
                else:
                    t = tracks[alias]
                    recoveries += bool(t.lost_age)
                    t.velocity = [(x-y)/(timestamp-t.timestamp) for x,y in zip(c['box'],t.box)]
                    t.box, t.timestamp, t.recent_image = c['box'], timestamp, crop.copy()
                    t.age += 1
                    t.recent_boxes = (t.recent_boxes+[c['box']])[-16:]
                    if reliable:
                        t.long_image, t.last_reliable_timestamp = crop.copy(), timestamp
                    t.lost_age, t.state = 0., 'ACTIVE'
            for alias, t in list(tracks.items()):
                if alias not in visible:
                    t.lost_age, t.state = timestamp-t.timestamp, 'LOST'
                    if t.lost_age > 3.:
                        del tracks[alias]
        self.clips.append({'sequence': sequence['sequence'], 'frames': frames,
            'clip_length': len(frames), 'stride': stride, 'hardness': score,
            'real_reappearances_after_missing_observation': recoveries,
            'close_pair_observations': close_pairs, 'low_visibility_candidates': low_visibility})

    def save(self):
        with (self.out/'train.jsonl').open('w') as target:
            for row in self.rows:
                target.write(json.dumps(row)+'\n')
        summary = {'status': 'LEGAL_REAL_IMAGE_TEACHER_CURRICULUM_READY_NOT_MODEL_ROLLOUT',
                   'rows': len(self.rows), 'seed': self.seed, 'counts': dict(self.counts), 'clips': self.clips,
                   'runtime_question_builder_reused': True, 'full_dataset_features_cached': False,
                   'feature_cache_budget_bytes': 25*2**30,
                   'history_png_bytes': sum(p.stat().st_size for p in (self.out/'history').glob('*.png')),
                   'official_test_labels_read': False, 'development_frames_used': False,
                   'next_curriculum': 'replace teacher proposals/states with real detector and own-model rollout; add verified false candidates and long occlusion events'}
        (self.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/groundjev/training_data/teacher_v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clips-per-sequence', type=int, default=6)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('preserve existing curriculum; choose a fresh output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = TrackDecisionDataset(args.out, args.seed)
    index = 0
    for protocol_name, name in [('mot17_half.json','MOT17-02-FRCNN'),
                                ('dancetrack.json','dancetrack0001'), ('mot20_half.json','MOT20-01')]:
        protocol = json.loads((PROTOCOL/protocol_name).read_text())
        sequence = next(s for s in protocol['sequences'] if s['sequence'] == name)
        frame_names = {int(Path(f).stem): f for f in sequence['train_frames']}
        frames = sorted(frame_names)
        gt = annotations(sequence, set(frames))
        for score, stride, clip in mine_clips(frames, gt, args.clips_per_sequence, 16):
            dataset.add_clip(sequence, gt, frame_names, score, stride, clip, index)
            index += 1
    dataset.save()


if __name__ == '__main__':
    main()
