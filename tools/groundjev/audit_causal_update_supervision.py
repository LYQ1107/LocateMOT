"""Fit-only audit: does the existing update label protect the actually chosen identity?"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from scipy.optimize import linear_sum_assignment
    from locatemot.groundjev.decision import overlap
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_video
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = ROOT/'outputs/groundjev/compact/rmot_native72_v1'
    pack = torch.load(root/'broad_native/examples.pt', map_location='cpu', weights_only=True)
    native = {e['id']: e for e in pack['examples'] if e['id'].startswith('native_rmot_')}
    counts, examples, conflicts = Counter(), [], []
    anchors, prior_ids, sequence = {}, set(), None
    checked = set()
    with (root/'native_rollout.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            video = assert_legal_video(row['sequence'], dataset='refer_kitti_v2')
            assert video in LEGAL_SPLITS['refer_kitti_v2']['fit']
            if video != sequence:
                sequence, anchors, prior_ids = video, {}, set()
            before = dict(anchors)
            label = ROOT.parent/'LocateMOT/data/refer_kitti_v2/labels_with_ids/image_02'/video/f"{row['frame']:06d}.txt"
            from PIL import Image
            with Image.open(row['image_path']) as image:
                width, height = image.size
            targets = []
            for line in label.read_text().splitlines() if label.exists() else []:
                f = line.split(); x, y, w, h = map(float, f[2:6])
                targets.append({'id': int(f[1]), 'box': [x*width, y*height, (x+w)*width, (y+h)*height]})
            matched, quality = {}, {}
            if row['candidates'] and targets:
                pairwise = np.asarray([[overlap(c['box'], g['box']) for g in targets] for c in row['candidates']])
                rr, cc = linear_sum_assignment(-pairwise)
                for i, j in zip(rr, cc):
                    if pairwise[i, j] >= .5:
                        matched[int(i)] = targets[int(j)]['id']
                        quality[int(i)] = float(pairwise[i, j])
            events = {e['candidate']: e for e in row['state']['events'] if 'candidate' in e}
            old_labels = []
            for i in range(len(row['candidates'])):
                event = events[i]
                gid = matched.get(i)
                ranked = event['decision']['shortlisted_ids']
                correct_option = any(gid is not None and before.get(t) == gid for t in ranked)
                known = any(gid is not None and before.get(t) == gid for t in prior_ids)
                old = float(quality.get(i, 0.) >= .7 and not (known and not correct_option))
                old_labels.append(old)
                assigned = event.get('track_id')
                counts['native_candidates'] += 1
                if assigned is None:
                    counts['no_assigned_identity_update_target_not_observable'] += 1
                    continue
                consistent = gid is not None and (event['action'] == 'NEW' or before.get(assigned) == gid)
                known_conflict = gid is not None and before.get(assigned) is not None and before[assigned] != gid
                unknown = not consistent and not known_conflict
                proposed = None if unknown else float(consistent and quality.get(i, 0.) >= .7)
                counts['assigned_candidate_events'] += 1
                counts['old_quality_proxy_positive'] += bool(old)
                counts['proposed_selected_identity_consistent_positive'] += bool(proposed)
                counts['selected_identity_unknown_not_a_proven_negative'] += unknown
                counts['old_positive_with_unknown_identity_not_a_proven_error'] += bool(old and unknown)
                counts['old_positive_with_two_known_conflicting_identities'] += bool(old and known_conflict)
                counts['old_positive_but_selected_identity_inconsistent'] += bool(old and not consistent)
                counts['old_zero_but_selected_identity_consistent_high_quality'] += bool(not old and proposed)
                if event['action'] != 'NEW' and event['memory_action'] == 'UPDATE':
                    counts['existing_memory_writes'] += 1
                    counts['existing_writes_inconsistent_or_unmatched_birth_anchor'] += not consistent
                    if gid is not None and before.get(assigned) is not None and before[assigned] != gid:
                        counts['existing_writes_between_two_different_annotated_identities'] += 1
                if old and not consistent and len(examples) < 12:
                    examples.append({'video': video, 'frame': row['frame'], 'candidate': i,
                        'action': event['action'], 'selected_track_id': assigned,
                        'immutable_birth_GT_identity': before.get(assigned), 'current_candidate_GT_identity': gid,
                        'IoU': quality.get(i, 0.), 'old_update_target': old,
                        'proposed_actual_assigned_identity_update_target': proposed,
                        'identity_status': 'unknown_not_a_proven_negative' if unknown else 'known_conflict',
                        'actual_memory_action': event['memory_action']})
                if known_conflict and len(conflicts) < 12:
                    conflicts.append({'video': video, 'frame': row['frame'], 'candidate': i,
                        'selected_track_id': assigned, 'birth_GT_identity': before[assigned],
                        'current_GT_identity': gid, 'IoU': quality.get(i, 0.),
                        'old_update_target': old, 'actual_memory_action': event['memory_action']})
            for batch, begin in enumerate(range(0, len(old_labels), 16)):
                for query in (0, 1):
                    name = f"native_rmot_{video}_{row['frame']}_{batch}_query{query}"
                    assert native[name]['update_target'].tolist() == old_labels[begin:begin+16]
                    checked.add(name)
            for i, event in events.items():
                if event['action'] == 'NEW':
                    anchors[event['track_id']] = matched.get(i)
            prior_ids = set(row['state']['persistent_ids'])
            counts['frames'] += 1
    assert counts['frames'] == 72 and counts['native_candidates'] == 165
    assert checked == {k for k, e in native.items() if len(e['candidates'])}
    report = {'status': 'ACTUAL_NATIVE_FIT_UPDATE_PROXY_AND_SELECTED_IDENTITY_AUDITED',
        'counts': dict(counts), 'checked_nonempty_primary_and_counter_group_labels': len(checked),
        'native_groups_in_pack': len(native), 'first12_inconsistent_positive_examples': examples,
        'first12_known_identity_conflicts': conflicts,
        'all_original_proxy_labels_reproduced_exactly': True,
        'original_pack_checkpoint_outputs_or_memory_modified': False,
        'GT_access': 'legal fit supervision audit only, never validation training or actor-state correction',
        'unknown_anchor_policy': 'A birth without GT ownership does not prove the later association wrong. Mask unobserved/unknown identity supervision rather than inventing a negative target; no model labels are changed by this audit.',
        'hypothesis_not_proven_causal_gain': 'Distinguish two known conflicting identities from unknown birth ownership. Only the former demonstrates a contradictory positive label. Current-policy rollout training remains required, but this audit does not prove the existing positive labels cause the full-video failures.',
        'no_new_features_or_GPU': True}
    (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
