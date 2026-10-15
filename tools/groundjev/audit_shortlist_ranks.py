"""Exact causal replay, frozen-trajectory shortlist ranks; no counterfactual accuracy."""
import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from locatemot.groundjev.decision import overlap
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import restore_completed_prefix
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    diagnostic = json.loads((args.run/'association_events.json').read_text())
    observations = {(r['frame'], r['candidate']): r for r in diagnostic['events']}
    specification = {'task': manifest['task'], 'vocabulary': manifest['vocabulary'],
                     'expression': manifest.get('expression')}
    tracker = OnlineTracker(None)
    ranks, coverage, counts = Counter(), Counter(), Counter()
    examples = []
    with args.source.open() as native, (args.run/'frame_decisions.jsonl').open() as recorded:
        for raw, saved in zip_longest(native, recorded):
            assert raw is not None and saved is not None
            raw, saved = json.loads(raw), json.loads(saved)
            index = counts['frames']
            assert raw['frame'] == saved['frame'] == manifest['frame_numbers'][index]
            assert raw['image_path'] == saved['image_path'] == manifest['images'][index]
            geometry = [{k: v for k, v in c.items() if k not in
                ('native_region_feature', 'temporal_region_feature')} for c in raw['candidates']]
            assert saved['candidates'] == geometry
            events = {e['candidate']: e for e in saved['state']['events'] if 'candidate' in e}
            timestamp = saved['state']['timestamp']
            tracks = list(tracker.tracks.values())
            counts['maximum_live_tracks'] = max(counts['maximum_live_tracks'], len(tracks))
            counts['frames_with_more_than32_live_tracks'] += len(tracks) > 32
            for begin in range(0, len(raw['candidates']), 16):
                group = raw['candidates'][begin:begin+16]
                ranked = sorted(tracks, key=lambda t: (-max(overlap(c['box'], t.predicted_box(timestamp))
                    for c in group), t.lost_age, t.track_id))
                ids = [t.track_id for t in ranked]
                positions = {tid: i+1 for i, tid in enumerate(ids)}
                for offset, candidate in enumerate(group):
                    current = begin+offset
                    assert events[current]['decision']['shortlisted_ids'] == ids[:32]
                    observation = observations[(saved['frame'], current)]
                    existing = observation['correct_surviving_anchor_ids']
                    if not existing:
                        continue
                    assert set(existing) <= set(ids)
                    rank = min(positions[tid] for tid in existing)
                    ranks[rank] += 1
                    counts['existing_identity_events'] += 1
                    for size in (32, 64, 128):
                        coverage[size] += rank <= size
                    if rank > 32 and len(examples) < 16:
                        examples.append({'frame': saved['frame'], 'candidate': current,
                            'best_correct_anchor_rank': rank, 'live_tracks': len(ids),
                            'correct_anchor_ids': existing})
            # Only the original raw candidates and recorded decisions enter the
            # real tracker. Posthoc GT ownership above never changes its state.
            complete = dict(saved, candidates=raw['candidates'])
            restore_completed_prefix(tracker, [complete], [(saved['image_path'], saved['frame'])],
                manifest['fps'], specification, {saved['frame']: timestamp})
            assert list(tracker.tracks) == saved['state']['persistent_ids']
            counts['frames'] += 1
            counts['candidates'] += len(raw['candidates'])
    assert counts['frames'] == len(manifest['images'])
    n = counts['existing_identity_events']
    assert n == diagnostic['counts']['existing_identity_events']
    assert coverage[32] == diagnostic['counts']['correct_identity_option_present']
    report = {'status': 'EXACT_CAUSAL_REPLAY_FROZEN_TRAJECTORY_SHORTLIST_RANKS_VALIDATED',
        'manifest': str(args.manifest), 'source': str(args.source), 'run': str(args.run),
        'counts': dict(counts), 'candidate_group_size': 16, 'actual_shortlist_size': 32,
        'fixed_original_trajectory_coverage': {str(k): coverage[k]/n if n else None for k in (32, 64, 128)},
        'all_live_id_upper_bound': 1. if n else None,
        'minimum_rank_of_any_correct_surviving_birth_anchor_histogram': dict(sorted(ranks.items())),
        'first16_rank_gt32_examples': examples,
        'every_actual_outputs_and_state_replayed_exactly': True,
        'GT_input_to_tracker_or_repaired_history': False,
        'GT_ownership_read_from_completed_posthoc_diagnostic_only': True,
        'counterfactual_tracking_accuracy_or_new_model_inference': False,
        'limit': 'Coverage only under the original causal trajectory. Larger context changes model softmax/context and subsequent histories; requires actual new full-video runs. Excludes already terminated or never correctly born identities.',
        'new_feature_cache_files_or_GPUs': False}
    (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
