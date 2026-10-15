"""Posthoc natural candidate returns; registered visibility gaps are not detection gaps."""
from collections import Counter
import json
from pathlib import Path


def analyze(manifest, run):
    diagnostic = json.loads((Path(run)/'association_events.json').read_text())
    frames = manifest['frame_numbers']
    position = {frame: i for i, frame in enumerate(frames)}
    fps = manifest['fps']
    case = manifest['challenge_registration']
    counts, returns = Counter(), []
    birth, previous, last_emitted = {}, {}, {}
    selected = {}
    for event in diagnostic['events']:
        gid, tid, frame = event['matched_gt_identity'], event['assigned_track_id'], event['frame']
        if gid is not None:
            if gid == case['identity']:
                selected[frame] = tid
            if gid in previous:
                before = previous[gid]
                gap = (frame-before)/fps
                if position[frame]-position[before] > 1 and 3 < gap <= 30:
                    counts['natural_candidate_returns_3to30s'] += 1
                    old = last_emitted.get(gid)
                    known = old is not None and birth.get(old) == gid
                    if known:
                        counts['known_birth_anchor_eligible_returns'] += 1
                        survived = old in event['correct_surviving_anchor_ids']
                        recovered = tid == old and event['past_anchor_gt_identity'] == gid
                        counts['previous_correct_ID_survives_before_return'] += survived
                        counts['same_known_correct_birth_ID_recovered'] += recovered
                    else:
                        counts['unknown_or_no_previous_correct_birth_anchor_excluded'] += 1
                        survived = recovered = None
                    returns.append({'GT_identity': gid, 'before': before, 'after': frame,
                        'candidate_gap_seconds': gap, 'previous_emitted_ID': old, 'return_ID': tid,
                        'known_birth_anchor_eligible': known, 'old_correct_ID_survives': survived,
                        'recovered_same_known_ID': recovered})
            previous[gid] = frame
            if tid is not None:
                last_emitted[gid] = tid
        if event['action'] == 'NEW':
            birth[tid] = gid
    window = [f for f in frames if case['before'] <= f <= case['after']]
    longest, missing = 0, 0
    for frame in window:
        missing = missing+1 if frame not in selected else 0
        longest = max(longest, missing)
    n = counts['known_birth_anchor_eligible_returns']
    report = {'status': 'NATURAL_CANDIDATE_RETURN_AND_REGISTERED_VISIBILITY_CASE_POSTHOC',
        'frames': len(frames), 'physical_fps': fps, 'registered_visibility_case': case,
        'selected_identity_candidate_coverage': {'registered_interval_frames': len(window),
            'covered_frames': sum(f in selected for f in window),
            'candidate_present_at_exact_registered_before': case['before'] in selected,
            'candidate_present_at_exact_registered_after': case['after'] in selected,
            'maximum_consecutive_no_aligned_candidate_frames': longest,
            'maximum_consecutive_missing_candidate_seconds': longest/fps,
            'not_assumed_equal_to_registered_visibility_gap': True},
        'counts': dict(counts), 'known_birth_anchor_recovery_rate_3to30s':
            counts['same_known_correct_birth_ID_recovered']/n if n else None,
        'all_candidate_returns_3to30s': returns,
        'existing_unmodified_posthoc_all_gap_buckets': diagnostic['posthoc_identity_recovery'],
        'GT_used_posthoc_only': True, 'runtime_state_or_official_metric_modified': False,
        'policy': 'All physical marked pedestrian GT, original IoU.5 one-to-one alignment; no visibility filter or artificial proposal removal. New recovery denominator requires a known correct immutable birth anchor; unknown/no anchors are counted separately, not fabricated negatives. Gap lengths are between actual aligned candidates, not GTvisibility endpoints.',
        'official_HOTA_IDF1_replacement': False}
    (Path(run)/'natural_recovery_case.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
