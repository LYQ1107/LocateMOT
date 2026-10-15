"""Validate immutable completed Full MOT units without treating the owner as done."""
from itertools import zip_longest
from collections import Counter
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def decision_diagnosis(events):
    updates = Counter()
    assert len(events['events']) == 13664
    for event in events['events']:
        # NEW initializes memory; the original analyzer counts it separately
        # even if the event's write action is named UPDATE.
        if event['action'] == 'NEW' or event['memory_action'] != 'UPDATE':
            continue
        target, anchor = event['matched_gt_identity'], event['past_anchor_gt_identity']
        if target is None:
            updates['unmatched_candidate_not_universal_negative_truth'] += 1
        elif anchor is None:
            updates['matched_GT_unknown_birth_anchor'] += 1
        elif target == anchor:
            updates['matched_GT_known_consistent_birth_anchor'] += 1
        else:
            updates['matched_GT_known_conflicting_birth_anchor'] += 1
    counts = events['counts']
    assert sum(updates.values()) == counts['memory_updates_identity_consistent'] + counts['memory_updates_wrong_identity_or_unmatched']
    return {'candidate_GT_recall_at_iou05':events['candidate_GT_recall_at_iou05'],
        'mean_native_correct_identity_option_mass':events['mean_native_correct_identity_option_mass'],
        'association_and_memory_counts':counts,'memory_UPDATE_known_anchor_breakdown':dict(updates)}


def main():
    directory = ROOT/'outputs/groundjev/training/onejev_full1203_controls_v2_wait_fix'
    plan = json.loads((ROOT/'configs/groundjev/onejev_full1203_controls_v1.json').read_text())
    owner = json.loads((directory/'owner_state.json').read_text())
    scope = plan['scopes']['mot_full1203']
    manifest = json.loads((ROOT/scope['manifest']).read_text())
    expected_names = [arm+'_mot_full1203' for arm in plan['arms']]
    rows = {}
    for name, entry in owner['completed'].items():
        assert name in expected_names
        arm = name.removesuffix('_mot_full1203')
        run = ROOT/entry['run']
        runtime = json.loads((run/'result.json').read_text())
        official = json.loads((run/'official_metrics.json').read_text())
        events = json.loads((run/'association_events.json').read_text())
        assert official == entry['official']
        assert runtime['frames'] == official['frames'] == events['frames'] == entry['decoded_frames'] == 1203
        assert runtime['all_input_frames_processed']
        assert not runtime['gt_read_by_runtime'] and not runtime['future_frames_read_by_runtime']
        assert runtime['backend'] == 'full' and runtime['evidence_format'] == 'v3_crop_motion'
        assert runtime['end_to_end_fps'] is None and runtime['resumed_completed_frames'] == 0
        assert Path(runtime['checkpoint']).resolve() == (ROOT/plan['arms'][arm]['checkpoint']).resolve()
        assert runtime['perception_rejected_frames'] == official['perception_rejected_frames']
        assert len(runtime['perception_rejected_frames']) == 27
        assert Path(runtime['shared_input_features_source']).resolve() == (ROOT/scope['candidates']).resolve()
        assert (ROOT/runtime['final_identity_memory_path']).is_file()
        count, frames = 0, []
        with (ROOT/scope['candidates']).open() as source, (run/'frame_decisions.jsonl').open() as trace:
            for before, after in zip_longest(source, trace):
                assert before is not None and after is not None
                before, after = json.loads(before), json.loads(after)
                expected = [{k:v for k,v in c.items() if k not in
                    ('native_region_feature','temporal_region_feature')} for c in before['candidates']]
                assert after['frame'] == before['frame'] and after['image_path'] == before['image_path']
                assert after['candidates'] == expected
                count += len(expected)
                frames.append(after['frame'])
        assert frames == manifest['frame_numbers'] and count == 13664
        decoded = int(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-count_frames','-show_entries','stream=nb_read_frames','-of',
            'default=noprint_wrappers=1:nokey=1',str(run/'tracks.mp4')],text=True))
        assert decoded == 1203
        rows[name] = {'run':str(run),'frames':decoded,'native_candidates':count,
            'same_exact_native_V1_candidates':True,'parser_rejected_frame_count':27,
            'metrics':official['metrics'],'checkpoint':runtime['checkpoint'],
            'correct_option_coverage':events['correct_option_coverage'],
            'existing_association_accuracy':events['existing_association_accuracy'],
            **decision_diagnosis(events),
            'posthoc_identity_recovery':events['posthoc_identity_recovery'],
            'perception_excluded_replay_FPS':runtime['association_replay_fps'],
            'actual_end_to_end_FPS':None}
    pending = [name for name in expected_names if name not in rows and name not in owner['failed']]
    report = {'status':'COMPLETED_FULL_MOT_UNITS_OFFICIAL_ARTIFACTS_VALIDATED',
        'results':rows,'pending_units':pending,'failed_units':owner['failed'],
        'all_three_registered_units_complete':len(rows)==3 and not owner['failed'],
        'full_goal_complete':False,
        'limits':'One complete legal DanceTrack0004 video, one seed, replay of the original '
            '13664 V1 candidates/27 rejected frames; no E2E FPS or complete-dataset claim. '
            'Teacher/native checkpoints add600updates to MOT500 (1100total), not600total. '
            'Shortlist coverage and decision accuracy use each model own endogenous event set, '
            'not a strict same-event causal comparison. Pending units have no inferred scores.'}
    (directory/'completed_units_validated_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'completed':list(rows),'pending':pending}))


if __name__ == '__main__':
    main()
