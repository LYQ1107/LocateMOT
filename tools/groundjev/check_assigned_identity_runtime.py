"""Real-data CPU execution checks; no new model predictions or tracking scores."""
import argparse
from contextlib import ExitStack
from copy import copy, deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]
PARENT_TRACKER_BLOB = '9f092b04d6743c99495b9179f8f5ef25718bcb43'


def saved_decisions(row):
    decisions = [None]*len(row['candidates']); lifecycle = {}
    for event in row['state']['events']:
        if 'candidate' in event:
            decisions[event['candidate']] = deepcopy(event['decision'])
        if event.get('lifecycle_decision') is not None:
            lifecycle[event['track_id']] = deepcopy(event['lifecycle_decision'])
    assert all(d is not None for d in decisions)
    return decisions, lifecycle, deepcopy(row['state']['decision_metadata'])


class RecordedDecision:
    def decide(self, image, candidates, tracks, timestamp, specification):
        return saved_decisions(self.row)


def default_full_controls():
    from PIL import Image
    from locatemot.groundjev.tracker import OnlineTracker
    source = subprocess.check_output(['git', 'show', PARENT_TRACKER_BLOB], cwd=ROOT, text=True)
    module = ModuleType('groundjev_parent_tracker_runtime_check')
    sys.modules[module.__name__] = module
    exec(compile(source, PARENT_TRACKER_BLOB, 'exec'), module.__dict__)
    root = ROOT/'outputs/groundjev/training/onejev_full1203_controls_v2_wait_fix'
    proof = json.loads((root/'completed_units_validated_summary.json').read_text())
    plan = json.loads((ROOT/'configs/groundjev/onejev_full1203_controls_v1.json').read_text())
    scope = plan['scopes']['mot_full1203']
    manifest = json.loads((ROOT/scope['manifest']).read_text())
    report = {}
    with ExitStack() as stack:
        inputs = stack.enter_context((ROOT/scope['candidates']).open())
        traces, trackers = {}, {}
        for name, result in proof['results'].items():
            traces[name] = stack.enter_context((Path(result['run'])/'frame_decisions.jsonl').open())
            decision = RecordedDecision()
            trackers[name] = decision, module.OnlineTracker(decision), OnlineTracker(decision)
            report[name] = {'frames': 0, 'candidates': 0, 'rejected_frames': 0}
        for number, path in zip(manifest['frame_numbers'], manifest['images']):
            before = json.loads(next(inputs)); assert before['frame'] == number and before['image_path'] == path
            candidates = before['candidates']
            geometry = [{k: v for k, v in c.items() if k not in ('native_region_feature', 'temporal_region_feature')} for c in candidates]
            with Image.open(path) as raw:
                image = raw.convert('RGB')
            for name, trace in traces.items():
                row = json.loads(next(trace)); assert row['frame'] == number and row['candidates'] == geometry
                decision, parent, current = trackers[name]; decision.row = row
                a = parent.step(image, candidates, number/manifest['fps'], {'task': 'mot', 'vocabulary': ['person'], 'expression': None})
                b = current.step(image, candidates, number/manifest['fps'], {'task': 'mot', 'vocabulary': ['person'], 'expression': None})
                assert a == b == (row['outputs'], row['state'])
                report[name]['frames'] += 1; report[name]['candidates'] += len(candidates)
                report[name]['rejected_frames'] += bool(row['perception'].get('model_response_rejected'))
        assert next(inputs, None) is None
        for name, trace in traces.items():
            assert next(trace, None) is None
            _, parent, current = trackers[name]
            assert parent.state_dict() == current.state_dict()
            assert report[name] == {'frames': 1203, 'candidates': 13664, 'rejected_frames': 27}
    return report


def conditional_fit_boundaries():
    from PIL import Image, ImageChops
    from locatemot.groundjev.assigned_identity_decision import AssignedIdentityUpdateDecision
    from locatemot.groundjev.assigned_identity_update import build_assigned_update_request
    from locatemot.groundjev.category import CategorizedDecision
    from locatemot.groundjev.decision import OneJevDecision
    from locatemot.groundjev.tracker import OnlineTracker
    source = ROOT/'outputs/groundjev/training_data/native600_fit_rollout_refresh_v1/rollout'
    grouped = {}
    for r in map(json.loads, (source/'train.jsonl').open()):
        if r['task'] == 'memory_update':
            match = re.fullmatch(r'(.+)_(\d+)_batch(\d+)_update_(\d+)', r['id'])
            name, frame, batch, local = match.groups()
            if int(local) == 0:
                grouped.setdefault((name, int(frame)), {})[int(batch)] = r
    registry = [json.loads(s) for s in (ROOT/'outputs/groundjev/training_data/assigned_identity_update_component_v1/all_candidate_registry.jsonl').open()]
    unknown = {(r['sequence'], r['frame'], r['candidate']) for r in registry
               if r['eligibility'] in ('unknown_birth_anchor', 'unmatched_not_universal_negative_truth')}
    counts = {'frames': 0, 'single_frame_branches': 0, 'scripted_second_pass_questions': 0,
              'pre_write_engine_calls_checked': 0, 'unknown_existing_assignments_asked_per_branch': 0,
              'training_runtime_canonical_state_question_matches': 0,
              'real_alias_history_crop_pixel_comparisons': 0}
    class ProbeEngine:
        def decide(self, request, media):
            owner = self.owner
            assert owner.tracker.state_dict() == owner.before
            batch = next(b for b in owner.assignment_batches if
                         b['state']['candidates'] == request.state['candidates'] and
                         b['state']['geometry_columns_track_ids'] == request.state['geometry_columns_track_ids'])
            expected = owner.expected[batch['offset']]
            canonical = deepcopy(request.state); aliases = expected['aliases']
            for track in canonical['historical_tracks']:
                track['id'] = aliases[track['id']]
            canonical['geometry_columns_track_ids'] = [aliases[i] for i in canonical['geometry_columns_track_ids']]
            actual_selected = canonical['final_assignment']['selected_existing_identity_by_candidate']
            canonical['final_assignment']['selected_existing_identity_by_candidate'] = {k: aliases[v] for k, v in actual_selected.items()}
            expected_request = build_assigned_update_request(expected['row']['state'],
                {int(k): aliases[v] for k, v in actual_selected.items()})
            assert canonical == expected_request.state
            assert len(media) == len(expected['row']['media'])
            for local, raw_identity in actual_selected.items():
                key = f'assigned_update_{local}'
                text = request.questions[key].instructions.replace(f'identity {raw_identity}?', f'identity {aliases[raw_identity]}?')
                assert text == expected_request.questions[key].instructions
                counts['training_runtime_canonical_state_question_matches'] += 1
                counts['unknown_existing_assignments_asked_per_branch'] += (owner.name, owner.frame, batch['offset']+int(local)) in unknown
            counts['scripted_second_pass_questions'] += len(request.questions)
            counts['pre_write_engine_calls_checked'] += 1
            return SimpleNamespace(answers={k: SimpleNamespace(noul=owner.gate) for k in request.questions}), {
                'functional_probe': True, 'scripted_Noul_answer': owner.gate, 'not_a_model_prediction': True}

    class RecordedAssociation(AssignedIdentityUpdateDecision):
        def __init__(self, name, frame, row, gate):
            self.name, self.frame, self.row, self.gate = name, frame, row, gate
            self.evidence_format, self.rmot_membership = 'v3_crop_motion', False
            self.engine = ProbeEngine(); self.engine.owner = self

        def decide(self, image, candidates, tracks, timestamp, specification):
            self.assignment_batches, self.expected = [], {}
            by_id = {t.track_id: t for t in tracks}
            events = {e['candidate']: e for e in self.row['state']['events'] if 'candidate' in e}
            offset = 0
            for _, old in sorted(grouped[self.name, self.frame].items()):
                group = candidates[offset:offset+len(old['state']['candidates'])]
                ranked = [by_id[i] for i in events[offset]['decision']['shortlisted_ids']]
                aliases = dict(zip([t.track_id for t in ranked], [t['id'] for t in old['state']['historical_tracks']]))
                self.accepted_batch(image, group, ranked, timestamp, specification)
                self.expected[offset] = {'row': old, 'aliases': aliases}
                if ranked:
                    shadows = []
                    for track in ranked:
                        shadow = copy(track); shadow.track_id = aliases[track.track_id]; shadows.append(shadow)
                    actual = OneJevDecision._history_sheet(self, shadows)
                    with Image.open(ROOT/old['media'][-1]['path']) as raw:
                        saved = raw.convert('RGB')
                    assert actual.size == saved.size and ImageChops.difference(actual, saved).getbbox() is None
                    counts['real_alias_history_crop_pixel_comparisons'] += 1
                offset += len(group)
            assert offset == len(candidates)
            return saved_decisions(self.row)

    previous = None
    for row in map(json.loads, (source/'native_rollout.jsonl').open()):
        name, frame = row['sequence'], row['frame']
        if name != previous:
            saved = RecordedDecision(); baseline = OnlineTracker(saved); previous = name
        saved.row = row
        before = deepcopy(baseline.state_dict())
        with Image.open(row['image_path']) as raw:
            image = raw.convert('RGB')
        fps = 30. if name == 'MOT17-02-FRCNN' else 20. if name == 'dancetrack0001' else 25.
        spec = {'task': 'mot', 'vocabulary': ['person'], 'expression': None}
        for gate in (0., 1.):
            decision = RecordedAssociation(name, frame, row, gate)
            tracker = OnlineTracker(decision); tracker.load_state_dict(deepcopy(before))
            decision.tracker, decision.before = tracker, deepcopy(tracker.state_dict())
            wrapper = CategorizedDecision(decision, None)
            assert wrapper.decide_assigned_updates == decision.decide_assigned_updates
            outputs, state = tracker.step(image, row['candidates'], frame/fps, spec)
            assert outputs == row['outputs']
            old_tracks = {t['track_id']: t for t in before['tracks']}
            new_tracks = {t['track_id']: t for t in tracker.state_dict()['tracks']}
            for event in state['events']:
                if 'candidate' not in event or event.get('track_id') not in old_tracks:
                    continue
                assert event['memory_action'] == ('UPDATE' if gate else 'HOLD')
                tid = event['track_id']
                assert old_tracks[tid]['anchor_image'] == new_tracks[tid]['anchor_image']
                if gate:
                    assert new_tracks[tid]['long_image'] == new_tracks[tid]['recent_image']
                    assert new_tracks[tid]['last_reliable_timestamp'] == frame/fps
                else:
                    assert new_tracks[tid]['long_image'] == old_tracks[tid]['long_image']
                    assert new_tracks[tid]['last_reliable_timestamp'] == old_tracks[tid]['last_reliable_timestamp']
            counts['single_frame_branches'] += 1
        outputs, state = baseline.step(image, row['candidates'], frame/fps, spec)
        assert (outputs, state) == (row['outputs'], row['state'])
        counts['frames'] += 1
    assert counts['frames'] == 48 and counts['single_frame_branches'] == 96
    assert counts['scripted_second_pass_questions'] == counts['training_runtime_canonical_state_question_matches'] == 1316
    assert counts['unknown_existing_assignments_asked_per_branch'] == 90
    return counts


def main():
    import torch
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'RUNNING_ACTUAL_DEFAULT_AND_SCRIPTED_CONDITIONAL_CPU_CHECKS',
             'new_GPU_or_native_model_loaded': False, 'new_tracking_benchmark': False}
    def save():
        (args.out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    save()
    try:
        state['default_three_arm_frozen_decision_replay'] = default_full_controls(); save()
        state['scripted_conditional_real_fit_execution'] = conditional_fit_boundaries()
        assert not torch.cuda.is_initialized()
        state.update(status='DEFAULT_THREE_ARM_EXACT_AND_REAL_FIT_CONDITIONAL_BOUNDARY_CHECKS_PASS',
            parent_tracker_blob=PARENT_TRACKER_BLOB,
            limitations='Frozen first-pass model decisions; second-pass answers are explicitly scripted 0/1 single-frame branches from the original prefix. No fresh model prediction, causal multi-frame conditional-policy rollout, model accuracy, E2E cost or trained benefit established. Parent/current final memory matches each other; visible-track lifecycle probabilities not saved in traces are not reconstructed as original full-memory truth.')
        save()
    except Exception as error:
        state.update(status='ASSIGNED_ID_RUNTIME_CPU_CHECK_FAILED', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
