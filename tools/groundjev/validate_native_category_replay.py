"""Native category parity and two continuous full-frame causal-memory replay."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


class IdentityTape:
    """Use the saved identity decisions solely for a paired CPU regression."""
    def __init__(self, strip_categories=False):
        self.strip_categories = strip_categories

    def decide(self, image, candidates, tracks, timestamp, specification):
        decisions, lifecycle = [None]*len(candidates), {}
        for event in self.row['state']['events']:
            if 'candidate' in event:
                decision = deepcopy(event['decision'])
                if self.strip_categories:
                    decision.pop('category_decision')
                decisions[event['candidate']] = decision
            if event.get('lifecycle_decision') is not None:
                lifecycle[event['track_id']] = deepcopy(event['lifecycle_decision'])
        assert all(d is not None for d in decisions)
        return decisions, lifecycle, deepcopy(self.row['state']['decision_metadata'])


def probability_difference(a, b):
    assert a.keys() == b.keys()
    differences = []
    for key in a:
        if isinstance(a[key], dict):
            differences.append(probability_difference(a[key], b[key]))
        elif isinstance(a[key], (float, int)) and not isinstance(a[key], bool):
            differences.append(abs(a[key]-b[key]))
        else:
            assert a[key] == b[key], (key, a[key], b[key])
    return max(differences, default=0.)


def main():
    from PIL import Image
    import torch
    from locatemot.groundjev.category import CategorizedDecision, NativeCategoryHead
    from locatemot.groundjev.category_replay import CausalNativeCategoryReplay, categories_from_frame
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import ReplayFrames

    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['combined_registered_feature_upper_bytes'] < plan['global_temporary_feature_budget_bytes']
    output = ROOT/plan['output']; output.mkdir(parents=True, exist_ok=False)
    def save(name, report):
        (output/name).write_text(json.dumps(report, indent=2)+'\n')
    save('registration.json', plan)
    source = ROOT/plan['source_run']; checkpoint = ROOT/plan['checkpoint']
    controls = json.loads((ROOT/plan['controls']).read_text())
    job = controls['scopes']['tao_raw_full40']
    manifest = json.loads((ROOT/job['manifest']).read_text())
    rows = []
    with (source/'frame_decisions.jsonl').open() as stream:
        for _ in range(plan['continuous_CPU_frames']):
            rows.append(json.loads(next(stream)))
    assert [r['frame'] for r in rows] == manifest['frame_numbers'][:len(rows)]
    originals = ReplayFrames(ROOT/job['candidates'])
    try:
        native_rows = [originals.frame(r['frame']) for r in rows]
    finally:
        originals.close()
    for row, original in zip(rows, native_rows):
        assert row['image_path'] == original['image_path']
        assert row['candidates'] == [{k: v for k, v in c.items()
            if k not in ('native_region_feature', 'temporal_region_feature')} for c in original['candidates']]

    print('WAITING_SHARED_GPU4_FREE_MEMORY_FOR_NATIVE_CATEGORY_PARITY', flush=True)
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(plan['physical_gpu']),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free >= plan['launch_min_free_MiB']:
            break
        time.sleep(20)
    from qev.mm_engine import MMDecisionEngine
    engine = MMDecisionEngine(str(checkpoint), device='cuda:0', dtype='bfloat16', cuda_graphs=False)
    head = NativeCategoryHead(engine)
    count = plan['native_recomputed_candidates']
    assert count == head.batch_candidates
    first = rows[0]
    recorded, _ = categories_from_frame(first, native_rows[0]['candidates'], checkpoint, manifest['vocabulary'])
    with Image.open(first['image_path']) as image:
        computed, report = head.classify(image.convert('RGB'), native_rows[0]['candidates'][:count], manifest['vocabulary'])
    difference = max(probability_difference(a, b) for a, b in zip(computed, recorded[:count]))
    native_report = dict(native_candidates=count, vocabulary_size=len(manifest['vocabulary']),
        max_probability_or_confidence_difference=difference,
        preregistered_gate=plan['probability_max_difference_gate'],
        choices_labels_and_all_recorded_fields_same=difference == 0,
        native_report=report, launch_free_memory_MiB=free)
    save('native_parity.json', native_report)
    assert len(computed) == count and difference <= plan['probability_max_difference_gate'], difference
    del head, engine
    torch.cuda.empty_cache()
    print('NATIVE_CATEGORY_PARITY_PASSED_CPU_FULL_FRAME_TRAJECTORY_STARTING', flush=True)

    reference_tape, consumer_tape = IdentityTape(), IdentityTape(strip_categories=True)
    replay = CausalNativeCategoryReplay(source, checkpoint, manifest, allow_partial_source_for_probe=True)
    reference = OnlineTracker(reference_tape)
    consumer = OnlineTracker(CategorizedDecision(consumer_tape, replay))
    specification = dict(task='ovmot', vocabulary=manifest['vocabulary'], expression=None)
    frame_reports = []
    try:
        for row, original in zip(rows, native_rows):
            reference_tape.row = consumer_tape.row = row
            replay.begin_frame(row['image_path'], row['frame'])
            with Image.open(row['image_path']) as image:
                image = image.convert('RGB')
            args_step = (image, original['candidates'], row['state']['timestamp'], specification)
            original_outputs, original_state = reference.step(*args_step)
            replay_outputs, replay_state = consumer.step(*args_step)
            assert original_outputs == row['outputs'] == replay_outputs
            assert original_state == row['state']
            # Only the diagnostic report changes; all decisions/events/IDs do not.
            replay_state['decision_metadata']['category_decision'] = deepcopy(
                original_state['decision_metadata']['category_decision'])
            assert original_state == replay_state
            assert reference.state_dict() == consumer.state_dict()
            frame_reports.append(dict(frame=row['frame'], candidates=len(original['candidates']),
                outputs=len(original_outputs), persistent_tracks=len(reference.tracks),
                outputs_events_IDs_and_full_numeric_feature_and_image_memory_exact=True))
    finally:
        replay.close()
    result = dict(status='NATIVE_CATEGORY_AND_CONTINUOUS_FULL_FRAME_MEMORY_REPLAY_VALIDATED',
        plan=str(args.plan), native_parity=native_report, CPU_frames=frame_reports,
        category_replay=replay.summary(), checkpoint=str(checkpoint),
        GT_or_future_frames_read=False, new_feature_or_image_archive=False,
        identity_model_rerun=False, identity_source='recorded identity decisions only for paired CPU regression',
        native_probe_is_not_full_video_fresh_model_parity=True,
        existing_experiments_changed=False, end_to_end_speed_or_tracking_gain_claim=False,
        full_goal_complete=False)
    save('result.json', result)
    actual = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    assert actual < plan['diagnostic_reserve_bytes']
    print(json.dumps(dict(status=result['status'], max_probability_difference=difference,
        frames=len(frame_reports), candidates=sum(r['candidates'] for r in frame_reports),
        actual_diagnostic_bytes=actual)), flush=True)


if __name__ == '__main__':
    main()
