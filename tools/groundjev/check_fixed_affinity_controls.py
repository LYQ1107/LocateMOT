"""Real-frame semantic/auxiliary preservation and lossless checkpoint next-frame check."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactDecision, CompactAffinityControl
    from locatemot.groundjev.decision import SimpleDecision, FeatureDecision
    from locatemot.groundjev.tracker import OnlineTracker
    from tools.groundjev.run_video import ReplayFrames
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt'
    source = ROOT/'outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/compact_distilled/frame_decisions.jsonl'
    replay = ReplayFrames(source)
    rows = [replay.frame(i) for i in range(1, 6)]
    replay.close()
    learned = CompactDecision(checkpoint, device='cpu')
    tracker = OnlineTracker(SimpleDecision())
    spec = {'task': 'mot', 'vocabulary': ['person'], 'expression': None}
    with Image.open(rows[0]['image_path']) as raw:
        image = raw.convert('RGB')
    tracker.step(image, rows[0]['candidates'], 1/20, spec)
    tracks = list(tracker.tracks.values())
    with Image.open(rows[1]['image_path']) as raw:
        image = raw.convert('RGB')
    preserved = {}
    for task in ('mot', 'rmot', 'ovmot'):
        current = {'task': task, 'vocabulary': ['person', 'car'] if task == 'ovmot' else ['person'],
                   'expression': 'person' if task == 'rmot' else None}
        original, life, _ = learned.decide(image, rows[1]['candidates'], tracks, 2/20, current)
        for association in ('fixed_iou', 'fixed_iou_cosine'):
            control = CompactAffinityControl(learned, association, 'learned')
            altered, next_life, meta = control.decide(image, rows[1]['candidates'], tracks, 2/20, current)
            assert next_life == life and not meta['native_probabilities']
            assert all({k: v for k, v in a.items() if k != 'probabilities'} ==
                       {k: v for k, v in b.items() if k != 'probabilities'} for a, b in zip(original, altered))
        fixed, fixed_life, meta = CompactAffinityControl(learned, 'learned', 'fixed').decide(
            image, rows[1]['candidates'], tracks, 2/20, current)
        assert not fixed_life and meta['native_probabilities']
        assert all(a['probabilities'] == b['probabilities'] and b['update_probability'] == 1.
                   for a, b in zip(original, fixed))
        preserved[task] = True
    fixed, _, _ = CompactAffinityControl(learned, 'fixed_iou_cosine', 'fixed').decide(
        image, rows[1]['candidates'], tracks, 2/20, spec)
    previous, _, _ = FeatureDecision().decide(image, rows[1]['candidates'], tracks, 2/20, spec)
    maximum = max(abs(a['probabilities'][k]-b['probabilities'][k]) for a, b in zip(fixed, previous)
                  for k in a['probabilities'])
    assert maximum < 1e-12
    manifest = json.loads((ROOT/'outputs/groundjev/pilot_data/dancetrack.json').read_text())
    manifest.update(images=manifest['images'][:4], frame_numbers=manifest['frame_numbers'][:4],
                    scope='explicit_affinity_storage_functional_check_only')
    manifest_path = args.out/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'))
    for name in ('json_reference', 'binary_final'):
        command = [sys.executable, 'tools/groundjev/run_video.py', '--task', 'mot', '--video', str(manifest_path),
            '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint', str(checkpoint),
            '--candidate-replay', str(source), '--omit-replay-feature-vectors', '--out', str(args.out/name)]
        if name == 'binary_final':
            command += ['--memory-checkpoint-format', 'binary', '--memory-export-interval', '0']
        with (args.out/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    a = [json.loads(line) for line in (args.out/'json_reference/frame_decisions.jsonl').open()]
    b = [json.loads(line) for line in (args.out/'binary_final/frame_decisions.jsonl').open()]
    assert len(a) == len(b) == 4
    assert all(x['outputs'] == y['outputs'] and x['state'] == y['state'] for x, y in zip(a, b))
    first = json.loads((args.out/'json_reference/final_identity_memory.json').read_text())
    second = torch.load(args.out/'binary_final/final_identity_memory.pt', map_location='cpu', weights_only=True)
    assert first == second
    left, right = OnlineTracker(learned), OnlineTracker(learned)
    left.load_state_dict(first)
    right.load_state_dict(second)
    with Image.open(rows[4]['image_path']) as raw:
        image = raw.convert('RGB')
    assert left.step(image, rows[4]['candidates'], 5/20, spec) == right.step(image, rows[4]['candidates'], 5/20, spec)
    for name in ('json_reference', 'binary_final'):
        report = json.loads((args.out/name/'result.json').read_text())
        assert report['end_to_end_fps'] is None and report['final_full_memory_export_in_pipeline_timer']
    result = {'status': 'REAL_AFFINITY_SEMANTICS_AUXILIARY_AND_LOSSLESS_MEMORY_VALIDATED',
        'same_semantic_and_auxiliary_heads_on_same_state_all_three_tasks': preserved,
        'fixed_score_not_native_probability': True, 'same_memory_vectors_feature_reference_max_difference': maximum,
        'all4_actual_frame_predictions_and_states_exact_across_storage': True,
        'complete_serialized_state_fields_and_next_actual_frame_exact': True,
        'GT_or_future_used': False, 'native_fresh_E2E_or_accuracy_claim': False}
    (args.out/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
