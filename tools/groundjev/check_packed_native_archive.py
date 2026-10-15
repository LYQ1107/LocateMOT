"""Real21-frame exact archive/replay check, including an actual rejected empty frame."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tools.groundjev.run_video import ReplayFrames
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1/broad_native_training/checkpoint.pt'
    source = ROOT/'outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/compact_distilled/frame_decisions.jsonl'
    manifest = json.loads((ROOT/'outputs/groundjev/pilot_data/dancetrack.json').read_text())
    manifest.update(images=manifest['images'][:21], frame_numbers=manifest['frame_numbers'][:21],
        scope='explicit_packed_archive_functional_check_not_full_video')
    manifest_path = args.out/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT/'external/OneJev'),
        OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    traces, reports = {}, {}
    for name in ('reference', 'packed_record', 'packed_replay'):
        replay = args.out/'packed_record/frame_candidates.jsonl' if name == 'packed_replay' else source
        command = [str(ROOT/'.venv-onejev-py310/bin/python'), 'tools/groundjev/run_video.py', '--task', 'mot',
            '--video', str(manifest_path), '--backend', 'compact', '--compact-device', 'cpu', '--checkpoint', str(checkpoint),
            '--candidate-replay', str(replay), '--omit-replay-feature-vectors', '--memory-checkpoint-format', 'binary',
            '--memory-export-interval', '0', '--out', str(args.out/name)]
        if name == 'packed_record':
            command += ['--record-native-candidates', '--native-candidate-format', 'packed_npz_f32',
                '--native-candidate-byte-budget', str(32*2**20)]
        with (args.out/(name+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        traces[name] = [json.loads(line) for line in (args.out/name/'frame_decisions.jsonl').open()]
        reports[name] = json.loads((args.out/name/'result.json').read_text())
        assert len(traces[name]) == 21 and reports[name]['end_to_end_fps'] is None
        assert reports[name]['perception_rejected_frames'] == [21]
    for name in ('packed_record', 'packed_replay'):
        assert all(a['outputs'] == b['outputs'] and a['state'] == b['state']
            for a, b in zip(traces['reference'], traces[name]))
    original, packed = ReplayFrames(source), ReplayFrames(args.out/'packed_record/frame_candidates.jsonl')
    candidates, raw_bytes = 0, 0
    for row in traces['packed_record']:
        a, b = original.frame(row['frame']), packed.frame(row['frame'])
        assert b == {'frame': row['frame'], 'image_path': row['image_path'],
            'candidates': a['candidates'], 'perception': row['perception']}
        candidates += len(a['candidates'])
        raw_bytes += len((json.dumps(b)+'\n').encode())
    next_frame = original.frame(22)
    original.close(); packed.close()
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactDecision
    from locatemot.groundjev.tracker import OnlineTracker
    torch.set_num_threads(4)
    states = [torch.load(args.out/name/'final_identity_memory.pt', map_location='cpu', weights_only=True)
        for name in ('reference', 'packed_replay')]
    assert states[0] == states[1]
    actor = CompactDecision(checkpoint, device='cpu')
    trackers = [OnlineTracker(actor), OnlineTracker(actor)]
    for tracker, state in zip(trackers, states):tracker.load_state_dict(state)
    with Image.open(next_frame['image_path']) as raw:image = raw.convert('RGB')
    spec = {'task': 'mot', 'vocabulary': ['person'], 'expression': None}
    assert trackers[0].step(image, next_frame['candidates'], 22/20, spec) == trackers[1].step(image, next_frame['candidates'], 22/20, spec)
    total = (args.out/'packed_record/frame_candidates.jsonl').stat().st_size
    total += sum(p.stat().st_size for p in (args.out/'packed_record/native_region_features').glob('*.npz'))
    assert total == reports['packed_record']['native_candidate_archive_bytes'] and total <= 32*2**20
    assert reports['packed_record']['native_candidate_archive_feature_files'] == 21
    result = {'status': 'REAL21_PACKED_NATIVE_ROI_ARCHIVE_REPLAY_AND_NEXT_FRAME_EXACT',
        'actual_frames': 21, 'native_candidates': candidates, 'actual_rejected_empty_frames_retained': [21],
        'all_candidate_fields_native_vectors_temporal_vectors_and_archived_metadata_exact': True,
        'all21_predictions_states_full_memory_and_next_real_frame22_exact': True,
        'FP32_inputs_lossless_not_quantized': True, 'raw_JSON_equivalent_bytes': raw_bytes,
        'packed_total_index_and_feature_bytes': total, 'new_feature_cap_bytes': 32*2**20,
        'GT_or_future_used_by_inference': False, 'fresh_DLM_E2E_or_accuracy_claim': False}
    (args.out/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
