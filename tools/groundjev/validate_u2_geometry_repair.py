"""Reproduce the real B2 failure and require the entire198-frame prefix unchanged."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from locatemot.groundjev.u2_bridge import GroundU2Bridge, candidate_tensors
    from locatemot.unified.runtime.u2_online_tracker import U2OnlineTracker
    from tools.groundjev.run_video import ReplayFrames
    root = ROOT/'outputs/groundjev/baselines/u2_core300_cpu_v1'
    torch.set_num_threads(4)
    model = GroundU2Bridge().eval()
    model.load_state_dict(torch.load(root/'training/checkpoint.pt', weights_only=True)['state_dict'])
    tracker = U2OnlineTracker(model.core)
    source = ReplayFrames(ROOT/'outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/compact_distilled/frame_decisions.jsonl')
    old = [json.loads(line) for line in (root/'full1203/frame_decisions.jsonl').open()]
    assert len(old) == 198
    maximum = 0.
    for previous in old:
        row = source.frame(previous['frame'])
        features, boxes, scores = candidate_tensors(row['candidates'])
        with torch.no_grad():
            output = tracker.step(model.candidates(features, boxes, scores), frame_index=previous['frame'])
        for key in ('pair_logits', 'new_logits', 'no_match_logits', 'alive_logits'):
            actual = output[key]
            expected = torch.tensor(previous['native_U2'][key]).reshape(actual.shape)
            difference = float((actual-expected).abs().max()) if actual.numel() else 0.
            maximum = max(maximum, difference)
            assert difference == 0, (previous['frame'], key, difference)
        assert tracker.state.track_ids.tolist() == previous['state']['persistent_ids']
        assert output['track_ids'].tolist() == [item['track_id'] for item in previous['outputs']]
    tracker.save_state(root/'validated_identity_state198.pt')
    features, boxes, scores = candidate_tensors(source.frame(199)['candidates'])
    with torch.no_grad():
        output = tracker.step(model.candidates(features, boxes, scores), frame_index=199)
    assert all(bool(torch.isfinite(output[key]).all())
               for key in ('pair_logits', 'new_logits', 'no_match_logits', 'alive_logits'))
    result = {'all198_prefix_decisions_exact': True, 'prefix_native_U2_logit_max_difference': maximum,
        'failed_frame199_now_finite': True, 'native_proposals_or_model_weights_or_thresholds_changed': False,
        'overflow_root': 'finite coordinates around7.6e19 overflow FP32 squared-sum norm; inf/inf center-distance produces NaN',
        'repair': 'only undefined FP32 inf_over_inf center distances recomputedFP64; every defined FP32 distance preserved',
        'resume_source_state': 'validated_identity_state198.pt', 'GT_used': False}
    (root/'numeric_repair_regression.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    source.close()


if __name__ == '__main__':
    main()
