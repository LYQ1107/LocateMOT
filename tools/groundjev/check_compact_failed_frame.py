"""Fresh-context regression of the actual failed frame and saved causal memory."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image
    from locatemot.groundjev.compact import CompactCore, CompactDecision
    from locatemot.groundjev.tracker import OnlineTracker
    base = ROOT/'outputs/groundjev/compact/rmot_native72_stage_v1'
    failed = base/'broad_native_full1203/joint1200'
    checkpoint = base/'broad_native_training/checkpoint.pt'
    failure = json.loads((failed/'failure.json').read_text())
    memory = json.loads((failed/'latest_identity_memory.json').read_text())
    assert failure['frame'] == 799 and memory['previous_timestamp'] == 798/20
    source = ROOT/'outputs/groundjev/compact/dancetrack0004_native300_v2_empty_roi_fixed/compact_distilled/frame_decisions.jsonl'
    with source.open() as rows:
        row = next(r for r in map(json.loads, rows) if r['frame'] == 799)
    decision = CompactDecision(checkpoint)
    tracker = OnlineTracker(decision); tracker.load_state_dict(memory)
    snapshots = []
    handle = decision.model.register_forward_pre_hook(lambda module, args: snapshots.append(
        {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in args[0].items()}))
    with Image.open(row['image_path']) as raw:
        image = raw.convert('RGB')
    outputs, state = tracker.step(image, row['candidates'], 799/20,
        {'task': 'mot', 'vocabulary': ['person'], 'expression': None})
    torch.cuda.synchronize(); handle.remove()
    cpu = CompactCore(semantic=True)
    cpu.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']); cpu.eval()
    difference = 0.
    with torch.no_grad():
        for e in snapshots:
            gpu_output = decision.model({k: v.cuda() if torch.is_tensor(v) else v for k, v in e.items()})
            cpu_output = cpu(e)
            for key in gpu_output:
                assert torch.isfinite(gpu_output[key]).all() and torch.isfinite(cpu_output[key]).all()
            difference = max(difference, float((gpu_output['match'].softmax(-1).cpu()-cpu_output['match'].softmax(-1)).abs().max()))
    receipt = {'status': 'ACTUAL_FAILED_FRAME_FRESH_CUDA_CONTEXT_PASSED', 'frame': 799,
        'saved_memory_prefix': 798, 'native_candidates': len(row['candidates']),
        'actual_output_tracks': len(outputs), 'all_head_outputs_finite': True,
        'GPU_CPU_match_probability_max_difference': difference, 'GT_read': False,
        'GPU_reset_or_driver_change': False, 'CUDA_LAUNCH_BLOCKING': True,
        'previous_async_launch_failure_external_root_cause': 'not established from accessible driver logs'}
    (base/'failed_frame799_regression.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
