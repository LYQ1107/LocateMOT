"""Real complete500 master initialization and all exported tensor changes on CPU."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    import torch
    from safetensors import safe_open
    from train import common, sft
    from tools.groundjev.onejev_resumable_sft import initialize_fp32_stage, protocol
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    training = ROOT/'outputs/groundjev/training/onejev_own_history_v3_exact500_replay_stateful'
    owner = json.loads((training/'owner_state.json').read_text())
    assert owner['status'] == 'STATEFUL_ONEJEV500_TRAINING_COMPLETE' and owner['done']['steps'] == 500
    checkpoint = training/'ckpt-step500'
    initial = ROOT/'outputs/groundjev/training/onejev_teacher_check_100/final'
    changed, frozen, total = {}, [], 0
    with safe_open(initial/'model.safetensors', framework='pt') as before, safe_open(checkpoint/'model.safetensors', framework='pt') as after:
        assert set(before.keys()) == set(after.keys())
        for key in before.keys():
            a, b = before.get_tensor(key), after.get_tensor(key)
            assert a.shape == b.shape and a.dtype == b.dtype
            difference = int((a != b).sum())
            changed[key] = difference; total += a.numel()
            if '.visual.' in '.'+key or key.startswith('visual'):
                assert difference == 0
                frozen.append(key)
    assert sum(changed.values()) > 0 and frozen
    state = torch.load(training/'latest_training_state.pt', map_location='cpu', weights_only=True)
    original_args = sft.parse_args(['--config', 'configs/groundjev/onejev_own_history_v3_exact500_replay_stateful.json'])
    assert state['protocol'] == protocol(original_args)
    assert state['step'] == 500 and state['seen'] == 2000
    actual_master_keys = len(state['fp32_trainable_parameters'])
    del state; gc.collect()
    bundle = common.load_bundle(str(checkpoint), device='cpu', dtype='float32', train=True, multimodal=True)
    for name, value in bundle.model.named_parameters():
        value.requires_grad = not ('.visual.' in '.'+name or name.startswith('visual'))
    args = sft.parse_args(['--config', 'configs/groundjev/onejev_unified600_teacher_v2.json'])
    args.runs_dir = str(options.out.parent); args.name = options.out.name
    initialization = initialize_fp32_stage(training/'latest_training_state.pt', bundle.model, args, 500)
    assert initialization['FP32_parameter_entries'] == actual_master_keys == 320
    assert initialization['parameters'] == 752393024
    receipt = {'status': 'ACTUAL_COMPLETE500_FP32_STAGE_INITIALIZATION_AND_FROZEN_VISION_VERIFIED',
        'source_steps': 500, 'source_examples': 2000, 'exported_tensor_entries': len(changed),
        'exported_elements': total, 'changed_BF16_export_elements_vs_initial100': sum(changed.values()),
        'changed_export_tensors': sum(x > 0 for x in changed.values()),
        'unchanged_frozen_visual_tensors': len(frozen), 'all_changed_elements_by_tensor': changed,
        'legacy_default_checkpoint_protocol_exactly_unchanged': True,
        'initialization': initialization, 'GPU_used': False, 'forward_or_new_training_performed': False,
        'original_state_or_weights_modified': False,
        'exact_old_optimizer_trajectory_resume_claim': False}
    (options.out/'validation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'all_changed_elements_by_tensor'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
