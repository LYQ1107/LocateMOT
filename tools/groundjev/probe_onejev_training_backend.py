"""Isolated real multimodal training probe; production entry points are unchanged."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/OneJev')]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['reference', 'fla'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'fla': sys.path.insert(0, str(ROOT/'.tools/onejev-fla052'))
    import torch
    from tools.groundjev.run_group_reference_sft import configure_group_reference_sft
    from train import common
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    output = Path(config['runs_dir'])/config['name']
    calls = 0
    if args.mode == 'fla':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        def training_chunk(*values, **kwargs):
            nonlocal calls
            assert not kwargs.get('output_final_state', False), 'This probe is training-only with use_cache=False'
            calls += 1
            return chunk_gated_delta_rule(*values, **kwargs)
        modeling.torch_chunk_gated_delta_rule = training_chunk
    native_bundle, native_clip = common.load_bundle, torch.nn.utils.clip_grad_norm_
    model = None
    captured = False

    def load_bundle(*values, **kwargs):
        nonlocal model
        bundle = native_bundle(*values, **kwargs)
        if bundle.model is not None: model = bundle.model
        return bundle

    def capture_gradients(parameters, *values, **kwargs):
        nonlocal captured
        parameters = list(parameters)
        if not captured:
            names = {id(p): name for name, p in model.named_parameters()}
            samples, shapes, norms = {}, {}, {}
            for parameter in parameters:
                name = names[id(parameter)]; gradient = parameter.grad.detach().float()
                assert torch.isfinite(gradient).all()
                flat = gradient.reshape(-1); stride = max(1, (flat.numel()+4095)//4096)
                samples[name] = flat[::stride].cpu().clone()
                shapes[name] = {'shape': list(gradient.shape), 'elements': flat.numel(), 'sample_stride': stride}
                norms[name] = float(gradient.norm())
            assert len(samples) == 320
            torch.save({'samples': samples, 'shapes': shapes, 'full_parameter_gradient_norms': norms}, output/'first_gradient_samples.pt')
            captured = True
        return native_clip(parameters, *values, **kwargs)

    common.load_bundle = load_bundle
    torch.nn.utils.clip_grad_norm_ = capture_gradients
    stateful = configure_group_reference_sft()
    sys.argv = [str(Path(__file__).resolve()), '--config', str(args.config), '--paired-index-shuffle',
        '--initial-training-state', str(ROOT/'outputs/groundjev/training/onejev_unified600_native_v2/latest_training_state.pt'),
        '--expected-initial-step', '600']
    code = stateful.main()
    assert code == 0 and captured and (calls > 0 if args.mode == 'fla' else calls == 0)
    report = {'mode': args.mode, 'actual_optimizer_steps': 4, 'actual_examples': 20,
              'FLA_chunk_training_calls': calls, 'sampled_gradient_bytes': (output/'first_gradient_samples.pt').stat().st_size,
              'all320_full_gradients_finite': True, 'gradient_sampling': 'deterministic uniform stride, at most4096 values per parameter before native clipping',
              'inference_backend_adopted': False, 'production_training_changed': False, 'full_goal_complete': False}
    (output/'backend_probe_receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
