"""Finite native DLM projector adaptation with dense/referring/identity losses."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'external/GroundAnything'), str(ROOT/'.tools/ground-train')]
os.environ['GAM_DLM_FLEX_CUDAGRAPHS'] = '0'


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch._functorch import config as functorch_config
    from PIL import Image
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from models.dlm.vlm import GAMQwen3DLM
    from train.dlm.data import DLMDataCollator
    from locatemot.groundjev.native_features import NativeGroundFeatures
    from tools.groundjev.native_ground_training_data import build_pairs

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/groundjev/ground_adaptation/native_tracking200_v1')
    parser.add_argument('--resume-state', type=Path)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/groundjev/native_ground_tracking200_20261014.json')
    parser.add_argument('--debug-kernel-sync', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--static-flex-shapes', action='store_true')
    parser.add_argument('--trace-flex-boundary', action='store_true')
    parser.add_argument('--initial-projector-state', type=Path)
    parser.add_argument('--verify-flex-reference', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert not (args.resume_state and args.initial_projector_state)
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    state = {'status': 'PREPARING_REGISTERED_NATIVE_DLM_TRACKING200', 'steps': 0,
        'target_steps': plan['steps'], 'physical_gpu': int(os.environ['CUDA_VISIBLE_DEVICES']), 'seed': plan['seed'],
        'native_language_and_vision_frozen': True, 'formal_adaptation_and_retention_complete': False}
    import importlib.metadata
    state['execution_runtime']={k:importlib.metadata.version(k) for k in ('torch','transformers','triton')}
    state['actual_checkpoint_interval']=args.checkpoint_every
    def save():
        (out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({k: v for k, v in state.items() if k not in ('training_log', 'gradient_probes')}), flush=True)
    save()
    try:
        if args.static_flex_shapes:
            from models.dlm import hybrid, vlm
            # Keep the original native operator, score modifier and block mask;
            # specialize only the compiler guards for the observed shape bugs.
            torch._dynamo.config.recompile_limit = 256
            torch._dynamo.config.accumulated_recompile_limit = 512
            operator = torch.compile(hybrid._flex_attention_kernel, dynamic=False, fullgraph=True,
                options={'triton.cudagraphs': False, 'triton.cudagraph_trees': False})
            hybrid._compiled_flex_attention = operator
            vlm._compiled_flex_attention = operator
            state['native_attention_compile_policy'] = 'same original native FlexAttention kernel/masks; concrete shapes, CUDA graphs disabled'
        if args.trace_flex_boundary:
            from models.dlm import hybrid, vlm
            compiled = vlm._compiled_flex_attention
            def traced_flex(query, key, value, block_mask, _compiled=compiled):
                boundary = {'step': state['steps']+1,
                    'q': {'shape': list(query.shape), 'stride': list(query.stride()), 'dtype': str(query.dtype)},
                    'k': {'shape': list(key.shape), 'stride': list(key.stride()), 'dtype': str(key.dtype)},
                    'v': {'shape': list(value.shape), 'stride': list(value.stride()), 'dtype': str(value.dtype)},
                    'block_mask_lengths': list(block_mask.seq_lengths)}
                (out/'last_flex_boundary.json').write_text(json.dumps(boundary, indent=2)+'\n')
                result = _compiled(query, key, value, block_mask)
                torch.cuda.synchronize()
                return result
            hybrid._compiled_flex_attention = traced_flex
            vlm._compiled_flex_attention = traced_flex
            state['trace_flex_boundary'] = True
        if args.verify_flex_reference:
            from models.dlm import hybrid, vlm
            from torch.nn.attention.flex_attention import flex_attention
            compiled = vlm._compiled_flex_attention
            verified = set()
            def verified_flex(query, key, value, block_mask, _compiled=compiled):
                category = state['current_supervision']
                if category not in verified:
                    # Keep this detached diagnostic graph outside native activation-checkpoint hooks.
                    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, lambda tensor: tensor):
                        inputs = [x.detach().clone().requires_grad_(True) for x in (query, key, value)]
                        actual = _compiled(*inputs, block_mask)
                        reference = flex_attention(*inputs, block_mask=block_mask, enable_gqa=True)
                        torch.testing.assert_close(actual, reference, rtol=.02, atol=.02)
                        a_grad = torch.autograd.grad(actual.float().sum()/actual.numel(), inputs)
                        r_grad = torch.autograd.grad(reference.float().sum()/reference.numel(), inputs)
                        relative = []
                        for a, r in zip(a_grad, r_grad):
                            delta = float((a.float()-r.float()).norm())
                            scale = float(r.float().norm())
                            ratio = delta/max(scale, 1e-12)
                            assert ratio < .05, (category, ratio)
                            relative.append(ratio)
                        verified.add(category)
                        state.setdefault('actual_native_flex_reference_checks', {})[category] = {
                            'q_shape': list(query.shape), 'kv_shape': list(key.shape),
                            'maximum_output_difference': float((actual.float()-reference.float()).abs().max()),
                            'QKV_gradient_relative_L2_difference': relative,
                            'scope': 'first real native layer of this supervision kind, not all steps/layers',
                            'no_RNG_consumed': True}
                        (out/'flex_reference_checks.json').write_text(json.dumps(state['actual_native_flex_reference_checks'], indent=2)+'\n')
                        del inputs, actual, reference, a_grad, r_grad
                return _compiled(query, key, value, block_mask)
            hybrid._compiled_flex_attention = verified_flex
            vlm._compiled_flex_attention = verified_flex
        if args.debug_kernel_sync:
            original_flex = GAMQwen3DLM._flex_qwen_attention
            def synchronized_flex(*values, **kwargs):
                result = original_flex(*values, **kwargs)
                torch.cuda.synchronize()
                return result
            GAMQwen3DLM._flex_qwen_attention = staticmethod(synchronized_flex)
            state['execution_diagnostic'] = 'synchronize native noisy FlexAttention before native clean attention; masks/math unchanged'
        torch.manual_seed(plan['seed']); random.seed(plan['seed']); np.random.seed(plan['seed'])
        functorch_config.donated_buffer = False
        pairs, corpus = build_pairs(plan)
        checkpoint = ROOT/'models/GroundAnything-DLM'
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        collator = DLMDataCollator(processor, max_length=2048, max_joint_length=4096, model_family='groundinganything_qwen3')
        # Every declared pair must fit the actual native collator; never truncate.
        for pair in pairs:
            trial = collator([r['sft'] for r in pair['rows']]); del trial
        state.update(status='ALL_REGISTERED_NATIVE_DENSE_REF_ROWS_PREFLIGHT_PASSED', corpus=corpus); save()
        raw = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        Config = get_class_from_dynamic_module('configuration_groundinganything.GroundAnythingVLMConfig', str(checkpoint), local_files_only=True)
        Model = get_class_from_dynamic_module('modeling_groundinganything.GroundAnythingVLMForConditionalGeneration', str(checkpoint), local_files_only=True)
        config = Config.from_dict(raw.to_dict()); config._attn_implementation = 'eager'
        with safe_open(checkpoint/'model.safetensors', framework='pt') as tensors:
            config.text_config.vocab_size = tensors.get_slice('base_model.model.language_model.embed_tokens.weight').get_shape()[0]
        assert config.text_config.vocab_size == len(processor.tokenizer)
        mask = processor.tokenizer.encode('|<MASK>|', add_special_tokens=False); assert len(mask) == 1
        with init_empty_weights(include_buffers=False):
            model = GAMQwen3DLM(Model(config), mask[0], processor.tokenizer.convert_tokens_to_ids('<|im_end|>'), block_size=32)
        model.load_state_dict(load_file(checkpoint/'model.safetensors'), strict=True, assign=True)
        model.to('cuda:0').requires_grad_(False)
        projector = model.base_model.model.visual.projector
        projector.float().requires_grad_(True)
        if args.initial_projector_state:
            assert plan['initial_projector_state'] == str(args.initial_projector_state)
            initialized = torch.load(args.initial_projector_state, map_location='cpu', weights_only=True)
            projector.load_state_dict(initialized['native_projector_state_dict'])
            state['initial_projector_optimization_steps'] = initialized['steps']
            state['new_stage_Adam_reset'] = True
            del initialized
        initial = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}
        model.train(); model.base_model.model.visual.vision_tower.eval()
        model.gradient_checkpointing_enable({'use_reentrant': False})
        optimizer = torch.optim.AdamW(projector.parameters(), lr=plan['lr'], weight_decay=plan['weight_decay'])
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(p.numel() for p in projector.parameters())
        pooler = NativeGroundFeatures.__new__(NativeGroundFeatures); pooler.processor = processor.image_processor
        rng = random.Random(plan['seed']); schedule = []; pools = {}; cursors = {}
        for family in ('dense', 'referring'):
            pools[family] = [i for i, p in enumerate(pairs) if p['family'] == family]
            rng.shuffle(pools[family]); cursors[family] = 0
        for step in range(plan['steps']):
            family = 'dense' if step % 2 == 0 else 'referring'
            if cursors[family] == len(pools[family]):
                rng.shuffle(pools[family]); cursors[family] = 0
            schedule.append(pools[family][cursors[family]]); cursors[family] += 1
        assert set(schedule) == set(range(len(pairs)))
        start_step = 0
        if args.resume_state:
            restored = torch.load(args.resume_state, map_location='cpu', weights_only=True)
            assert restored['protocol'] == plan and restored['schedule'] == schedule
            projector.load_state_dict(restored['native_projector_state_dict']); optimizer.load_state_dict(restored['optimizer'])
            torch.set_rng_state(restored['torch_rng']); torch.cuda.set_rng_state_all(restored['cuda_rng'])
            random.setstate(restored['python_rng']); n = restored['numpy_rng']
            np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
            start_step = restored['steps']
        captured = []
        hook = projector.register_forward_hook(lambda module, args, output: captured.append(output))
        def contrastive(spatials, metadata, pair, source):
            common = sorted(set(pair['rows'][0][source]) & set(pair['rows'][1][source]))
            if len(common) <= 1:
                return spatials[0].sum()*0, len(common)
            regions = []
            for spatial, meta, row in zip(spatials, metadata, pair['rows']):
                pooled, _ = pooler.pool(spatial, meta, [row[source][identity] for identity in common])
                regions.append(F.normalize(pooled.float(), dim=-1))
            logits = regions[0]@regions[1].T/.07; target = torch.arange(len(common), device='cuda:0')
            return .5*(F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target)), len(common)
        def checkpoint_state(step):
            n = np.random.get_state()
            payload = {'native_projector_state_dict': {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'steps': step, 'seed': plan['seed'], 'protocol': plan, 'schedule': schedule,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'python_rng': random.getstate(), 'numpy_rng': (n[0], n[1].tolist(), n[2], n[3], n[4]),
                'initial_DLM_checkpoint': str(checkpoint), 'trainable_policy': 'native_projector_only'}
            if args.initial_projector_state:
                payload['initial_projector_state'] = str(args.initial_projector_state)
            temporary = out/'latest_training_state.tmp'; torch.save(payload, temporary)
            temporary.replace(out/'latest_training_state.pt')
            (out/'latest_training_state.json').write_text(json.dumps({'steps': step, 'FP32_projector_Adam_all_RNG_cursor': True,
                'bytes': (out/'latest_training_state.pt').stat().st_size, 'feature_cache': False}, indent=2)+'\n')
            return payload
        logs = []; probes = {}; trained_pairs = set(); start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        state.update(status='RUNNING_NATIVE_GROUND_TRACKING200', steps=start_step,
            resume_state=str(args.resume_state) if args.resume_state else None,
            trainable_parameters=sum(p.numel() for p in projector.parameters())); save()
        for step in range(start_step+1, plan['steps']+1):
            pair = pairs[schedule[step-1]]
            state['current_supervision'] = 'absent_referring' if pair.get('absent_expression') else pair['family']
            with_images = []
            for row in pair['rows']:
                with Image.open(row['image_path']) as raw_image:
                    with_images.append(raw_image.convert('RGB'))
            batch = collator([r['sft'] for r in pair['rows']])
            batch = {k: v.to('cuda:0') if torch.is_tensor(v) else v for k, v in batch.items()}
            captured.clear(); optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(**batch); assert len(captured) == 1
                features = torch.cat(captured[0], dim=0); offset = 0; spatials = []; metadata = []
                for image, (t, h, w) in zip(with_images, batch['image_grid_thw'].tolist()):
                    assert t == 1
                    gh, gw = h//2, w//2; count = gh*gw
                    spatials.append(features[offset:offset+count].reshape(gh, gw, -1).permute(2, 0, 1).unsqueeze(0)); offset += count
                    metadata.append({'image_size': [image.width, image.height],
                        'resize_and_padding': processor.image_processor.get_resize_config({'type': 'image', 'image': image})})
                assert offset == len(features)
                teacher, gt_common = contrastive(spatials, metadata, pair, 'gt')
                native, native_common = contrastive(spatials, metadata, pair, 'native')
                loss = output.loss+.1*teacher+.1*native
            if pair['family'] not in probes:
                parameters = list(projector.parameters()); record = {}
                for label, component, supported in [('native_DLM', output.loss, True), ('teacher_identity', teacher, gt_common > 1), ('decoded_proposal_identity', native, native_common > 1)]:
                    if supported:
                        grads = torch.autograd.grad(component, parameters, retain_graph=True, allow_unused=True)
                        norm = float(sum(g.float().square().sum() for g in grads if g is not None).sqrt())
                        assert norm > 0; record[label+'_gradient_norm'] = norm; del grads
                probes[pair['family']] = record
                state['gradient_probes'] = probes
            loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.))
            assert torch.isfinite(loss) and norm > 0
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            optimizer.step(); trained_pairs.add(schedule[step-1])
            state['steps'] = step
            entry = {'step': step, 'family': pair['family'], 'pair_index': schedule[step-1], 'native_loss': float(output.loss),
                'mdm_loss': float(output.mdm_loss), 'causal_loss': float(output.causal_loss), 'teacher_identity_loss': float(teacher),
                'decoded_proposal_identity_loss': float(native), 'GT_common_IDs': gt_common, 'native_common_IDs': native_common,
                'total_loss': float(loss), 'gradient_norm': norm, 'elapsed_seconds': time.perf_counter()-start}
            if plan.get('absent_expression_registry'):
                entry['absent_expression_pair'] = bool(pair.get('absent_expression'))
            logs.append(entry)
            with (out/'log.jsonl').open('a') as log:
                log.write(json.dumps(entry)+'\n')
            if step == 1 or step % 10 == 0:
                print(json.dumps(entry), flush=True)
            if step == 1 or step % args.checkpoint_every == 0:
                checkpoint_state(step); state.update(steps=step, latest_complete_training_state_step=step); save()
        hook.remove()
        assert set(schedule[:start_step]) | trained_pairs == set(range(len(pairs)))
        changed = sum(int((v.detach().cpu() != initial[k]).sum()) for k, v in projector.state_dict().items())
        bf16_changed = sum(int((v.detach().cpu().bfloat16() != initial[k].bfloat16()).sum()) for k, v in projector.state_dict().items())
        assert changed > 0 and bf16_changed > 0
        payload = checkpoint_state(plan['steps']); torch.save(payload, out/'projector_delta.pt')
        reloaded = torch.load(out/'projector_delta.pt', map_location='cuda:0', weights_only=True)
        difference = max(float((v-reloaded['native_projector_state_dict'][k]).abs().max()) for k, v in projector.state_dict().items())
        assert difference == 0
        state.update(status='NATIVE_GROUND_TRACKING200_TRAINING_COMPLETE', steps=plan['steps'], corpus=corpus,
            changed_FP32_elements=changed, changed_BF16_elements=bf16_changed, checkpoint_reload_max_difference=difference,
            gradient_probes=probes, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            all_registered_pairs_visited=True, registered_pair_count=len(pairs), full_model_original_tensor_count=569,
            dense_referring_tracking_retention_evaluation='pending actual unchanged before/after scoring',
            complete_training_not_full_research_goal=True); save()
    except Exception as error:
        (out/'failure_traceback.txt').write_text(traceback.format_exc())
        state.update(status='NATIVE_GROUND_TRACKING200_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
