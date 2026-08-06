"""Instrumented LocateAnything generation with PBD block event tracing.

This module does NOT modify third_party/Eagle. It drives the same language-model
forward calls as the official ``generate`` (NVlabs/Eagle commit 783f656d,
Apache-2.0) while requesting ``output_hidden_states=True``, and re-runs the
official ``sample_tokens`` / ``handle_pattern`` decoding functions on the same
logits to record accepted/rejected blocks and their hidden-state positions.
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .types import GenerationBlockEvent


def load_generate_utils(model_dir: str):
    """Load the official generate_utils.py from a local LocateAnything dir."""
    path = os.path.join(model_dir, "generate_utils.py")
    spec = importlib.util.spec_from_file_location("locatemot_la_generate_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def force_vision_attention_eager(model) -> None:
    """Switch MoonViT encoder layers to the official eager attention path.

    The official sdpa path uses a 3D bool mask with F.scaled_dot_product_attention
    and can fail/OOM on A100 for larger token counts. Eager is the fallback
    attention implementation provided by the same official code.
    """
    vision = getattr(model, "vision_model", None)
    if vision is None:
        return
    try:
        for block in vision.encoder.blocks:
            block.attn_implementation = "eager"
        vision.config._attn_implementation = "eager"
    except AttributeError:
        pass


class InstrumentedLocateAnythingGeneration:
    """Replicates the official generate loop with block event + hidden tracing."""

    def __init__(self, model, tokenizer, processor, model_dir: str, seed: Optional[int] = None):
        self.model = model
        force_vision_attention_eager(model)
        self.tokenizer = tokenizer
        self.processor = processor
        self.gu = load_generate_utils(model_dir)
        self.seed = seed
        self.n_future_tokens = 6

    def _build_inputs(self, image, question: str) -> Dict[str, Any]:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        )
        return inputs

    def run(
        self,
        image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.1,
        keep_layers: Tuple[int, ...] = (-1, -2),
        in_token_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        if in_token_limit is not None:
            self.processor.image_processor.in_token_limit = in_token_limit
        if self.seed is not None:
            torch.manual_seed(self.seed)
        inputs = self._build_inputs(image, question)
        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        vision_dtype = next(self.model.vision_model.parameters()).dtype
        pixel_values = inputs["pixel_values"].to(vision_dtype).to(device)
        image_grid_hws = inputs.get("image_grid_hws", None)
        if image_grid_hws is not None:
            image_grid_hws = torch.as_tensor(image_grid_hws, dtype=torch.int32, device=device)
        image_size = [image.size[0], image.size[1]]

        return self._generate_loop(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            question=question,
            image_size=image_size,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            keep_layers=keep_layers,
        )

    @torch.no_grad()
    def _generate_loop(
        self,
        input_ids,
        pixel_values,
        image_grid_hws,
        question: str,
        image_size: List[int],
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: float,
        keep_layers: Tuple[int, ...],
    ) -> Dict[str, Any]:
        model = self.model
        gu = self.gu
        token_ids = model.token_ids
        tokenizer = self.tokenizer
        device = input_ids.device
        dtype = model.language_model.dtype

        assert generation_mode in ("fast", "slow", "hybrid")
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, "official standard runtime supports batch_size=1 only"

        generated = input_ids.clone()
        total_gen_length = min(tokenizer.model_max_length, seq_len + max_new_tokens)
        past_key_values = None
        use_mtp = generation_mode in ("fast", "hybrid")
        switch_to_ar_count = 0

        # visual features: same as official generate; keep raw list for region reuse
        raw_vision_features = model.extract_feature(pixel_values, image_grid_hws)
        vit_embeds = raw_vision_features
        if image_grid_hws is not None:
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = model.mlp1(vit_embeds)

        default_mask_token_id = token_ids["default_mask_token_id"]
        pre_mask_tokens = torch.full(
            (1, self.n_future_tokens - 1),
            default_mask_token_id,
            dtype=generated.dtype,
            device=device,
        )
        max_possible_len = total_gen_length + self.n_future_tokens
        full_position_ids = torch.arange(0, max_possible_len, device=device).unsqueeze(0)

        gen_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        events: List[GenerationBlockEvent] = []
        hidden_slices: List[Dict[int, torch.Tensor]] = []
        ar_box_tokens: List[Dict[str, Any]] = []  # pending AR box accumulation
        output_order_counter = 0

        # Capture only the final-norm output and the last decoder layer output
        # via forward hooks instead of output_hidden_states=True (large memory
        # saving). Layer indices: n_layers = post-norm output (official
        # hidden_states[-1]); n_layers - 1 = last layer output (official
        # hidden_states[-2]).
        n_layers = model.language_model.config.num_hidden_layers
        layer_indices = [n_layers, n_layers - 1]
        hook_capture: Dict[str, torch.Tensor] = {}

        def _make_hook(key):
            def _hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                hook_capture[key] = h.detach().to("cpu", dtype=torch.float32)
            return _hook

        hooks = []
        hooks.append(model.language_model.model.norm.register_forward_hook(_make_hook("last")))
        hooks.append(
            model.language_model.model.layers[-1].register_forward_hook(_make_hook("penultimate"))
        )

        def _prepare_mtp(current_generated):
            generated_with_mask = torch.cat(
                (current_generated, current_generated[:, -1].unsqueeze(1), pre_mask_tokens),
                dim=1,
            )
            start_idx = (
                past_key_values[0][0].size(2) if past_key_values is not None else 0
            )
            position_ids = full_position_ids[:, start_idx : generated_with_mask.size(1)].clone()
            position_ids[0, -self.n_future_tokens :] -= 1
            prep = model.language_model.prepare_inputs_for_generation(
                generated_with_mask,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=position_ids,
            )
            return prep

        def _prepare_ar(current_generated):
            start_idx = (
                past_key_values[0][0].size(2) if past_key_values is not None else 0
            )
            position_ids = full_position_ids[:, start_idx : current_generated.size(1)]
            prep = model.language_model.prepare_inputs_for_generation(
                current_generated,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=position_ids,
            )
            return prep

        def _coord_value(tok_id: int) -> int:
            return int(tok_id - token_ids["coord_start_token_id"])

        def _finalize_ar_box() -> Optional[GenerationBlockEvent]:
            nonlocal output_order_counter
            if not ar_box_tokens:
                return None
            token_ids_list = [t["token_id"] for t in ar_box_tokens]
            decoded = [t["decoded"] for t in ar_box_tokens]
            coord_vals = [
                _coord_value(tok)
                for tok in token_ids_list
                if token_ids["coord_start_token_id"] <= tok <= token_ids["coord_end_token_id"]
            ]
            is_none = any(tok == token_ids["none_token_id"] for tok in token_ids_list)
            ev = GenerationBlockEvent(
                generation_step=ar_box_tokens[-1]["step"],
                decode_mode="empty/None" if is_none else "AR/NTP",
                attempted_mode="NTP",
                accepted_mode="NTP",
                fallback_occurred=False,
                token_ids=token_ids_list,
                decoded_tokens=decoded,
                block_type="empty_box" if is_none else "coord_box",
                block_start_position=ar_box_tokens[0]["output_pos"],
                block_end_position=ar_box_tokens[-1]["output_pos"],
                hidden_state_positions=[t["input_pos"] for t in ar_box_tokens],
                logits_positions=[t["output_pos"] for t in ar_box_tokens],
                parsed_box=None if is_none or len(coord_vals) != 4 else [float(v) for v in coord_vals],
                normalized_box=None if is_none or len(coord_vals) != 4 else [float(v) / 1000 for v in coord_vals],
                accepted=False if is_none or len(coord_vals) != 4 else True,
                rejection_reason="" if (is_none or len(coord_vals) == 4) else "ar_box_incomplete",
                output_order=output_order_counter if (not is_none and len(coord_vals) == 4) else -1,
                query_text=question,
                image_size=image_size,
                generation_score=float(ar_box_tokens[-1]["score"]),
                extra={
                    "token_steps": [t["step"] for t in ar_box_tokens],
                    "hidden_rel_positions": [t["hidden_rel"] for t in ar_box_tokens],
                },
            )
            if ev.accepted:
                output_order_counter += 1
            return ev

        start_time = time.time()
        step = 0
        while generated.size(1) < total_gen_length:
            step += 1
            use_mtp_before = use_mtp
            L = generated.size(1)
            prepare_inputs = _prepare_mtp(generated) if use_mtp else _prepare_ar(generated)
            if step == 1:
                prepare_inputs.update({
                    "visual_features": vit_embeds,
                    "image_token_index": model.config.image_token_index,
                })

            hook_capture.clear()
            outputs = model.language_model(
                **prepare_inputs,
                output_hidden_states=False,
                return_dict=True,
            )

            step_hidden = {
                layer_indices[0]: hook_capture.get("last"),
                layer_indices[1]: hook_capture.get("penultimate"),
            }
            step_hidden = {k: v for k, v in step_hidden.items() if v is not None}
            hidden_slices.append(step_hidden)

            past_key_values = tuple(
                (kv[0][:, :, :L, :], kv[1][:, :, :L, :])
                for kv in outputs.past_key_values
            )

            if use_mtp:
                next_logits = outputs.logits[:, -self.n_future_tokens :, :]
                probs, confidence, x0, box_avg = gu.sample_tokens(
                    next_logits, generated, token_ids, keep_k=5, **gen_kwargs
                )
                is_box_empty = bool((box_avg[0] == 0).all().item())
                new_tokens = x0[0] if is_box_empty else box_avg[0]
                out_pattern = gu.handle_pattern(new_tokens, token_ids, generation_mode)
                out_type = out_pattern["type"]
                out_token = torch.tensor(
                    out_pattern["tokens"], dtype=x0.dtype, device=device
                )
                token_list = out_token.tolist()
                decoded_list = tokenizer.batch_decode(
                    out_token.unsqueeze(0), skip_special_tokens=False
                )
                if isinstance(decoded_list, list) and decoded_list:
                    decoded_list = decoded_list[0].split(">") if decoded_list[0].count(">") > 0 else [decoded_list[0]]
                    decoded_list = [d + ">" for d in decoded_list if d.strip()]
                score = float(confidence[0].mean().item())

                ev = GenerationBlockEvent(
                    generation_step=step,
                    decode_mode={
                        "coord_box": "MTP/PBD accepted",
                        "point_box": "point",
                        "empty_box": "empty/None",
                        "error_box": "MTP rejected",
                        "ref_object": "ref object",
                        "im_end": "end block",
                    }.get(out_type, out_type),
                    attempted_mode="MTP",
                    accepted_mode="MTP" if out_type in ("coord_box", "point_box", "empty_box", "ref_object") else "NONE",
                    fallback_occurred=(out_type == "error_box"),
                    token_ids=token_list,
                    decoded_tokens=decoded_list,
                    block_type=out_type,
                    block_start_position=L,
                    block_end_position=L + len(token_list) - 1,
                    hidden_state_positions=list(range(L, L + len(token_list))),
                    logits_positions=list(range(L, L + len(token_list))),
                    parsed_box=None,
                    normalized_box=None,
                    accepted=False,
                    output_order=-1,
                    query_text=question,
                    image_size=image_size,
                    generation_score=score,
                )

                if out_type == "coord_box":
                    coords = [_coord_value(t) for t in token_list[1:5]]
                    ev.parsed_box = [float(c) for c in coords]
                    ev.normalized_box = [float(c) / 1000 for c in coords]
                    ev.accepted = True
                    ev.output_order = output_order_counter
                    output_order_counter += 1
                elif out_type == "error_box":
                    ev.rejection_reason = "format_irregular_or_ambiguous"
                    if (
                        token_list
                        and token_list[0] == token_ids["box_start_token_id"]
                    ):
                        ar_box_tokens = [
                            {
                                "token_id": tok,
                                "decoded": tok,
                                "output_pos": L + idx,
                                "input_pos": L + idx,
                                "step": step,
                                "score": score,
                                "hidden_rel": idx,
                            }
                            for idx, tok in enumerate(token_list)
                        ]
            else:
                next_logits = outputs.logits[:, -1:, :]
                probs, confidence, x0, _ = gu.sample_tokens(
                    next_logits, generated, token_ids, **gen_kwargs
                )
                out_token = x0[0]
                token_val = out_token[0].item()
                score = float(confidence[0].item())
                decoded = tokenizer.decode(out_token[0], skip_special_tokens=False)
                ev = GenerationBlockEvent(
                    generation_step=step,
                    decode_mode="AR/NTP",
                    attempted_mode="NTP",
                    accepted_mode="NTP",
                    fallback_occurred=False,
                    token_ids=[token_val],
                    decoded_tokens=[decoded],
                    block_type="ar_token",
                    block_start_position=L,
                    block_end_position=L,
                    hidden_state_positions=[L - 1],
                    logits_positions=[L],
                    parsed_box=None,
                    normalized_box=None,
                    accepted=False,
                    output_order=-1,
                    query_text=question,
                    image_size=image_size,
                    generation_score=score,
                )
                if generation_mode == "hybrid":
                    if token_val == token_ids["box_end_token_id"]:
                        ev.decode_mode = "AR/NTP"
                        ev.fallback_occurred = True  # switch back to MTP at this token
                        ev.block_type = "box_end_ar"
                    elif (
                        token_ids["coord_start_token_id"] <= token_val <= token_ids["coord_end_token_id"]
                        or token_val == token_ids["none_token_id"]
                    ):
                        ev.block_type = "coord_ar"
                    else:
                        ev.block_type = "im_end"
                        ev.decode_mode = "end block"
                else:
                    if token_val == token_ids["im_end_token_id"]:
                        ev.block_type = "im_end"
                        ev.decode_mode = "end block"

                # AR box accumulation
                if token_val == token_ids["box_start_token_id"]:
                    ar_box_tokens = [{
                        "token_id": token_val,
                        "decoded": decoded,
                        "output_pos": L,
                        "input_pos": L - 1,
                        "step": step,
                        "score": score,
                        "hidden_rel": 0,
                    }]
                    ev.block_type = "box_start_ar"
                elif ar_box_tokens:
                    ar_box_tokens.append({
                        "token_id": token_val,
                        "decoded": decoded,
                        "output_pos": L,
                        "input_pos": L - 1,
                        "step": step,
                        "score": score,
                        "hidden_rel": 0,
                    })
                    if token_val == token_ids["box_end_token_id"]:
                        box_ev = _finalize_ar_box()
                        if box_ev is not None:
                            events.append(box_ev)
                        ar_box_tokens = []

            generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)

            if out_type == "im_end" if use_mtp_before else (
                token_val == token_ids["im_end_token_id"]
            ):
                events.append(ev)
                break

            if generation_mode == "hybrid" and use_mtp_before:
                if out_type == "error_box":
                    use_mtp = False
                    switch_to_ar_count += 1
                elif out_type == "box_end_ar":
                    use_mtp = True
            elif generation_mode == "hybrid" and not use_mtp_before:
                if token_val == token_ids["box_end_token_id"]:
                    use_mtp = True

            events.append(ev)

        elapsed = time.time() - start_time
        for h in hooks:
            h.remove()
        generated_ids = generated[:, seq_len:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        answer = response[0] if response else ""

        return {
            "answer": answer,
            "events": events,
            "hidden_slices": hidden_slices,
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
            "raw_vision_features": raw_vision_features,
            "image_size": image_size,
            "generation_mode": generation_mode,
            "num_steps": step,
            "switch_to_ar_count": switch_to_ar_count,
            "elapsed_seconds": elapsed,
            "input_ids_length": seq_len,
        }
