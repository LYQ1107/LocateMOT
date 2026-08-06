"""Top-level ObjectToken extractor combining PBD hidden states, MoonViT region
features, geometry, and (untrained) projection."""
from __future__ import annotations

from typing import List, Optional

import torch

from .generation_trace import InstrumentedLocateAnythingGeneration
from .pbd_extractor import PBDObjectExtractor
from .projection import ObjectTokenProjection
from .region_extractor import MoonViTRegionExtractor
from .types import ObjectToken


class ObjectTokenExtractor:
    def __init__(
        self,
        model,
        tokenizer,
        processor,
        model_dir: str,
        model_commit: str,
        checkpoint_hash: str,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_dir = model_dir
        self.model_commit = model_commit
        self.checkpoint_hash = checkpoint_hash
        self.trace_runner = InstrumentedLocateAnythingGeneration(
            model, tokenizer, processor, model_dir, seed=seed
        )
        self.pbd = PBDObjectExtractor()
        self.region = MoonViTRegionExtractor(model)
        self.projection = ObjectTokenProjection()

    def extract(
        self,
        image,
        question: str,
        semantic_label: str,
        source_frame: str,
        generation_mode: str = "hybrid",
        in_token_limit: Optional[int] = None,
        **gen_kwargs,
    ) -> dict:
        trace = self.trace_runner.run(
            image, question, generation_mode=generation_mode,
            in_token_limit=in_token_limit, **gen_kwargs,
        )
        tokens = self.extract_from_trace(
            trace, question=question, semantic_label=semantic_label, source_frame=source_frame
        )
        return {"trace": trace, "object_tokens": tokens}

    def extract_from_trace(
        self,
        trace: dict,
        question: str,
        semantic_label: str,
        source_frame: str,
    ) -> List[ObjectToken]:
        events = trace["events"]
        hidden_slices = trace["hidden_slices"]
        pbd_entries = self.pbd.extract(events, hidden_slices)
        pbd_entries.sort(key=lambda e: e["event"].output_order)

        normalized_boxes = [e["event"].normalized_box for e in pbd_entries]
        pixel_values = trace["pixel_values"]
        grid_hws = trace["image_grid_hws"]
        region_results = (
            self.region.extract(
                pixel_values,
                grid_hws,
                0,
                normalized_boxes,
                vision_features=trace.get("raw_vision_features"),
            )
            if grid_hws is not None
            else [None] * len(normalized_boxes)
        )

        tokens: List[ObjectToken] = []
        for idx, entry in enumerate(pbd_entries):
            ev = entry["event"]
            region = region_results[idx] if idx < len(region_results) else None
            if region is None or region.get("region_feature") is None:
                region_feat_t = None
            else:
                region_feat_t = region["region_feature"].to(
                    dtype=torch.float32, device="cpu"
                )

            pbd_last = entry.get("full_mean")
            if pbd_last is None:
                continue
            pbd_last_t = pbd_last.detach().to(dtype=torch.float32, device="cpu")
            region_dim = region_feat_t.shape[0] if region_feat_t is not None else 4608
            region_fallback = torch.zeros(region_dim, dtype=torch.float32)
            region_for_proj = region_feat_t if region_feat_t is not None else region_fallback

            nb = ev.normalized_box or [0.0, 0.0, 0.0, 0.0]
            image_size = trace.get("image_size") or [1, 1]
            pixel_box = [
                nb[0] * image_size[0],
                nb[1] * image_size[1],
                nb[2] * image_size[0],
                nb[3] * image_size[1],
            ]
            area = max(0.0, (nb[2] - nb[0]) * (nb[3] - nb[1]))
            geometry = torch.tensor(
                [nb[0], nb[1], nb[2], nb[3], min(area, 1.0)],
                dtype=torch.float32,
            )
            gen_vec = torch.zeros(16, dtype=torch.float32)
            gen_vec[0] = float(ev.generation_score or 0.0)

            with torch.no_grad():
                fused = self.projection(pbd_last_t, region_for_proj, geometry, gen_vec)

            to_list = lambda t: (t.detach().cpu().tolist() if t is not None else None)
            tokens.append(ObjectToken(
                object_index=ev.output_order,
                box_xyxy=pixel_box,
                normalized_box=ev.normalized_box,
                query_text=question,
                semantic_label=semantic_label,
                decode_mode=ev.decode_mode,
                block_start=ev.block_start_position,
                block_end=ev.block_end_position,
                pbd_box_end_feature=to_list(entry.get("box_end")),
                pbd_coordinate_mean_feature=to_list(entry.get("coord_mean")),
                pbd_full_block_mean_feature=to_list(entry.get("full_mean")),
                pbd_box_end_penultimate_feature=to_list(entry.get("box_end_penultimate")),
                pbd_coordinate_mean_penultimate_feature=to_list(entry.get("coord_mean_penultimate")),
                pbd_full_block_mean_penultimate_feature=to_list(entry.get("full_mean_penultimate")),
                region_feature=to_list(region_feat_t),
                geometry_feature=geometry.tolist(),
                confidence_feature=None,
                generation_score=float(ev.generation_score or 0.0),
                fused_feature=to_list(fused),
                image_size=trace.get("image_size"),
                feature_grid_shape=region.get("feature_grid_shape") if region else None,
                region_token_count=region.get("region_token_count", 0) if region else 0,
                box_in_feature_coordinates=region.get("box_in_feature_coordinates") if region else None,
                source_frame=source_frame,
                model_commit=self.model_commit,
                checkpoint_hash=self.checkpoint_hash,
                extra={
                    "generation_step": ev.generation_step,
                    "fallback_occurred": ev.fallback_occurred,
                },
            ))
        return tokens
