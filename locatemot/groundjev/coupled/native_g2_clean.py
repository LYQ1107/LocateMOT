"""Native GroundAnything G2 clean-DLM feature provider.

This adapter deliberately keeps the official DLM as the only visual and
language backbone.  It restores a protected G2 adapter in memory, renders a
small clean row for the requested image/query, runs the native clean stream
once, and pools image hidden tokens into candidate regions.  It is intended
for bounded coupling smoke tests and later training entrypoints; it never
writes to the formal G2 run directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


class NativeG2CleanDLMProvider:
    """One frozen native DLM forward yielding region and context tokens."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        adapter: str | Path,
        runtime_root: str | Path,
        device: str = "cuda:0",
        task_query: str = "person",
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.adapter_path = Path(adapter).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.device = torch.device(device)
        self.task_query = str(task_query or "person")
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("NativeG2CleanDLMProvider requires CUDA")
        os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
        os.environ.setdefault(
            "LOCATEMOT_GROUNDANYTHING_ROOT",
            str(self.runtime_root / "external/GroundAnything"),
        )
        sys.path.insert(0, str(self.runtime_root))
        sys.path.insert(0, str(self.runtime_root / "external/GroundAnything"))
        from tools.groundjev.g2.compat import patch_transformers_causal_mask
        from tools.groundjev.g2.train_g2_dlm import load_model, _strict_load_trainable_state

        self.mask_compatibility = patch_transformers_causal_mask()
        self.model, self.processor, _, self.audit = load_model(self.checkpoint, self.device)
        saved = torch.load(self.adapter_path, map_location="cpu", weights_only=False)
        if int(saved.get("step", -1)) <= 0:
            raise RuntimeError("native G2 adapter has no positive step")
        _strict_load_trainable_state(self.model, saved["model_trainable_state"])
        self.adapter_step = int(saved["step"])
        self.model.set_padding_free_packing(False)
        self.model.gradient_checkpointing_disable()
        self.model.eval()
        from train.dlm.data import DLMDataCollator

        self.collator = DLMDataCollator(
            self.processor,
            max_length=32768,
            max_joint_length=65536,
            model_family="groundinganything_qwen3",
        )
        self.forward_count = 0

    @staticmethod
    def _box_answer(query: str, boxes: Sequence[Sequence[float]]) -> str:
        values: list[str] = []
        for box in boxes:
            coords = [str(max(0, int(round(float(value))))) for value in box]
            if len(coords) != 4:
                raise ValueError("every candidate box must contain four coordinates")
            values.append("<" + "><".join(coords) + ">")
        return (
            f"<|object_ref_start|>{query}<|object_ref_end|>"
            f"<|box_start|>{','.join(values)}<|box_end|>"
        )

    def _row(self, image_key: str, boxes: Sequence[Sequence[float]], query: str) -> dict[str, Any]:
        image_path = Path(image_key).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"native DLM provider requires an image path: {image_path}")
        return {
            "id": f"groundjev_native_{image_path.stem}",
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>\nLocate all the instances that match the following categories: {query}.",
                    "loss": None,
                },
                {"role": "assistant", "content": self._box_answer(query, boxes), "loss": None},
            ],
            "images": [{"path": str(image_path), "bytes": None}],
            "source": "GroundJEV:native_clean_context",
        }

    def encode_frame(
        self,
        image: Any,
        boxes: Sequence[Sequence[float]],
        *,
        image_key: str,
        task_query: str | None = None,
    ) -> dict[str, Any]:
        del image
        query = str(task_query or self.task_query)
        row = self._row(image_key, boxes, query)
        batch = self.collator([row])
        tensor_batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        visual_dtype = next(self.model.base_model.model.visual.parameters()).dtype
        for key in ("pixel_values", "pixel_values_videos"):
            if key in tensor_batch:
                tensor_batch[key] = tensor_batch[key].to(dtype=visual_dtype)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            clean_embeds, position_ids = self.model._embed_clean(
                tensor_batch["input_ids"], tensor_batch["attention_mask"],
                tensor_batch.get("pixel_values"), tensor_batch.get("image_grid_thw"),
                tensor_batch.get("pixel_values_videos"), tensor_batch.get("video_grid_thw"),
                tensor_batch.get("mm_token_type_ids"), tensor_batch.get("patch_positions"),
            )
            streams = self.model._build_streams(
                tensor_batch["input_ids"], tensor_batch["labels"], tensor_batch["attention_mask"],
                clean_embeds, position_ids,
            )
            _, clean_hidden = self.model._hybrid_language_forward(streams)
        image_pad_id = int(self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        valid = tensor_batch["attention_mask"].bool()
        image_positions = tensor_batch["input_ids"].eq(image_pad_id) & valid
        image_tokens = clean_hidden[image_positions]
        grid = tensor_batch["image_grid_thw"][0].tolist()
        _, height, width = (int(value) for value in grid)
        merge = int(self.model.base_model.model.visual.spatial_merge_size)
        grid_height, grid_width = height // merge, width // merge
        expected = grid_height * grid_width
        if int(image_tokens.shape[0]) != expected:
            raise RuntimeError(
                f"native image hidden layout mismatch: tokens={image_tokens.shape[0]} expected={expected}"
            )
        spatial = image_tokens.reshape(grid_height, grid_width, -1).permute(2, 0, 1).unsqueeze(0).contiguous()
        pooled, local = self._pool(spatial, boxes, image_key)
        text_positions = valid & ~image_positions
        context = clean_hidden[text_positions].mean(dim=0, keepdim=True)
        self.forward_count += 1
        return {
            "region_tokens": pooled,
            "local_tokens": local,
            "contextual_tokens": context.float(),
            "metadata": {
                "feature_source": "official_G2_native_clean_DLM_hidden",
                "adapter_step": self.adapter_step,
                "native_dlm_forward_count": self.forward_count,
                "image_hidden_token_count": int(image_tokens.shape[0]),
                "context_hidden_token_count": int(text_positions.sum().item()),
                "image_grid_thw": [int(value) for value in grid],
                "merged_grid": [grid_height, grid_width],
                "mask_compatibility": self.mask_compatibility,
                "single_backbone": True,
            },
        }

    def _pool(
        self,
        spatial: torch.Tensor,
        boxes: Sequence[Sequence[float]],
        image_key: str,
        *,
        local_size: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image
        from torchvision.ops import roi_align

        image = Image.open(image_key).convert("RGB")
        image_processor = self.processor.image_processor
        resize = image_processor.get_resize_config({"type": "image", "image": image})
        cell = int(image_processor.patch_size) * int(image_processor.merge_size)
        if not boxes:
            return (
                spatial.new_empty((0, spatial.shape[1]), dtype=torch.float32),
                spatial.new_empty((0, spatial.shape[1], local_size, local_size), dtype=torch.float32),
            )
        box_tensor = torch.tensor(boxes, device=spatial.device, dtype=torch.float32)
        scales = box_tensor.new_tensor([
            resize["new_width"] / image.width / cell,
            resize["new_height"] / image.height / cell,
        ] * 2)
        feature_boxes = box_tensor * scales
        rois = torch.cat([
            torch.zeros((len(boxes), 1), device=spatial.device), feature_boxes,
        ], dim=1)
        local = roi_align(
            spatial.float(), rois, output_size=local_size,
            spatial_scale=1.0, sampling_ratio=2, aligned=True,
        )
        return local.mean(dim=(-1, -2)), local

    def __call__(self, image: Any, boxes: Sequence[Sequence[float]]):
        return self.encode_frame(image, boxes, image_key=str(image), task_query=self.task_query)
