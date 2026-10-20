"""One-call GroundAnything visual bridge for the coupled runtime."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from ..native_features import NativeGroundFeatures
from .region_tokens import pool_region_tokens
from .shared_ground_state import SharedGroundState


class G2FeatureBridge:
    """Encode one image once and expose shared visual evidence.

    The bridge can load the formal G2 projector from an adapter-only checkpoint
    while the full DLM bundle is still training.  This is a visual-state Phase
    A path; it records that language LoRA is not loaded into the visual bridge
    and therefore cannot be mistaken for a complete G2 DLM forward.
    """

    def __init__(self, *, checkpoint: str | Path,
                 device: str = "cuda:0",
                 g2_adapter_checkpoint: str | Path | None = None,
                 local_source_root: str | Path | None = None) -> None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.checkpoint = checkpoint
        self.device = device
        self.local_source_root = Path(local_source_root).resolve() if local_source_root else None
        self.extractor = NativeGroundFeatures(
            checkpoint=checkpoint,
            device=device,
            adapter_checkpoint=g2_adapter_checkpoint,
            source_root=self.local_source_root,
        )
        self.forward_calls = 0
        self.pool_calls = 0

    @property
    def model_sha256(self) -> str:
        return self.extractor.model_sha256

    @property
    def feature_dimension(self) -> int:
        return int(self.extractor.feature_dimension)

    def encode(self, image: Any, boxes: Sequence[Sequence[float]], *,
               frame_id: int | str, image_key: str,
               task_query: str = "", contextual_tokens: Optional[torch.Tensor] = None,
               metadata: Optional[dict[str, Any]] = None) -> SharedGroundState:
        """Run exactly one native visual encoder call for this frame."""
        if not image_key:
            raise ValueError("image_key is required for causal feature provenance")
        spatial, native_metadata = self.extractor.encode(image)
        self.forward_calls += 1
        self.pool_calls += 1
        pooled, _ = self.extractor.pool(spatial, native_metadata, boxes)
        t, h, w = (int(v) for v in native_metadata["grid_thw"])
        gh, gw = (int(v) for v in native_metadata["merged_grid"])
        all_metadata = dict(native_metadata)
        all_metadata.update(metadata or {})
        all_metadata.update({
            "bridge": "G2FeatureBridge",
            "visual_forward_calls_for_frame": 1,
            "bridge_total_forward_calls": self.forward_calls,
            "bridge_total_pool_calls": self.pool_calls,
            "language_context_source": "caller_optional_contextual_tokens",
            "second_visual_encoder": False,
        })
        # NativeGroundFeatures returns [1,C,H,W].  Keep the spatial map as the
        # shared visual token tensor and the region rows in candidate order.
        boxes_tuple = tuple(tuple(float(v) for v in box) for box in boxes)
        return SharedGroundState(
            frame_id=frame_id,
            image_key=image_key,
            model_sha256=self.model_sha256,
            boxes=boxes_tuple,
            vision_tokens=spatial,
            region_tokens=pooled,
            image_grid_thw=(t, h, w),
            spatial_layout=(gh, gw),
            image_size=(int(native_metadata["image_size"][0]),
                        int(native_metadata["image_size"][1])),
            task_query=task_query,
            contextual_tokens=contextual_tokens,
            metadata=all_metadata,
        )

    def encode_from_spatial(self, *, spatial: torch.Tensor,
                            image_size: tuple[int, int],
                            boxes: Sequence[Sequence[float]],
                            frame_id: int | str, image_key: str,
                            image_grid_thw: tuple[int, int, int] = (1, 0, 0),
                            task_query: str = "", contextual_tokens: Optional[torch.Tensor] = None,
                            metadata: Optional[dict[str, Any]] = None) -> SharedGroundState:
        """Build a state from an already-produced spatial map for CPU tests.

        This method never calls a visual encoder and is useful for checking the
        readout/lifecycle contract independently.  Callers must label the
        provenance in ``metadata`` when using synthetic features.
        """
        spatial = spatial if spatial.ndim == 4 else spatial.unsqueeze(0)
        if spatial.shape[0] != 1:
            raise ValueError("encode_from_spatial expects one frame")
        pooled, _ = pool_region_tokens(spatial, boxes, image_size=image_size)
        gh, gw = int(spatial.shape[-2]), int(spatial.shape[-1])
        return SharedGroundState(
            frame_id=frame_id,
            image_key=image_key,
            model_sha256=self.model_sha256,
            boxes=tuple(tuple(float(v) for v in box) for box in boxes),
            vision_tokens=spatial,
            region_tokens=pooled,
            image_grid_thw=image_grid_thw,
            spatial_layout=(gh, gw),
            image_size=image_size,
            task_query=task_query,
            contextual_tokens=contextual_tokens,
            metadata={"bridge": "G2FeatureBridge.encode_from_spatial",
                      "visual_forward_calls_for_frame": 0,
                      "second_visual_encoder": False, **(metadata or {})},
        )
