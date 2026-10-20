"""Adapter from the official GroundAnything visual forward to shared state."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .shared_ground_state import SharedGroundState


class G2FeatureBridge:
    """Call one GroundAnything feature provider per frame.

    The provider is injected so tests can use a deterministic fixture while
    the real runner uses :class:`NativeGroundFeatures` loaded from G2.
    """

    def __init__(self, provider: Any, *, model_sha256: str, device: str | None = None):
        self.provider = provider
        self.model_sha256 = str(model_sha256)
        self.device = device
        self.forward_count = 0
        self.last_frame_id: int | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, device: str = "cuda:0",
                         model_sha256: str | None = None) -> "G2FeatureBridge":
        checkpoint = Path(checkpoint).resolve()
        # The manifest's base-model digest is passed explicitly.  Hashing a
        # multi-gigabyte safetensors file during every smoke launch would make
        # provenance checks needlessly expensive.
        if not model_sha256:
            model_sha256 = f"unregistered:{checkpoint.name}:{(checkpoint / 'model.safetensors').stat().st_size}"
        from ..native_features import NativeGroundFeatures
        return cls(NativeGroundFeatures(checkpoint=checkpoint, device=device),
                   model_sha256=model_sha256, device=device)

    def encode(
        self,
        image: Any,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        *,
        frame_id: int,
        image_key: str,
        task_query: str | None = None,
        scores: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SharedGroundState:
        if self.last_frame_id is not None and int(frame_id) <= self.last_frame_id:
            raise ValueError("frame_id must increase monotonically for a causal bridge")
        result = self.provider(image, boxes)
        contextual = None
        if isinstance(result, dict):
            pooled, local = result["region_tokens"], result.get("local_tokens")
            contextual = result.get("contextual_tokens")
            provider_metadata = dict(result.get("metadata") or {})
        elif isinstance(result, tuple) and len(result) >= 2:
            pooled, local = result[0], result[1]
            provider_metadata = dict(result[2] if len(result) > 2 and result[2] else {})
            contextual = result[3] if len(result) > 3 else None
        else:
            raise TypeError("feature provider must return a dict or (pooled_tokens, local_tokens, metadata)")
        self.forward_count += 1
        self.last_frame_id = int(frame_id)
        state_metadata = {**provider_metadata, **dict(metadata or {})}
        state_metadata.update({"bridge_forward_count": self.forward_count,
                               "single_backbone": True,
                               "feature_provider": type(self.provider).__name__})
        return SharedGroundState.from_region_tokens(
            frame_id=frame_id, image_key=image_key, model_sha256=self.model_sha256,
            boxes=boxes, labels=labels, region_tokens=pooled, local_tokens=local,
            contextual_tokens=contextual,
            image_grid_thw=None, spatial_layout={"provider": provider_metadata.get("merged_grid")},
            task_query=task_query, scores=scores, metadata=state_metadata,
        )
