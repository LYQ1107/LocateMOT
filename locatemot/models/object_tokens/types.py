"""Data types for PBD generation traces and ObjectTokens.

Reference: NVlabs/Eagle (Apache-2.0) commit 783f656d; the event schema follows
the official PBD decode semantics documented in Embodied/eaglevl/utils/locany.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationBlockEvent:
    """One generation step (MTP block attempt or AR token) recorded from the
    official LocateAnything generate loop.

    ``token_positions`` are absolute output-sequence positions; for MTP they are
    [L, L+1, ..., L+5]; for AR they are [L]. ``hidden_state_positions`` are the
    absolute input positions whose hidden states predict those tokens.
    """

    sample_index: int = 0
    batch_index: int = 0
    generation_step: int = 0
    decode_mode: str = ""
    attempted_mode: str = ""
    accepted_mode: str = ""
    fallback_occurred: bool = False
    token_ids: List[int] = field(default_factory=list)
    decoded_tokens: List[str] = field(default_factory=list)
    block_type: str = ""
    block_start_position: int = 0
    block_end_position: int = 0
    hidden_state_positions: List[int] = field(default_factory=list)
    logits_positions: List[int] = field(default_factory=list)
    parsed_box: Optional[List[float]] = None
    normalized_box: Optional[List[float]] = None
    accepted: bool = False
    rejection_reason: str = ""
    output_order: int = -1
    query_text: str = ""
    image_size: Optional[List[int]] = None
    generation_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "sample_index": self.sample_index,
            "batch_index": self.batch_index,
            "generation_step": self.generation_step,
            "decode_mode": self.decode_mode,
            "attempted_mode": self.attempted_mode,
            "accepted_mode": self.accepted_mode,
            "fallback_occurred": self.fallback_occurred,
            "token_ids": self.token_ids,
            "decoded_tokens": self.decoded_tokens,
            "block_type": self.block_type,
            "block_start_position": self.block_start_position,
            "block_end_position": self.block_end_position,
            "hidden_state_positions": self.hidden_state_positions,
            "logits_positions": self.logits_positions,
            "parsed_box": self.parsed_box,
            "normalized_box": self.normalized_box,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "output_order": self.output_order,
            "query_text": self.query_text,
            "image_size": self.image_size,
            "generation_score": self.generation_score,
            "extra": self.extra,
        }
        return d


@dataclass
class ObjectToken:
    """Per-object token extracted from one accepted PBD coordinate box block.

    ``pbd_*`` features come from official model hidden states; ``region_feature``
    comes from MoonViT raw features; ``fused_feature`` and all ``*_projected``
    fields are produced by randomly-initialized projection layers and are for
    interface testing only (not trained in Stage L0-B).
    """

    object_index: int = 0
    box_xyxy: Optional[List[float]] = None
    normalized_box: Optional[List[float]] = None
    query_text: str = ""
    semantic_label: str = ""
    decode_mode: str = ""
    block_start: int = 0
    block_end: int = 0
    pbd_box_end_feature: Optional[List[float]] = None
    pbd_coordinate_mean_feature: Optional[List[float]] = None
    pbd_full_block_mean_feature: Optional[List[float]] = None
    pbd_box_end_penultimate_feature: Optional[List[float]] = None
    pbd_coordinate_mean_penultimate_feature: Optional[List[float]] = None
    pbd_full_block_mean_penultimate_feature: Optional[List[float]] = None
    region_feature: Optional[List[float]] = None
    geometry_feature: Optional[List[float]] = None
    confidence_feature: Optional[float] = None
    generation_score: Optional[float] = None
    fused_feature: Optional[List[float]] = None
    image_size: Optional[List[int]] = None
    feature_grid_shape: Optional[List[int]] = None
    region_token_count: int = 0
    box_in_feature_coordinates: Optional[List[float]] = None
    source_frame: str = ""
    model_commit: str = ""
    checkpoint_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_index": self.object_index,
            "box_xyxy": self.box_xyxy,
            "normalized_box": self.normalized_box,
            "query_text": self.query_text,
            "semantic_label": self.semantic_label,
            "decode_mode": self.decode_mode,
            "block_start": self.block_start,
            "block_end": self.block_end,
            "pbd_box_end_feature": self.pbd_box_end_feature,
            "pbd_coordinate_mean_feature": self.pbd_coordinate_mean_feature,
            "pbd_full_block_mean_feature": self.pbd_full_block_mean_feature,
            "pbd_box_end_penultimate_feature": self.pbd_box_end_penultimate_feature,
            "pbd_coordinate_mean_penultimate_feature": self.pbd_coordinate_mean_penultimate_feature,
            "pbd_full_block_mean_penultimate_feature": self.pbd_full_block_mean_penultimate_feature,
            "region_feature": self.region_feature,
            "geometry_feature": self.geometry_feature,
            "confidence_feature": self.confidence_feature,
            "generation_score": self.generation_score,
            "fused_feature": self.fused_feature,
            "image_size": self.image_size,
            "feature_grid_shape": self.feature_grid_shape,
            "region_token_count": self.region_token_count,
            "box_in_feature_coordinates": self.box_in_feature_coordinates,
            "source_frame": self.source_frame,
            "model_commit": self.model_commit,
            "checkpoint_hash": self.checkpoint_hash,
            "extra": self.extra,
        }
