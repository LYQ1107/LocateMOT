"""UnifiedSpecTrack composition root for LocateMOT-U."""
from __future__ import annotations

import torch
from torch import nn

from .cue_fusion import CueFusion
from .foundation import SharedVisualFoundation
from .motion_reasoner import MotionReasoner
from .perception_decoder import UniversalPerceptionDecoder
from .relation_reasoner import RelationReasoner
from .semantic_router import SemanticRouter
from .spec_encoder import SpecificationEncoder
from .task_heads import TaskHeads
from .track_decoder import UniversalTrackDecoder


class UnifiedSpecTrack(nn.Module):
    """One shared foundation, perception path and track decoder for all tasks."""

    task_to_id = {name: index for index, name in enumerate(("mot", "ovmot", "rmot", "point", "box", "mask", "grounding"))}

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.foundation = SharedVisualFoundation()
        self.spec_encoder = SpecificationEncoder(hidden_dim=hidden_dim)
        self.perception = UniversalPerceptionDecoder(hidden_dim=hidden_dim)
        self.track_decoder = UniversalTrackDecoder(hidden_dim=hidden_dim)
        self.router = SemanticRouter(hidden_dim=hidden_dim)
        self.motion = MotionReasoner(hidden_dim=hidden_dim)
        self.relation = RelationReasoner(hidden_dim=hidden_dim)
        self.fusion = CueFusion(hidden_dim=hidden_dim)
        self.heads = TaskHeads(hidden_dim=hidden_dim)

    def forward(self, frames: torch.Tensor, task_type: str, specification: torch.Tensor, online_state: object | None = None) -> dict[str, object]:
        if task_type not in self.task_to_id:
            raise KeyError(task_type)
        if specification.ndim != 2:
            raise ValueError("specification must be token ids [B,L]")
        task_id = torch.full((frames.shape[0],), self.task_to_id[task_type], device=frames.device, dtype=torch.long)
        visual = self.foundation(frames)
        spec = self.spec_encoder(specification, task_id)
        perception = self.perception(visual["frame_features"][:, -1], spec["spec_global"])
        tracks = self.track_decoder(perception["object_queries"], perception["frame_context"])
        routed = self.router(spec["spec_tokens"])
        motion = self.motion(tracks["track_queries"], routed["motion_spec"])
        relation = self.relation(tracks["track_queries"], routed["relation_spec"])
        fused = self.fusion(tracks["track_queries"], motion, relation)
        task_score = self.heads(fused, task_type)
        return {"tracks": tracks | {"task_score": task_score}, "task_output": task_score, "online_state": online_state, "spec": routed}
