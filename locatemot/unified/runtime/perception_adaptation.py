"""Checkpoint-compatible GroundingDINO with frozen U1 BERT for bounded U2-P."""
from __future__ import annotations

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS


@MODELS.register_module()
class U2PerceptionGroundingDINO(GroundingDINO):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.language_model.requires_grad_(False)
        self.language_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.language_model.eval()
        return self
