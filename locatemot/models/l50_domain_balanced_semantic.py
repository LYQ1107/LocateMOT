"""L50 semantic matcher with the L49 pooled architecture kept unchanged.

L50 deliberately changes no representation or network capacity.  The only
new behavior lives in the train/evaluation contract: domain/video-balanced
sampling, temporal feature augmentation, and L29 rank-preserving losses.
"""
from __future__ import annotations

from locatemot.models.l48_joint_rmot import L48SemanticMatcher


class L50DomainBalancedSemanticMatcher(L48SemanticMatcher):
    """Exact L48/L49 semantic core, versioned for the L50 experiment."""

    def config(self) -> dict:
        config = super().config()
        config.update({
            "format": "locatemot-l50-domain-balanced-semantic-v1",
            "architecture_unchanged_from": "L48SemanticMatcher/L49 semantic core",
            "hidden": self.hidden,
            "heads": 4,
            "training_changes_only": [
                "domain_balanced_video_balanced_sampling",
                "video_level_temporal_feature_augmentation",
                "L29_teacher_rank_preservation",
            ],
            "pooled_inputs": [
                "clip_512", "history_clip_512", "geometry_7", "motion_8",
                "context_8", "lifecycle_8", "objectness_1", "relation_4",
                "word_tokens_768",
            ],
            "new_backbone": False,
            "local_visual_stream": False,
            "identity_sequence_heads": False,
            "semantic_inputs_excluded": [
                "source_id", "pool_id", "group_id", "state_key", "query_id_as_feature",
            ],
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_language_mask": "UNALIGNED/not claimed",
            "rmot_only": True,
            "ordinary_mot_ovmot_imported": False,
        })
        return config
