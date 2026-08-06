"""Hybrid fallback: only the accepted AR-finalized box becomes a token."""
import torch

from locatemot.models.object_tokens.pbd_extractor import PBDObjectExtractor
from locatemot.models.object_tokens.types import GenerationBlockEvent


def test_hybrid_fallback_finalizes_one_accepted_box():
    rejected = GenerationBlockEvent(
        generation_step=2, decode_mode="MTP rejected", attempted_mode="MTP",
        block_type="error_box", accepted=False, rejection_reason="fallback",
        hidden_state_positions=[10, 11, 12], logits_positions=[10, 11, 12],
    )
    accepted = GenerationBlockEvent(
        generation_step=5, decode_mode="AR/NTP", attempted_mode="NTP",
        accepted_mode="NTP", block_type="coord_box", accepted=True,
        block_start_position=10, block_end_position=15,
        hidden_state_positions=[10, 11, 12, 12, 13, 14],
        logits_positions=[10, 11, 12, 13, 14, 15],
        parsed_box=[100.0, 200.0, 300.0, 400.0],
        normalized_box=[0.1, 0.2, 0.3, 0.4],
        output_order=0,
        extra={"token_steps": [2, 2, 2, 3, 4, 5], "hidden_rel_positions": [0, 1, 2, 0, 0, 0]},
    )
    hidden = [
        {35: torch.zeros((1, 6, 8)), 34: torch.zeros((1, 6, 8))},  # step1 unused
        {35: torch.zeros((1, 6, 8)), 34: torch.zeros((1, 6, 8))},  # step2 MTP
        {35: torch.zeros((1, 1, 8)), 34: torch.zeros((1, 1, 8))},  # step3 AR
        {35: torch.zeros((1, 1, 8)), 34: torch.zeros((1, 1, 8))},  # step4 AR
        {35: torch.zeros((1, 1, 8)), 34: torch.zeros((1, 1, 8))},  # step5 AR
    ]
    feats = PBDObjectExtractor().extract([rejected, accepted], hidden)
    assert len(feats) == 1
    assert feats[0]["event"].parsed_box == [100.0, 200.0, 300.0, 400.0]
