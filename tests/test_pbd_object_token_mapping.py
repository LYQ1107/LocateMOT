"""Stub-event tests: accepted coordinate blocks produce exactly one token."""
from locatemot.models.object_tokens.pbd_extractor import PBDObjectExtractor
from locatemot.models.object_tokens.types import GenerationBlockEvent


def _hidden_slices():
    return [
        {35: __import__("torch").zeros((1, 6, 8)), 34: __import__("torch").zeros((1, 6, 8))},
    ]


def test_accepted_block_count_equals_tokens():
    ev = GenerationBlockEvent(
        generation_step=1,
        decode_mode="MTP/PBD accepted",
        attempted_mode="MTP",
        accepted_mode="MTP",
        block_type="coord_box",
        block_start_position=10,
        block_end_position=15,
        hidden_state_positions=list(range(10, 16)),
        logits_positions=list(range(10, 16)),
        parsed_box=[1.0, 2.0, 3.0, 4.0],
        normalized_box=[0.001, 0.002, 0.003, 0.004],
        accepted=True,
        output_order=0,
    )
    feats = PBDObjectExtractor().extract([ev], _hidden_slices())
    assert len(feats) == 1
    assert feats[0]["box_end"].shape == (8,)
    assert feats[0]["coord_mean"].shape == (8,)
    assert feats[0]["full_mean"].shape == (8,)


def test_rejected_and_point_blocks_do_not_produce_tokens():
    rejected = GenerationBlockEvent(
        generation_step=1, decode_mode="MTP rejected", attempted_mode="MTP",
        block_type="error_box", accepted=False, rejection_reason="bad",
        hidden_state_positions=[0, 1, 2], logits_positions=[0, 1, 2],
    )
    point = GenerationBlockEvent(
        generation_step=1, decode_mode="point", attempted_mode="MTP",
        block_type="point_box", accepted=False,
        hidden_state_positions=[0, 1, 2, 3], logits_positions=[0, 1, 2, 3],
    )
    feats = PBDObjectExtractor().extract([rejected, point], _hidden_slices())
    assert len(feats) == 0
