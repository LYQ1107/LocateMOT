from __future__ import annotations

import unittest

import torch

from locatemot.groundjev.coupled.memory_tokens import (
    IdentityBoxStateTokenProjector,
    IdentityMemoryTokenProjector,
    inject_identity_memory,
)


class MemoryConditionedInput(unittest.TestCase):
    def test_disabled_path_is_exact_baseline(self):
        prompt = torch.randn(2, 7, 8)
        mask = torch.ones(2, 7, dtype=torch.long)
        memory = torch.randn(3, 8)
        result = inject_identity_memory(prompt, mask, memory, answer_start=5, enabled=False)
        assert result.inputs_embeds is prompt
        assert result.attention_mask is mask
        assert result.inserted_tokens == 0
        assert result.memory_enabled is False

    def test_memory_is_inserted_before_answer_and_preserves_causality(self):
        prompt = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
        mask = torch.ones(2, 6, dtype=torch.long)
        memory = torch.full((2, 3, 4), 7.0, requires_grad=True)
        result = inject_identity_memory(prompt, mask, memory, answer_start=2)
        assert result.inputs_embeds.shape == (2, 9, 4)
        assert result.attention_mask.shape == (2, 9)
        assert torch.equal(result.inputs_embeds[:, :2], prompt[:, :2])
        assert torch.equal(result.inputs_embeds[:, 2:5], memory)
        assert torch.equal(result.inputs_embeds[:, 5:], prompt[:, 2:])
        assert result.insertion_index == 2
        result.inputs_embeds.square().mean().backward()
        assert memory.grad is not None and torch.isfinite(memory.grad).all()

    def test_projector_to_prompt_contract(self):
        projector = IdentityMemoryTokenProjector(4, 8, hidden_dim=6)
        memories = {2: torch.randn(4), 7: torch.randn(4)}
        tokens = projector(memories, ordered_ids=[7, 2])
        prompt = torch.randn(1, 5, 8)
        mask = torch.ones(1, 5, dtype=torch.long)
        result = inject_identity_memory(prompt, mask, tokens, answer_start=4)
        assert result.inserted_tokens == 2
        assert result.inputs_embeds.shape == (1, 7, 8)
        assert torch.allclose(result.inputs_embeds[0, 4:6], tokens)

    def test_box_state_projector_preserves_registered_order_and_gradients(self):
        projector = IdentityBoxStateTokenProjector(4, 8, hidden_dim=6)
        memories = {2: torch.randn(4), 7: torch.randn(4)}
        states = {2: torch.tensor([0.1, 0.2, 0.3, 0.4]), 7: torch.tensor([0.5, 0.6, 0.7, 0.8])}
        tokens = projector(memories, states, ordered_ids=[7, 2])
        assert tokens.shape == (2, 8)
        tokens.square().mean().backward()
        assert all(parameter.grad is not None for parameter in projector.parameters())


if __name__ == "__main__":
    unittest.main()
