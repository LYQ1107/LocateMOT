from __future__ import annotations

import types
import unittest

import torch

from locatemot.groundjev.coupled.causal_memory import (
    build_causal_memory_input,
    causal_prefill_with_memory,
)


class FakeLanguageModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, input_ids, inputs_embeds, position_ids, attention_mask,
                 past_key_values, use_cache, cache_position, return_dict):
        del input_ids, position_ids, attention_mask, past_key_values, use_cache, cache_position, return_dict
        self.calls += 1
        return types.SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values=object())


class CausalMemoryPrefillContract(unittest.TestCase):
    def test_enabled_memory_extends_terminal_prompt_and_positions(self):
        prompt = torch.randn(1, 4, 6)
        mask = torch.ones(1, 4, dtype=torch.long)
        positions = torch.arange(4, dtype=torch.long).view(1, 1, 4)
        memory = torch.full((1, 2, 6), 3.0)
        result = build_causal_memory_input(prompt, mask, positions, memory)
        self.assertEqual(tuple(result.inputs_embeds.shape), (1, 6, 6))
        self.assertEqual(tuple(result.position_ids.shape), (1, 1, 6))
        self.assertTrue(torch.equal(result.inputs_embeds[:, :4], prompt))
        self.assertTrue(torch.equal(result.position_ids[0, 0], torch.arange(6)))
        self.assertEqual(result.insertion_index, 4)
        self.assertEqual(result.inserted_tokens, 2)

    def test_disabled_path_is_tensor_identical(self):
        prompt = torch.randn(1, 4, 6)
        mask = torch.ones(1, 4, dtype=torch.long)
        positions = torch.arange(4, dtype=torch.long).view(1, 1, 4)
        memory = torch.randn(1, 2, 6)
        result = build_causal_memory_input(prompt, mask, positions, memory, enabled=False)
        self.assertIs(result.inputs_embeds, prompt)
        self.assertIs(result.attention_mask, mask)
        self.assertIs(result.position_ids, positions)
        self.assertFalse(result.memory_enabled)

    def test_memory_changes_next_token_logits_with_one_prefill(self):
        prompt = torch.zeros(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.long)
        positions = torch.arange(3, dtype=torch.long).view(1, 1, 3)
        memory = torch.full((1, 1, 4), 5.0)
        language_model = FakeLanguageModel()
        lm_head = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]))
        baseline = causal_prefill_with_memory(
            language_model, lm_head, prompt, mask, positions,
            memory, enabled=False,
        )
        conditioned = causal_prefill_with_memory(
            language_model, lm_head, prompt, mask, positions,
            memory, enabled=True,
        )
        self.assertEqual(language_model.calls, 2)
        self.assertFalse(torch.equal(baseline["logits"], conditioned["logits"]))
        self.assertEqual(conditioned["prefill_calls"], 1)
        self.assertTrue(conditioned["single_backbone"])
        self.assertFalse(conditioned["second_vlm_loaded"])


if __name__ == "__main__":
    unittest.main()
