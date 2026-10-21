from __future__ import annotations

import unittest

import torch

from locatemot.groundjev.coupled.official_dlm_memory import (
    OfficialDLMIdentityMemory,
    append_memory_markers,
)


class _FakeModel:
    def _embed_clean(self, input_ids, *args, **kwargs):
        del args, kwargs
        return input_ids.float().unsqueeze(-1).expand(-1, -1, 4), torch.arange(
            input_ids.shape[1]
        ).view(1, 1, -1)


class OfficialDLMIdentityMemoryContract(unittest.TestCase):
    def test_marker_append_preserves_mask_and_types(self):
        ids = torch.tensor([[3, 4]])
        mask = torch.ones_like(ids)
        types = torch.zeros_like(ids)
        out_ids, out_mask, out_types = append_memory_markers(
            ids, mask, types, marker_token_id=99, memory_count=2
        )
        self.assertEqual(out_ids.tolist(), [[3, 4, 99, 99]])
        self.assertEqual(out_mask.tolist(), [[1, 1, 1, 1]])
        self.assertEqual(out_types.tolist(), [[0, 0, 0, 0]])

    def test_injected_tokens_are_visible_and_method_restores(self):
        model = _FakeModel()
        baseline, _ = model._embed_clean(torch.tensor([[3, 4, 99, 99]]))
        memory = torch.randn(2, 4, requires_grad=True)
        with OfficialDLMIdentityMemory(model, marker_token_id=99, memory_tokens=memory):
            injected, positions = model._embed_clean(torch.tensor([[3, 4, 99, 99]]))
            self.assertTrue(torch.equal(injected[0, :2], baseline[0, :2]))
            self.assertTrue(torch.allclose(injected[0, 2:], memory))
            self.assertEqual(tuple(positions.shape), (1, 1, 4))
            injected.square().mean().backward()
        self.assertIsNotNone(memory.grad)
        self.assertTrue(torch.isfinite(memory.grad).all())
        restored, _ = model._embed_clean(torch.tensor([[3, 4, 99, 99]]))
        self.assertTrue(torch.equal(restored, baseline))

    def test_marker_order_is_required(self):
        model = _FakeModel()
        with OfficialDLMIdentityMemory(model, marker_token_id=99, memory_tokens=torch.randn(1, 4)):
            with self.assertRaises(ValueError):
                model._embed_clean(torch.tensor([[3, 99, 4]]))


if __name__ == "__main__":
    unittest.main()
