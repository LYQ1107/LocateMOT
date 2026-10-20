from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tools.groundjev.coupled.evaluate_fixed_candidate_clip import CachedNativeProvider


class FixedCandidateContract(unittest.TestCase):
  def test_cached_native_provider_preserves_registered_geometry(self):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
      tmp_path = Path(temp)
      clip = tmp_path / "clip"
      feature_dir = clip / "native_region_features"
      feature_dir.mkdir(parents=True)
      boxes = [[1.0, 2.0, 8.0, 9.0], [10.0, 12.0, 17.0, 22.0]]
      np.savez(feature_dir / "regions_00000001.npz", regions=np.zeros((2, 2560), dtype=np.float32))
      rows = [{
          "frame": 1,
          "image_path": str(tmp_path / "000001.jpg"),
          "candidates": [{"box": box, "label": "person"} for box in boxes],
          "native_region_features_npz": "native_region_features/regions_00000001.npz",
      }]
      provider = CachedNativeProvider(clip, rows, torch.device("cpu"))
      output = provider.encode_frame(None, boxes, image_key=rows[0]["image_path"])
      self.assertEqual(tuple(output["region_tokens"].shape), (2, 2560))
      self.assertEqual(provider.forward_count, 1)
      provider = CachedNativeProvider(clip, rows, torch.device("cpu"))
      with self.assertRaisesRegex(ValueError, "candidate geometry changed"):
          provider.encode_frame(None, [[0, 0, 1, 1], boxes[1]], image_key=rows[0]["image_path"])


if __name__ == "__main__":
    unittest.main()
