"""Batch isolation: region extractor must not read tokens across images."""
import torch

from locatemot.models.object_tokens.region_extractor import MoonViTRegionExtractor


class FakeVisionModel:
    merge_kernel_size = (2, 2)


class FakeModel:
    def __init__(self):
        self.vision_model = FakeVisionModel()

    def extract_feature(self, pixel_values, image_grid_hws):
        # two images: image0 all ones, image1 all twos
        f0 = torch.ones((4, 4), dtype=torch.float32)
        f1 = torch.full((4, 4), 2.0, dtype=torch.float32)
        return [f0, f1]


def test_region_extract_does_not_cross_images():
    ex = MoonViTRegionExtractor(FakeModel())
    pixel = torch.zeros(1)
    grid = torch.tensor([[4, 4], [4, 4]], dtype=torch.int32)
    r0 = ex.extract(pixel, grid, 0, [[0, 0, 1000, 1000]])[0]
    r1 = ex.extract(pixel, grid, 1, [[0, 0, 1000, 1000]])[0]
    assert torch.allclose(r0["region_feature"], torch.ones(4))
    assert torch.allclose(r1["region_feature"], torch.full((4,), 2.0))
