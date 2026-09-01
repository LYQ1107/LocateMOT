"""Tests for MoonViT box-to-grid mapping using a fake model."""
import numpy as np
import torch

from locatemot.models.object_tokens.region_extractor import MoonViTRegionExtractor


class FakeVisionModel:
    merge_kernel_size = (2, 2)


class FakeModel:
    def __init__(self):
        self.vision_model = FakeVisionModel()

    def extract_feature(self, pixel_values, image_grid_hws):
        return [torch.zeros((4, 4), dtype=torch.float32)]


def test_region_mapping_shape_and_coords():
    extractor = MoonViTRegionExtractor(FakeModel())
    pixel = torch.zeros(1)
    grid = torch.tensor([[4, 4]], dtype=torch.int32)  # pre-merge grid 4x4 -> feature 2x2
    boxes = [[0, 0, 500, 1000]]  # left column of 2x2 feature grid
    res = extractor.extract(pixel, grid, 0, boxes)[0]
    assert res["feature_grid_shape"] == [2, 2]
    assert res["region_token_count"] == 2
    assert res["box_in_feature_coordinates"] == [0, 0, 1, 2]
    assert res["region_feature"].shape == (4,)


def test_region_mapping_full_box():
    extractor = MoonViTRegionExtractor(FakeModel())
    pixel = torch.zeros(1)
    grid = torch.tensor([[4, 4]], dtype=torch.int32)
    boxes = [[0, 0, 1000, 1000]]
    res = extractor.extract(pixel, grid, 0, boxes)[0]
    assert res["region_token_count"] == 4
    assert res["box_in_feature_coordinates"] == [0, 0, 2, 2]
