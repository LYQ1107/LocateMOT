"""L78 full-frame spatial ROI/set correspondence model.

The frozen CLIP spatial feature map is passed in for the current image and
candidate boxes are pooled inside this module with differentiable
``grid_sample``.  Only the small correspondence/set adapter is trainable.
No candidate row is removed or ranked before the set block.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class L78FullFrameROISet(nn.Module):
    def __init__(self, visual_dim: int = 512, text_dim: int = 512,
                 hidden: int = 128, heads: int = 4, roi_grid: int = 4) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if roi_grid < 1:
            raise ValueError("roi_grid must be positive")
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.roi_grid = int(roi_grid)
        self.visual_norm = nn.LayerNorm(self.visual_dim)
        self.visual_proj = nn.Linear(self.visual_dim, self.hidden)
        self.text_norm = nn.LayerNorm(self.text_dim)
        self.text_proj = nn.Linear(self.text_dim, self.hidden)
        self.global_norm = nn.LayerNorm(self.visual_dim)
        self.global_proj = nn.Linear(self.visual_dim, self.hidden)
        self.query_to_roi = nn.MultiheadAttention(
            self.hidden, self.heads, dropout=0.0, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.hidden)
        set_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=self.heads,
            dim_feedforward=self.hidden * 2, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.set_competition = nn.TransformerEncoder(set_layer, num_layers=1)
        self.set_norm = nn.LayerNorm(self.hidden)
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.GELU(),
            nn.LayerNorm(self.hidden), nn.Linear(self.hidden, 1)
        )
        self.absent_head = nn.Sequential(
            nn.LayerNorm(self.hidden * 2), nn.Linear(self.hidden * 2, self.hidden),
            nn.GELU(), nn.Linear(self.hidden, 1)
        )

    @staticmethod
    def _check_boxes(boxes: Tensor) -> None:
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] == 0:
            raise ValueError(f"boxes must be [N,4], got {tuple(boxes.shape)}")
        if not torch.isfinite(boxes).all():
            raise ValueError("nonfinite normalized boxes")
        if bool((boxes < 0).any()) or bool((boxes > 1).any()):
            raise ValueError("normalized boxes outside [0,1]")
        if bool((boxes[:, 2:] < boxes[:, :2]).any()):
            raise ValueError("inverted normalized boxes")

    def roi_tokens(self, spatial_map: Tensor, boxes: Tensor) -> Tensor:
        """Pool [1,C,H,W] at every normalized [N,4] box into [N,K,C]."""
        if spatial_map.ndim != 4 or spatial_map.shape[0] != 1:
            raise ValueError(f"spatial_map must be [1,C,H,W], got {tuple(spatial_map.shape)}")
        if spatial_map.shape[1] != self.visual_dim:
            raise ValueError(f"spatial map channel mismatch {spatial_map.shape[1]} != {self.visual_dim}")
        self._check_boxes(boxes)
        n = boxes.shape[0]
        frac = (torch.arange(self.roi_grid, device=boxes.device, dtype=boxes.dtype) + 0.5) / self.roi_grid
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x = x1[:, None] + (x2 - x1)[:, None] * frac[None, :]
        y = y1[:, None] + (y2 - y1)[:, None] * frac[None, :]
        grid_x = x[:, None, :].expand(-1, self.roi_grid, -1)
        grid_y = y[:, :, None].expand(-1, -1, self.roi_grid)
        # align_corners=False: normalized coordinate -1/1 denotes the outside
        # edge convention; the fixed box-center lattice above is in [0,1].
        grid = torch.stack((grid_x * 2.0 - 1.0, grid_y * 2.0 - 1.0), dim=-1)
        source = spatial_map.expand(n, -1, -1, -1)
        pooled = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
        tokens = pooled.flatten(2).transpose(1, 2)
        if tokens.shape != (n, self.roi_grid * self.roi_grid, self.visual_dim):
            raise AssertionError(f"ROI token shape drift: {tuple(tokens.shape)}")
        return tokens

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor, dim: int) -> Tensor:
        weight = mask.to(value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        denom = weight.sum(dim=dim).clamp_min(1.0)
        return (value * weight).sum(dim=dim) / denom

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        spatial_map = batch["spatial_map"]
        global_token = batch["global_token"]
        text = batch["text"]
        text_mask = batch["text_mask"].bool()
        boxes = batch["boxes"]
        if text.ndim != 2 or text.shape[1] != self.text_dim:
            raise ValueError(f"text must be [T,{self.text_dim}], got {tuple(text.shape)}")
        if text_mask.shape != text.shape[:1] or not bool(text_mask.any()):
            raise ValueError("text mask shape/validity contract failed")
        if global_token.ndim != 1 or global_token.shape[0] != self.visual_dim:
            raise ValueError(f"global token must be [{self.visual_dim}]")
        if not torch.isfinite(spatial_map).all() or not torch.isfinite(global_token).all() or not torch.isfinite(text).all():
            raise ValueError("nonfinite L78 input")
        roi = self.roi_tokens(spatial_map, boxes)
        roi_hidden = self.visual_proj(self.visual_norm(roi))
        words = self.text_proj(self.text_norm(text)).unsqueeze(0).expand(roi.shape[0], -1, -1)
        query = words
        key_value = roi_hidden
        candidate_from_words, attention = self.query_to_roi(
            query, key_value, key_value, need_weights=True,
            average_attn_weights=False,
        )
        text_valid = text_mask.unsqueeze(0).expand(roi.shape[0], -1)
        candidate = self._masked_mean(candidate_from_words, text_valid, dim=1)
        candidate = self.cross_norm(candidate + roi_hidden.mean(dim=1))
        g = self.global_proj(self.global_norm(global_token)).unsqueeze(0)
        candidate = candidate + g
        # Complete current-frame set, with all candidate rows in one shared block.
        set_tokens = self.set_competition(candidate.unsqueeze(0)).squeeze(0)
        set_tokens = self.set_norm(set_tokens + candidate)
        raw_score = self.score_head(set_tokens).squeeze(-1)
        match_logits = 2.0 * torch.tanh(raw_score / 2.0)
        query_vector = F.normalize(self._masked_mean(words[0], text_mask, dim=0), dim=-1)
        set_summary = set_tokens.mean(dim=0)
        absent_logit = self.absent_head(torch.cat((query_vector, set_summary), dim=-1)).reshape(())
        if match_logits.shape != (boxes.shape[0],) or not torch.isfinite(match_logits).all():
            raise AssertionError("L78 score shape/finite contract failed")
        return {
            "match_logits": match_logits,
            "raw_score": raw_score,
            "absent_logit": absent_logit,
            "roi_tokens": roi,
            "candidate_tokens": candidate,
            "set_tokens": set_tokens,
            "query_vector": query_vector,
            "cross_attention": attention,
        }

    def parameter_summary(self) -> dict[str, int | float | str]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "visual_dim": self.visual_dim,
            "text_dim": self.text_dim,
            "hidden": self.hidden,
            "heads": self.heads,
            "roi_grid": self.roi_grid,
            "roi_tokens_per_candidate": self.roi_grid * self.roi_grid,
            "bounded_logit": "2*tanh(raw_score/2)",
            "trainable_scope": "L78 adapter/set head only; CLIP is external frozen input",
        }
