"""Parallel object queries over the full native spatial grid.

Set prediction/matching follows DETR's public design (Apache-2.0):
https://github.com/facebookresearch/detr . Uses standard PyTorch attention;
no new CUDA extension or autoregressive generation.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn


def position_2d(height, width, dim, device, dtype):
    y = (torch.arange(height, device=device) + .5) / height * 2 * math.pi
    x = (torch.arange(width, device=device) + .5) / width * 2 * math.pi
    frequency = 10000 ** (torch.arange(dim // 4, device=device) / (dim // 4))
    y, x = y[:, None] / frequency, x[:, None] / frequency
    py = torch.cat((y.sin(), y.cos()), -1)[:, None].expand(-1, width, -1)
    px = torch.cat((x.sin(), x.cos()), -1)[None].expand(height, -1, -1)
    return torch.cat((py, px), -1).reshape(1, height * width, dim).to(dtype)


class ObjectDecoder(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=256, queries=128, layers=3,
                 spatial_attention_sigma=None):
        super().__init__()
        self.spatial_attention_sigma = spatial_attention_sigma
        self.projection = nn.Conv2d(input_dim, hidden_dim, 1)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.queries = nn.Embedding(queries, hidden_dim)
        layer = nn.TransformerDecoderLayer(hidden_dim, 8, hidden_dim * 4,
                                           dropout=0., batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, 4))
        self.objectness = nn.Linear(hidden_dim, 1)
        # Image-independent learnable reference boxes, never GT-based queries.
        side = math.ceil(math.sqrt(queries))
        centers = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side),
                                             indexing="ij"), -1).reshape(-1, 2)[:queries]
        anchors = torch.cat(((centers[:, [1, 0]] + .5) / side,
                             torch.full((queries, 2), .12)), -1)
        self.reference_logits = nn.Parameter(torch.logit(anchors.clamp(.001, .999)))
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)
        nn.init.constant_(self.objectness.bias, -2.)

    def forward(self, spatial, valid_ratio=(1., 1.)):
        memory = self.projection(spatial.float())
        b, c, h, w = memory.shape
        memory = self.memory_norm(memory.flatten(2).transpose(1, 2))
        position = position_2d(h, w, c, memory.device, memory.dtype)
        # Bottom/right processor padding is not image content.
        yy, xx = torch.meshgrid((torch.arange(h, device=memory.device) + .5) / h,
                                (torch.arange(w, device=memory.device) + .5) / w,
                                indexing="ij")
        mask = ((xx >= valid_ratio[0]) | (yy >= valid_ratio[1])).flatten()[None].expand(b, -1)
        attention_bias = None
        padding_mask = mask
        if self.spatial_attention_sigma is not None:
            # A data-independent spatial prior ties each object query to its
            # own learned reference center. The centers are model parameters,
            # never GT/proposals. This is ordinary attention-mask arithmetic.
            centers = self.reference_logits[:, :2].sigmoid()
            dx = xx.flatten()[None] - centers[:, 0:1]
            dy = yy.flatten()[None] - centers[:, 1:2]
            attention_bias = -(dx.square() + dy.square()) / (2*self.spatial_attention_sigma**2)
            attention_bias = attention_bias.masked_fill(mask[0:1], float('-inf'))
            padding_mask = None
        query = self.queries.weight[None].expand(b, -1, -1)
        outputs = []
        for layer in self.layers:
            query = layer(query, memory + position, memory_mask=attention_bias,
                          memory_key_padding_mask=padding_mask)
            tokens = self.norm(query)
            outputs.append({"boxes": (self.box_head(tokens) + self.reference_logits).sigmoid(),
                            "logits": self.objectness(tokens).squeeze(-1),
                            "visual_tokens": tokens})
        return {**outputs[-1], "auxiliary": outputs[:-1]}
