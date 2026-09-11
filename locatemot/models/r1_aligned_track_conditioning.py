"""R1 aligned track-conditioning sidecar.

The module is deliberately independent of the production LocateMOT entry
points.  ``FrozenL89EAnchor`` is an exact, frozen copy of the selected L89E
Stage-S model; the R1 modules only produce residuals from query-conditioned
Z0/Z1/Z4 states and causal observation features.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
RULE_NAME = "B"
ANCHOR_SHA256 = "5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_l89e_rule(selection_root: str | Path) -> dict[str, Any]:
    """Load the one authoritative L89E Rule-B object.

    All inference, loss and emission callers use this helper rather than
    copying thresholds into a second configuration file.
    """
    root = Path(selection_root).resolve()
    path = root / "checkpoint_selection.json" if root.is_dir() else root
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("final_selection")
    if payload.get("status") != "complete" or not isinstance(selected, dict):
        raise ValueError(f"invalid L89E selection artifact: {path}")
    if str(selected.get("rule")) != RULE_NAME:
        raise ValueError(f"R1 requires frozen Rule-B, got {selected.get('rule')}")
    rule = {
        "rule": RULE_NAME,
        "candidate_threshold": float(selected["rule_object"]["candidate_threshold"]),
        "presence_threshold": float(selected["rule_object"]["presence_threshold"]),
        "null_margin": float(selected["rule_object"]["null_margin"]),
        "selection_path": str(path),
        "selection_sha256": sha256_file(path),
        "checkpoint": dict(selected["checkpoint_info"]),
    }
    if not all(torch.isfinite(torch.tensor(rule[key])) for key in
               ("candidate_threshold", "presence_threshold", "null_margin")):
        raise ValueError("nonfinite frozen L89E rule")
    return rule


@dataclass(frozen=True)
class R1Config:
    dim: int = 256
    heads: int = 8
    stage_layers: int = 1
    detail_layers: int = 2
    set_layers: int = 2
    ffn_dim: int = 1024
    dropout: float = 0.10
    raw_visual_tokens: int = 72
    geometry_steps: int = 5
    geometry_dim: int = 10
    set_margin: float = 0.50
    duplicate_weight: float = 0.25
    presence_weight: float = 0.50
    residual_reg_weight: float = 0.01
    # The frozen Stage-S anchor already provides the registered presence
    # signal.  The R1 plan enables a presence residual only if the anchor's
    # legal-dev presence-only miss is >= .05; the frozen epoch-004 audit is
    # below that threshold, so the formal R1 configuration is explicitly off.
    presence_residual: bool = False


class FrozenL89EAnchor(nn.Module):
    """Exact frozen L89E Stage-S anchor with no optimizer-visible params."""

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        package = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(package, dict) or not isinstance(package.get("model_config"), dict):
            raise ValueError("L89E package has no model_config")
        self.checkpoint = str(path)
        self.checkpoint_sha256 = sha256_file(path)
        if self.checkpoint_sha256 != ANCHOR_SHA256:
            raise AssertionError(f"L89E anchor SHA drift: {self.checkpoint_sha256}")
        self.model_config = dict(package["model_config"])
        self.model = L89FullRMOT(L89Config(**self.model_config))
        state = package.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("L89E package lacks model_state_dict")
        self.model.load_state_dict(state, strict=True)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(
        self,
        z1: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        text_global: torch.Tensor,
        frame_global: torch.Tensor,
        current_observation: torch.Tensor,
        history_observations: torch.Tensor,
        history_mask: torch.Tensor,
        history_frame_ids: torch.Tensor,
        cutoff_frame: int,
    ) -> dict[str, torch.Tensor]:
        if history_observations.shape[1] != 8:
            raise ValueError("R1 anchor requires history length 8")
        if bool(history_mask.any()) and bool((history_frame_ids[history_mask] > cutoff_frame).any()):
            raise AssertionError("future history entered frozen anchor")
        with torch.no_grad():
            output = self.model(
                z1, text_tokens, text_mask, text_global, frame_global,
                current_observation, history_observations, history_mask,
                history_frame_ids, cutoff_frame, temporal_enabled=False,
            )
        return {key: value.detach() for key, value in output.items()}


class R1StageMixer(nn.Module):
    def __init__(self, config: R1Config) -> None:
        super().__init__()
        d = config.dim
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, d))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, z0: torch.Tensor, z1: torch.Tensor, z4: torch.Tensor, text_global: torch.Tensor) -> torch.Tensor:
        if z0.shape != z1.shape or z4.shape != z1.shape or z1.ndim != 3 or text_global.shape != z1.shape[:1] + z1.shape[2:]:
            raise ValueError("R1 stage shape drift")
        stages = torch.stack((z0, z1, z4), dim=2)
        context = F.normalize(z1 + text_global[:, None, :], dim=-1)
        flat = (stages + context[:, :, None, :]).reshape(-1, 3, z1.shape[-1])
        normalized = self.norm(flat)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        mixed = flat + self.dropout(attended)
        mixed = mixed + self.dropout(self.ffn(self.ffn_norm(mixed)))
        return mixed.mean(dim=1).reshape(z1.shape)


class R1QueryConditionedDetailHook(nn.Module):
    """Exactly two text-token -> raw 72-token cross-attention blocks."""

    def __init__(self, config: R1Config) -> None:
        super().__init__()
        d = config.dim
        self.blocks = nn.ModuleList()
        for _ in range(config.detail_layers):
            self.blocks.append(nn.ModuleDict({
                "text_norm": nn.LayerNorm(d),
                "text_attn": nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True),
                "raw_norm": nn.LayerNorm(d),
                "raw_attn": nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True),
                "ffn_norm": nn.LayerNorm(d),
                "ffn": nn.Sequential(nn.Linear(d, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, d)),
                "dropout": nn.Dropout(config.dropout),
            }))

    def forward(self, candidates: torch.Tensor, text_tokens: torch.Tensor, text_mask: torch.Tensor, raw_tokens: torch.Tensor) -> torch.Tensor:
        q, n, d = candidates.shape
        if text_tokens.shape[0] != q or text_mask.shape != text_tokens.shape[:2] or raw_tokens.shape[:3] != (q, n, raw_tokens.shape[2]):
            raise ValueError("R1 detail query/candidate shape drift")
        if raw_tokens.shape[2] != 72 or raw_tokens.shape[-1] != d:
            raise ValueError("R1 raw token contract requires [Q,N,72,256]")
        x = candidates.reshape(q * n, 1, d)
        text = text_tokens[:, None].expand(q, n, text_tokens.shape[1], d).reshape(q * n, text_tokens.shape[1], d)
        text_valid = text_mask[:, None].expand(q, n, text_mask.shape[1]).reshape(q * n, text_mask.shape[1]).bool()
        raw = raw_tokens.reshape(q * n, 72, d)
        for block in self.blocks:
            y = block["text_norm"](x)
            attended, _ = block["text_attn"](y, text, text, key_padding_mask=~text_valid, need_weights=False)
            x = x + block["dropout"](attended)
            y = block["raw_norm"](x)
            attended, _ = block["raw_attn"](y, raw, raw, need_weights=False)
            x = x + block["dropout"](attended)
            x = x + block["dropout"](block["ffn"](block["ffn_norm"](x)))
        return x.reshape(q, n, d)


class R1GeometryEncoder(nn.Module):
    def __init__(self, config: R1Config) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(config.geometry_dim), nn.Linear(config.geometry_dim, 128), nn.GELU(), nn.Linear(128, config.dim))
        self.gru = nn.GRU(config.dim, config.dim, batch_first=True)

    def forward(self, geometry: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if geometry.ndim != 3 or geometry.shape[1:] != (5, 10):
            raise ValueError(f"R1 geometry must be [N,5,10], got {tuple(geometry.shape)}")
        x = self.proj(geometry.float())
        with torch.autocast(device_type=geometry.device.type, enabled=False):
            out, _ = self.gru(x.float())
        if mask is None:
            return out[:, -1]
        valid = mask.bool()
        if valid.shape != geometry.shape[:2]:
            raise ValueError("R1 geometry mask shape drift")
        indices = valid.long().sum(dim=1).clamp_min(1) - 1
        result = out[torch.arange(out.shape[0], device=out.device), indices]
        return result.masked_fill(~valid.any(dim=1, keepdim=True), 0.0)


class R1ResidualSetReasoner(nn.Module):
    def __init__(self, config: R1Config) -> None:
        super().__init__()
        d = config.dim
        self.input = nn.Sequential(nn.LayerNorm(d * 4), nn.Linear(d * 4, d), nn.GELU())
        layer = nn.TransformerEncoderLayer(d, config.heads, config.ffn_dim, config.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.set_layers)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, detail: torch.Tensor, stage: torch.Tensor, geometry: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        x = self.input(torch.cat((detail, stage, geometry, box), dim=-1))
        x = self.encoder(x)
        return self.out_norm(x)


class R1AlignedTrackConditioning(nn.Module):
    """Trainable R1 residual sidecar; the anchor is passed in as frozen outputs."""

    def __init__(self, config: R1Config | None = None, presence_residual: bool | None = None) -> None:
        super().__init__()
        self.config = config or R1Config()
        c = self.config
        self.stage_mixer = R1StageMixer(c)
        self.detail = R1QueryConditionedDetailHook(c)
        self.geometry = R1GeometryEncoder(c)
        self.box = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 128), nn.GELU(), nn.Linear(128, c.dim))
        self.reasoner = R1ResidualSetReasoner(c)
        self.delta_energy = nn.Linear(c.dim, 1)
        self.delta_presence = nn.Linear(c.dim, 1)
        nn.init.zeros_(self.delta_energy.weight); nn.init.zeros_(self.delta_energy.bias)
        nn.init.zeros_(self.delta_presence.weight); nn.init.zeros_(self.delta_presence.bias)
        self.presence_residual_enabled = bool(
            self.config.presence_residual if presence_residual is None else presence_residual
        )
        if not self.presence_residual_enabled:
            for parameter in self.delta_presence.parameters():
                parameter.requires_grad_(False)

    def parameter_report(self) -> dict[str, Any]:
        trainable = {name: int(value.numel()) for name, value in self.named_parameters() if value.requires_grad}
        return {"total": int(sum(value.numel() for value in self.parameters())), "trainable": int(sum(trainable.values())),
                "trainable_by_name": trainable, "config": asdict(self.config),
                "presence_residual_enabled": self.presence_residual_enabled}

    def forward(
        self, z0: torch.Tensor, z1: torch.Tensor, z4: torch.Tensor,
        text_tokens: torch.Tensor, text_mask: torch.Tensor, text_global: torch.Tensor,
        raw_tokens: torch.Tensor, geometry: torch.Tensor, geometry_mask: torch.Tensor,
        boxes_norm: torch.Tensor, base_candidate_energy: torch.Tensor,
        base_presence: torch.Tensor, base_null: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if z1.ndim != 3 or base_candidate_energy.shape != z1.shape[:2]:
            raise ValueError("R1 candidate/base shape drift")
        stage = self.stage_mixer(z0, z1, z4, text_global)
        detail = self.detail(z1, text_tokens, text_mask, raw_tokens)
        geom = self.geometry(geometry, geometry_mask)[None].expand(z1.shape[0], -1, -1)
        box = self.box(boxes_norm.float())[None].expand(z1.shape[0], -1, -1)
        reasoned = self.reasoner(detail, stage, geom, box)
        delta_energy = self.delta_energy(reasoned).squeeze(-1)
        delta_presence = self.delta_presence(reasoned).squeeze(-1).mean(dim=1)
        if not self.presence_residual_enabled:
            delta_presence = delta_presence * 0.0
        final_energy = base_candidate_energy + delta_energy
        final_presence = base_presence + delta_presence
        for value in (final_energy, final_presence, base_null, reasoned):
            if not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError("nonfinite R1 sidecar output")
        return {"final_energy": final_energy, "final_presence": final_presence, "final_null": base_null,
                "delta_energy": delta_energy, "delta_presence": delta_presence,
                "stage_state": stage, "detail_state": detail, "reasoned_state": reasoned}


def _smooth_max(value: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    return temperature * torch.logsumexp(value / temperature, dim=0)


def target_bag_loss(
    energy: torch.Tensor,
    candidate_gt: Iterable[str | None],
    target_ids: Iterable[str],
    category: str,
    *,
    margin: float = 0.50,
    duplicate_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Target-bag loss; every referred target bag and every current row remains visible."""
    if energy.ndim != 1:
        raise ValueError("target_bag_loss expects [N]")
    gt = [None if value is None else str(value) for value in candidate_gt]
    targets = [str(value) for value in target_ids]
    if len(gt) != int(energy.numel()):
        raise ValueError("candidate_gt/energy length drift")
    positive_rows = {index for index, value in enumerate(gt) if value in targets}
    active_covered = category not in {"inactive", "present_uncovered"} and bool(positive_rows)
    terms: list[torch.Tensor] = []
    bag_positive_count = 0
    duplicate_terms: list[torch.Tensor] = []
    if active_covered:
        for target in targets:
            indices = [index for index, value in enumerate(gt) if value == target]
            if not indices:
                continue
            bag_positive_count += 1
            scores = energy[torch.tensor(indices, device=energy.device)]
            winner = _smooth_max(scores)
            terms.append(F.softplus(-winner))
            if len(indices) > 1:
                for score in scores:
                    duplicate_terms.append(F.softplus(winner.detach() - score))
        negative = energy[torch.tensor([i for i in range(len(gt)) if i not in positive_rows], device=energy.device)]
        if negative.numel():
            terms.append(F.softplus(_smooth_max(negative)))
        positive = energy[torch.tensor(sorted(positive_rows), device=energy.device)]
        if positive.numel() and negative.numel():
            terms.append(F.softplus(margin - positive.min() + negative.max()))
    else:
        # Missing target is not changed to inactive in metadata, but no current
        # candidate can emit it.  The current candidate emission is therefore
        # explicitly all-negative while coverage is reported separately.
        terms.append(F.softplus(energy + margin).mean())
    if duplicate_terms:
        terms.append(float(duplicate_weight) * torch.stack(duplicate_terms).mean())
    loss = torch.stack(terms).mean() if terms else energy.sum() * 0.0
    stats = {"category": str(category), "active_covered": active_covered,
             "positive_row_count": len(positive_rows), "negative_row_count": len(gt) - len(positive_rows),
             "positive_bag_count": bag_positive_count, "duplicate_term_count": len(duplicate_terms),
             "finite": bool(torch.isfinite(loss.float()))}
    return loss, stats


__all__ = [
    "ANCHOR_SHA256", "FrozenL89EAnchor", "R1AlignedTrackConditioning", "R1Config", "R1GeometryEncoder",
    "R1QueryConditionedDetailHook", "R1ResidualSetReasoner", "R1StageMixer", "THREAD", "load_l89e_rule",
    "sha256_file", "target_bag_loss",
]
