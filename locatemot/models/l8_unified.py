"""Stage L8 Unified Observation / Specification Adapter.

Scientific claim: MOT / OVMOT / RMOT differ only in the target
specification (WHAT); the identity-dynamics process (HOW) is shared.
This module fuses identity-discriminative PBD evidence, open-vocabulary
CLIP evidence, and a specification embedding into one unified observation
token consumed by the same UIDM core, plus a lightweight relevance head
for language-driven target selection.

Design evidence: see reports/l8_literature_and_code_audit.md.  The gated
semantic residue (identity stream stays in the UIDM core) is a clean
reimplementation; no code is copied from iKUN / TransRMOT / MOTIP.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def l2norm(x, dim=-1):
    return F.normalize(x.float(), dim=dim, eps=1e-6)


class _Proj(nn.Module):
    """Two-layer MLP with LayerNorm (same style as UIDM PBDEncoder)."""

    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * d_model),
            nn.LayerNorm(2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model))

    def forward(self, x):
        return self.mlp(l2norm(x))


class UnifiedObservationAdapter(nn.Module):
    """CLIP crop + spec -> semantic residue token and relevance logit.

    The semantic residue is added to the UIDM's PBD-derived candidate token
    inside the core (cand_sem), so the observation consumed by the shared
    identity dynamics is z = PBD_token + gated(CLIP + spec).

    Modes (ablation):
      unified : residue = clip + gate*spec; identity stream kept
      identity: residue = 0; relevance from PBD projection only
      semantic: residue = clip + gate*spec; PBD stream zeroed
    """

    def __init__(self, d_model=320, pbd_dim=2048, clip_dim=512, spec_dim=512,
                 dropout=0.1, mode="unified"):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.pbd_proj = _Proj(pbd_dim, d_model, dropout)
        self.clip_proj = _Proj(clip_dim, d_model, dropout)
        self.spec_proj = _Proj(spec_dim, d_model, dropout)
        # per-dimension gate on the spec residue
        self.gate = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model))
        self.relevance = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.relevance[-1].weight)
        nn.init.zeros_(self.relevance[-1].bias)

    def forward(self, pbd, clip, spec):
        """Returns (semantic_residue [...,d], relevance_logit [...,1])."""
        if spec.dim() < clip.dim():
            spec = spec.unsqueeze(1)
        c = self.clip_proj(clip)
        s = self.spec_proj(spec)
        g = torch.sigmoid(self.gate(c))
        sem = c + g * s
        if self.mode == "identity":
            rel_in = self.pbd_proj(pbd)
            sem = torch.zeros_like(sem)
        else:
            rel_in = sem
        rel = self.relevance(rel_in).squeeze(-1)
        return sem, rel


class L8UnifiedUIDM(nn.Module):
    """Unified Observation Adapter in front of the shared UIDM core.

    The UIDM core is constructed with app_dim = d_model, so candidate
    appearance tokens are exactly the unified tokens z.
    """

    def __init__(self, d_model=320, n_layers=4, n_heads=8, ffn_dim=1280,
                 dropout=0.1, no_interaction=False, use_cue_rel=False,
                 pbd_dim=2048, clip_dim=512, spec_dim=512, mode="unified",
                 core=None):
        super().__init__()
        from locatemot.models.l6_uidm import UIDM
        self.d_model = d_model
        self.app_dim = pbd_dim
        self.uidm = core if core is not None else UIDM(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, dropout=dropout,
            no_interaction=no_interaction, use_cue_rel=use_cue_rel,
            app_dim=pbd_dim)
        self.adapter = UnifiedObservationAdapter(
            d_model=d_model, pbd_dim=pbd_dim, clip_dim=clip_dim,
            spec_dim=spec_dim, dropout=dropout, mode=mode)

    @property
    def memory(self):
        return self.uidm.memory

    def forward_frame(self, frame):
        sem, rel = self.adapter(frame["cand_pbd"], frame["cand_clip"],
                                frame["spec"])
        f2 = dict(frame)
        if self.adapter.mode == "semantic":
            f2["cand_pbd"] = torch.zeros_like(frame["cand_pbd"])
        f2["cand_sem"] = sem
        pred = self.uidm.forward_frame(f2)
        pred["relevance"] = rel
        return pred

    def forward(self, frame):
        return self.forward_frame(frame)


def clip_text_embed(texts, model=None, device="cuda"):
    """Frozen CLIP ViT-B/32 text embeddings (512-d, L2-normalized)."""
    if model is None:
        import clip
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()
    import clip
    toks = clip.tokenize(texts).to(device)
    with torch.no_grad():
        e = model.encode_text(toks).float()
    return F.normalize(e, dim=-1)
