"""Stage L4: specification-equivariant shared identity core.

Scientific question (Stage L4): object specification determines WHAT to
track, while the shared identity core determines HOW to track.  A real
object's persistent identity should not arbitrarily change when the
requested candidate subset (specification restriction) changes:

    T_theta(R_s(X)) ≈ R_s(T_theta(X))   on common objects,

where ≈ is measured with a permutation-invariant co-identity agreement
(never raw track integer IDs).

Architecture: the shared core is the L1-D EGRA associator (U0, same
set-level transformer + bounded residual + reliability gate).  The only
specification-conditioned part is a small, shared, type-level embedding
(ALL / category / instance), which does NOT select candidates itself and
is not dataset-specific.  Restriction-equivariant learning is imposed by
paired-view consistency losses in `tools/train_l4.py`.

Design references (clean reimplementation, no code copied):
- CAMELTrack GAFFE set-level interaction + ranking training (commit 46a74bb);
- Path Consistency multi-view association consistency (commit f4b7d26d)
  as conceptual basis for permutation-invariant assignment agreement;
- V2-SAM visual-prompt matcher contrastive alignment (commit 31c3babf)
  as reference for cross-view representation agreement.
"""
from __future__ import annotations

import torch
from torch import nn

from locatemot.models.l1d_association import (
    L1DAssociator,
    PAIR_FEATURES,
)


class L4SpecEqAssociator(L1DAssociator):
    """U0 shared core + bounded type-level specification conditioning."""

    def __init__(self, n_spec: int = 3, d_spec: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.use_spec = True
        self.n_spec = n_spec
        self.d_spec = d_spec
        self.spec_embed = nn.Embedding(n_spec, d_spec)
        self.spec_proj = nn.Linear(d_spec, self.d_model)
        self._init_weights()

    def forward(self, batch):
        pair = torch.nan_to_num(batch["pair_feats"].float())
        trk_f = torch.nan_to_num(batch["track_feats"].float())
        cand_f = torch.nan_to_num(batch["cand_feats"].float())
        base = torch.nan_to_num(batch["base"].float())
        B, T, N, Fp = pair.shape
        trk_mask = batch.get("trk_mask")
        cand_mask = batch.get("cand_mask")
        if trk_mask is None:
            trk_mask = torch.ones(B, T, dtype=torch.bool, device=pair.device)
        if cand_mask is None:
            cand_mask = torch.ones(B, N, dtype=torch.bool, device=pair.device)
        spec = batch.get("spec")
        if spec is None:
            spec = torch.zeros(B, dtype=torch.long, device=pair.device)
        spec_tok = self.spec_proj(self.spec_embed(spec))  # [B,D]

        trk_tok = self.track_proj(trk_f) + spec_tok.unsqueeze(1)
        cand_tok = self.cand_proj(cand_f) + spec_tok.unsqueeze(1)
        tokens = torch.cat([cand_tok, trk_tok], dim=1)
        mask = torch.cat([cand_mask, trk_mask], dim=1)
        out = self.set_encoder(tokens, src_key_padding_mask=~mask)
        cand_out = out[:, :N]
        trk_out = out[:, N:]

        t_exp = trk_out.unsqueeze(2).expand(B, T, N, self.d_model)
        c_exp = cand_out.unsqueeze(1).expand(B, T, N, self.d_model)
        pair_in = torch.cat([t_exp, c_exp, pair], dim=-1)
        delta = self.delta_scale * torch.tanh(self.pair_head(pair_in).squeeze(-1))

        row_sel = torch.stack([
            trk_f[..., 9], trk_f[..., 10], trk_f[..., 11], trk_f[..., 12],
            trk_f[..., 13], trk_f[..., 6],
            pair.max(dim=2).values[..., 11],
        ], dim=-1)
        rel_in = torch.cat([trk_out, row_sel], dim=-1)
        rel_logit = self.reliability_head(rel_in).squeeze(-1)
        final = base + torch.sigmoid(rel_logit).unsqueeze(-1) * delta
        return {
            "final": final,
            "delta": delta,
            "reliability_logit": rel_logit,
            "reliability": torch.sigmoid(rel_logit),
            "trk_tok": trk_out,
            "cand_tok": cand_out,
        }
