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


class TrajectoryLanguageMemory(nn.Module):
    """Small shared space for UIDM trajectory states and specifications.

    This module is deliberately auxiliary.  It can score an active UIDM
    state against a specification for a contrastive training loss and can
    emit a zero-initialized residual for the UIDM track token.  It never
    replaces UIDM's transition, lifecycle, or memory-update heads.
    """

    def __init__(self, d_model=320, spec_dim=512, dropout=0.1):
        super().__init__()
        self.trajectory_encoder = _Proj(d_model, d_model, dropout)
        self.language_encoder = _Proj(spec_dim, d_model, dropout)
        self.track_delta = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model))
        self.logit_scale = nn.Parameter(torch.tensor(float(1.0 / 0.07)).log())
        # The adapter is an exact no-op at initialization.  The contrastive
        # objective can train the shared-space encoders before this optional
        # residual is allowed to influence UIDM decisions.
        self.delta_gate = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.track_delta[-1].weight)
        nn.init.zeros_(self.track_delta[-1].bias)

    def encode_language(self, spec):
        return self.language_encoder(spec)

    def forward(self, track_tok, spec, spec_valid=None):
        """Return cosine-like scores [B,T] and a conditioned delta [B,T,D]."""
        if spec.dim() != 2:
            spec = spec.reshape(spec.shape[0], -1)
        lang = self.encode_language(spec)
        traj = self.trajectory_encoder(track_tok)
        lang_n = l2norm(lang).unsqueeze(1)
        traj_n = l2norm(traj)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        scores = (traj_n * lang_n).sum(dim=-1) * scale
        lang_exp = lang.unsqueeze(1).expand_as(track_tok)
        delta = self.track_delta(torch.cat([track_tok, lang_exp], dim=-1))
        delta = delta * self.delta_gate.tanh()
        if spec_valid is not None:
            valid = spec_valid.float().reshape(-1, 1, 1)
            delta = delta * valid
            scores = scores * valid.squeeze(-1)
        return scores, delta


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
                 dropout=0.1, mode="unified", cond_gated=False,
                 spec_conditioned=False, trajectory_memory=True):
        super().__init__()
        self.mode = mode
        self.cond_gated = cond_gated
        self.spec_conditioned = bool(spec_conditioned)
        self.use_trajectory_memory = bool(
            spec_conditioned and trajectory_memory)
        self.d_model = d_model
        self.pbd_proj = _Proj(pbd_dim, d_model, dropout)
        self.clip_proj = _Proj(clip_dim, d_model, dropout)
        self.spec_proj = _Proj(spec_dim, d_model, dropout)
        # per-dimension gate on the spec residue
        self.gate = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model))
        # L9: specification-conditioned identity interaction.
        # gate = sigmoid(MLP([z_id, sem])); residue = gate * W(sem).
        # Initialised so the first forward equals the L8-B1 sem-in-core
        # behavior (gate ~ 1, W = identity) and the gate can learn to
        # down-weight semantics when identity evidence is strong.
        self.cond_gate = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model))
        self.sem_transform = nn.Linear(d_model, d_model)
        self.relevance = nn.Sequential(
            nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, 1))
        if self.spec_conditioned:
            # Explicit L14 path: Specification Encoder -> Specification
            # Adapter.  The final projection is zeroed so loading an L11
            # checkpoint preserves its valid-specification behavior until
            # Stage 1 has trained the new parameters.
            self.spec_encoder = _Proj(spec_dim, d_model, dropout)
            self.spec_adapter = nn.Sequential(
                nn.Linear(2 * d_model, 2 * d_model), nn.GELU(),
                nn.Linear(2 * d_model, d_model))
            self.spec_track = nn.Sequential(
                nn.Linear(d_model, 2 * d_model), nn.GELU(),
                nn.Linear(2 * d_model, d_model))
            self.trajectory_language = (
                TrajectoryLanguageMemory(d_model, spec_dim, dropout)
                if self.use_trajectory_memory else None)
        else:
            self.spec_encoder = None
            self.spec_adapter = None
            self.spec_track = None
            self.trajectory_language = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.cond_gate[-1].weight)
        nn.init.ones_(self.cond_gate[-1].bias)  # sigmoid(1) ~ 0.73
        nn.init.eye_(self.sem_transform.weight)
        nn.init.zeros_(self.sem_transform.bias)
        nn.init.zeros_(self.relevance[-1].weight)
        nn.init.zeros_(self.relevance[-1].bias)
        if self.spec_conditioned:
            nn.init.zeros_(self.spec_adapter[-1].weight)
            nn.init.zeros_(self.spec_adapter[-1].bias)
            nn.init.zeros_(self.spec_track[-1].weight)
            nn.init.zeros_(self.spec_track[-1].bias)

    @staticmethod
    def _valid_spec(spec, spec_valid=None):
        """Return one boolean/float validity value per batch item."""
        if spec_valid is not None:
            values = torch.as_tensor(
                spec_valid, device=spec.device, dtype=spec.dtype).reshape(-1)
            if values.numel() == 1:
                return values.expand(spec.shape[0])
            if values.numel() != spec.shape[0]:
                raise ValueError(
                    f"spec_valid has {values.numel()} values for "
                    f"batch size {spec.shape[0]}"
                )
            return values
        base = spec if spec.dim() == 2 else spec[:, 0]
        return (base.float().norm(dim=-1) > 1e-6).float()

    def track_condition(self, trk_tok, spec, spec_valid=None,
                        return_scores=False):
        """Condition UIDM track tokens and optionally return memory scores."""
        if not self.spec_conditioned or self.mode == "identity":
            zero = torch.zeros_like(trk_tok)
            return (zero, None) if return_scores else zero
        if spec.dim() != 2:
            spec = spec.reshape(spec.shape[0], -1)
        valid = self._valid_spec(spec, spec_valid)
        spec_tok = self.spec_encoder(spec)
        delta = self.spec_track(spec_tok).unsqueeze(1).expand_as(trk_tok)
        scores = None
        if self.trajectory_language is not None:
            scores, traj_delta = self.trajectory_language(
                trk_tok, spec, valid)
            delta = delta + traj_delta
        delta = delta * valid[:, None, None]
        return (delta, scores) if return_scores else delta

    def forward(self, pbd, clip, spec, cond_gated=None, spec_valid=None,
                track_tok=None, return_track_cond=False):
        """Returns (semantic_residue [...,d], relevance_logit [...,1]).

        With cond_gated=True the returned residue is
        sigmoid(MLP([z_id, sem])) * W(sem): the specification semantics
        condition the identity stream through a learned gate.
        """
        if spec.dim() < clip.dim():
            spec = spec.unsqueeze(1)
        valid = self._valid_spec(spec, spec_valid)
        c = self.clip_proj(clip)
        s = self.spec_proj(spec)
        g = torch.sigmoid(self.gate(c))
        # Keep the legacy semantic input separate from the L14 residual.  The
        # existing relevance head is an acquisition/calibration signal and
        # must remain comparable to L11 while the new residual conditions the
        # UIDM observation token.
        sem_base = c + g * s
        sem_raw = sem_base
        if self.spec_conditioned and self.mode != "identity":
            spec_tok = self.spec_encoder(spec)
            sem_raw = sem_raw + self.spec_adapter(
                torch.cat([c, spec_tok.expand_as(c)], dim=-1))
        # A null specification must be exactly the original PBD UIDM path:
        # no semantic candidate residue and no relevance signal.
        sem_valid_shape = (valid.shape[0],) + (1,) * (sem_raw.dim() - 1)
        sem_raw = sem_raw * valid.reshape(sem_valid_shape)
        use_cond = self.cond_gated if cond_gated is None else cond_gated
        if use_cond and self.mode != "identity":
            z_id = self.pbd_proj(pbd)
            gate = torch.sigmoid(self.cond_gate(
                torch.cat([z_id, sem_raw], dim=-1)))
            sem = gate * self.sem_transform(sem_raw)
        else:
            sem = sem_raw
        if self.mode == "identity":
            rel_in = self.pbd_proj(pbd)
            sem = torch.zeros_like(sem)
        else:
            # The L14 residual is deliberately not fed into the legacy
            # acquisition head.  Otherwise its absolute logit scale can
            # invalidate the fixed rel>0 output protocol before calibration.
            rel_in = sem_base
        rel = self.relevance(rel_in).squeeze(-1)
        rel_valid_shape = (valid.shape[0],) + (1,) * (rel.dim() - 1)
        rel = rel * valid.reshape(rel_valid_shape)
        if return_track_cond:
            if track_tok is None:
                track_delta, traj_scores = None, None
            else:
                track_delta, traj_scores = self.track_condition(
                    track_tok, spec[:, 0] if spec.dim() > 2 else spec,
                    valid, return_scores=True)
            return sem, rel, track_delta, traj_scores
        return sem, rel


class L8UnifiedUIDM(nn.Module):
    """Unified Observation Adapter in front of the shared UIDM core.

    The UIDM core is constructed with app_dim = d_model, so candidate
    appearance tokens are exactly the unified tokens z.
    """

    def __init__(self, d_model=320, n_layers=4, n_heads=8, ffn_dim=1280,
                 dropout=0.1, no_interaction=False, use_cue_rel=False,
                 pbd_dim=2048, clip_dim=512, spec_dim=512, mode="unified",
                 core=None, sem_in_core=True, cond_gated=False,
                 spec_conditioned=False, trajectory_memory=True):
        super().__init__()
        from locatemot.models.l6_uidm import UIDM
        self.d_model = d_model
        self.app_dim = pbd_dim
        self.sem_in_core = sem_in_core
        self.cond_gated = cond_gated
        self.spec_conditioned = bool(spec_conditioned)
        self.trajectory_memory = bool(trajectory_memory)
        self.uidm = core if core is not None else UIDM(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, dropout=dropout,
            no_interaction=no_interaction, use_cue_rel=use_cue_rel,
            app_dim=pbd_dim)
        self.adapter = UnifiedObservationAdapter(
            d_model=d_model, pbd_dim=pbd_dim, clip_dim=clip_dim,
            spec_dim=spec_dim, dropout=dropout, mode=mode,
            cond_gated=cond_gated, spec_conditioned=spec_conditioned,
            trajectory_memory=trajectory_memory)

    @property
    def memory(self):
        return self.uidm.memory

    def forward_frame(self, frame):
        sem, rel, track_delta, traj_scores = self.adapter(
            frame["cand_pbd"], frame["cand_clip"], frame["spec"],
            cond_gated=self.cond_gated,
            spec_valid=frame.get("spec_valid"),
            track_tok=frame.get("trk_tok"), return_track_cond=True)
        f2 = dict(frame)
        if self.adapter.mode == "semantic":
            f2["cand_pbd"] = torch.zeros_like(frame["cand_pbd"])
        if self.sem_in_core:
            f2["cand_sem"] = sem
        else:
            f2["cand_sem"] = torch.zeros_like(sem)
        if track_delta is not None and f2.get("trk_tok") is not None:
            f2["trk_tok"] = f2["trk_tok"] + track_delta
        pred = self.uidm.forward_frame(f2)
        pred["relevance"] = rel
        if traj_scores is not None:
            pred["trajectory_scores"] = traj_scores
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


def load_l8_state(model, ck_sd):
    """Load an L8 state dict, accepting both prefixed (uidm./adapter.) and
    bare L6 core keys (no uidm. prefix)."""
    norm = {}
    for k, v in ck_sd.items():
        if k.startswith("uidm.") or k.startswith("adapter."):
            norm[k] = v
        else:
            norm["uidm." + k] = v
    return model.load_state_dict(norm, strict=False)
