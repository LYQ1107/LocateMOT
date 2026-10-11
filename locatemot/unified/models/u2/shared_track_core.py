"""Causal persistent identity core shared by ordinary MOT and OVMOT.

The module keeps the state transition explicit.  A candidate's
query-independent visual feature is the only visual input to the identity
path.  OVMOT class text is consumed after identity logits have been computed,
through a bounded semantic residual; it is never an input to the identity
GRU, anchor, or appearance EMA.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .association_head import AssociationHead
from .candidate_encoder import CandidateEncoder
from .lifecycle_head import LifecycleHead
from .motion_head import MotionHead
from .ovmot_adapter import OVMOTAdapter
from .query_interaction import QueryInteraction
from .set_interaction import SetInteraction
from .track_state import UnifiedTrackState


PAIR_FEATURE_NAMES = (
    "iou_last",
    "iou_predicted",
    "center_distance",
    "predicted_center_distance",
    "log_width_ratio",
    "log_height_ratio",
    "frame_gap",
    "track_age",
    "track_hits",
    "candidate_score",
    "id_cosine",
    "anchor_cosine",
    "appearance_cosine",
)


def _iou_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return pairwise xyxy IoU without using future frames or labels."""
    if left.shape[0] == 0 or right.shape[0] == 0:
        return left.new_empty((left.shape[0], right.shape[0]))
    l = left[:, None, :]
    r = right[None, :, :]
    inter0 = torch.maximum(l[..., :2], r[..., :2])
    inter1 = torch.minimum(l[..., 2:], r[..., 2:])
    wh = (inter1 - inter0).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    la = (l[..., 2] - l[..., 0]).clamp_min(0) * (l[..., 3] - l[..., 1]).clamp_min(0)
    ra = (r[..., 2] - r[..., 0]).clamp_min(0) * (r[..., 3] - r[..., 1]).clamp_min(0)
    return inter / (la + ra - inter).clamp_min(1e-6)


def _box_geometry(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    sizes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-5)
    return centers, sizes


def _apply_motion(boxes: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Apply normalized ``[dcx/scale, dcy/scale, dlogw, dlogh]`` motion."""
    if boxes.shape[0] == 0:
        return boxes
    centers, sizes = _box_geometry(boxes)
    next_centers = centers + velocity[:, :2] * sizes
    next_sizes = sizes * velocity[:, 2:].clamp(-2.0, 2.0).exp()
    return torch.cat([next_centers - next_sizes * 0.5, next_centers + next_sizes * 0.5], dim=-1)


def _motion_delta(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    if previous.shape[0] == 0:
        return previous.new_empty((0, 4))
    previous_centers, previous_sizes = _box_geometry(previous)
    current_centers, current_sizes = _box_geometry(current)
    center_delta = (current_centers - previous_centers) / previous_sizes
    log_size_delta = (current_sizes / previous_sizes).log()
    return torch.cat([center_delta, log_size_delta], dim=-1)


def _pair_features(
    state: UnifiedTrackState,
    candidate_id_tokens: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    frame_delta: int = 1,
) -> torch.Tensor:
    """Build the registered 13-dimensional causal pair feature vector."""
    t, n = state.num_tracks, boxes.shape[0]
    if t == 0 or n == 0:
        return boxes.new_empty((t, n, len(PAIR_FEATURE_NAMES)))

    predicted = _apply_motion(state.boxes, state.velocity * frame_delta)
    iou_last = _iou_matrix(state.boxes, boxes)
    iou_pred = _iou_matrix(predicted, boxes)
    track_centers = (state.boxes[:, :2] + state.boxes[:, 2:]) * 0.5
    predicted_centers = (predicted[:, :2] + predicted[:, 2:]) * 0.5
    candidate_centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    extent = (state.boxes[:, 2:] - state.boxes[:, :2]).clamp_min(1e-5)
    scale = extent.norm(dim=-1).clamp_min(1.0)
    center_distance = (track_centers[:, None] - candidate_centers[None]).norm(dim=-1) / scale[:, None]
    predicted_distance = (predicted_centers[:, None] - candidate_centers[None]).norm(dim=-1) / scale[:, None]

    track_size = (state.boxes[:, 2:] - state.boxes[:, :2]).clamp_min(1e-5)
    candidate_size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-5)
    width_ratio = (track_size[:, None, 0] / candidate_size[None, :, 0]).log().clamp(-5.0, 5.0)
    height_ratio = (track_size[:, None, 1] / candidate_size[None, :, 1]).log().clamp(-5.0, 5.0)

    # Keep continuous geometry in a stable range before the association MLP.
    frame_gap = (state.lost_age + frame_delta).float().clamp_max(30)[:, None].expand(t, n) / 30.0
    track_age = state.age.float().clamp_max(1000)[:, None].expand(t, n) / 100.0
    track_hits = state.hits.float().clamp_max(1000)[:, None].expand(t, n) / 100.0
    candidate_score = scores[None, :].expand(t, n).clamp(0.0, 1.0)
    id_cosine = F.cosine_similarity(state.id_memory[:, None, :], candidate_id_tokens[None, :, :], dim=-1)
    anchor_cosine = F.cosine_similarity(state.identity_anchor[:, None, :], candidate_id_tokens[None, :, :], dim=-1)
    appearance_cosine = F.cosine_similarity(state.appearance_ema[:, None, :], candidate_id_tokens[None, :, :], dim=-1)
    return torch.stack(
        [
            iou_last,
            iou_pred,
            center_distance.clamp_max(20.0) / 20.0,
            predicted_distance.clamp_max(20.0) / 20.0,
            width_ratio / 5.0,
            height_ratio / 5.0,
            frame_gap,
            track_age,
            track_hits,
            candidate_score,
            id_cosine,
            anchor_cosine,
            appearance_cosine,
        ],
        dim=-1,
    )


class SharedTrackCore(nn.Module):
    """One state transition used by MOT and OVMOT.

    Hungarian assignment is performed over the current frame only.  Unmatched
    tracks remain in ``UnifiedTrackState`` for ``max_age`` frames and are never
    emitted as motion-only boxes.  IDs are monotonic inside an online state.
    """

    def __init__(self, hidden_dim: int = 256, max_age: int = 30) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_age = int(max_age)
        self.candidate_encoder = CandidateEncoder(hidden_dim=hidden_dim, visual_dim=hidden_dim)
        self.set_interaction = SetInteraction(hidden_dim=hidden_dim, pair_dim=len(PAIR_FEATURE_NAMES))
        self.association = AssociationHead(hidden_dim=hidden_dim, pair_dim=len(PAIR_FEATURE_NAMES))
        self.ovmot = OVMOTAdapter(hidden_dim=hidden_dim)
        self.lifecycle = LifecycleHead(hidden_dim=hidden_dim)
        self.motion = MotionHead(hidden_dim=hidden_dim)
        self.identity_gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.semantic_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.frame_context = nn.Linear(hidden_dim, hidden_dim)
        self.query_interaction = QueryInteraction(drop_probability=0.10, false_positive_probability=0.20)

    @staticmethod
    def _assignment(
        pair_logits: torch.Tensor, no_match_logits: torch.Tensor, new_logits: torch.Tensor
    ) -> tuple[list[tuple[int, int]], set[int]]:
        """Apply one-to-one Hungarian matching and zero decision margins."""
        t, n = pair_logits.shape
        matched: list[tuple[int, int]] = []
        pair_values = pair_logits.detach().float().cpu().numpy()
        no_match_values = no_match_logits.detach().float().cpu().numpy()
        new_values = new_logits.detach().float().cpu().numpy()
        if t and n:
            rows, cols = linear_sum_assignment(-pair_values)
            for row, col in zip(rows.tolist(), cols.tolist()):
                score = float(pair_values[row, col])
                no_match = float(no_match_values[row])
                if score > 0.0 and score > no_match:
                    matched.append((int(row), int(col)))
        matched_candidates = {col for _, col in matched}
        births = {
            col
            for col in range(n)
            if col not in matched_candidates and float(new_values[col]) > 0.0
        }
        return matched, births

    def prepare_state(self, state: UnifiedTrackState, current_frame: int) -> UnifiedTrackState:
        """Retire expired identities before current-frame matching.

        The temporal wrapper calls this before constructing external GT targets;
        the forward call also applies it for ordinary online inference.
        """
        if not state.num_tracks:
            return state
        keep = current_frame - state.last_frame <= self.max_age
        fields = {name: value[keep] if isinstance(value, torch.Tensor) else value
                  for name, value in state.__dict__.items()}
        fields["metadata"] = dict(state.metadata)
        return UnifiedTrackState(**fields)

    def forward(
        self,
        candidates: dict[str, torch.Tensor],
        state: UnifiedTrackState | None = None,
        *,
        task_type: str = "mot",
        class_text: torch.Tensor | None = None,
        frame_index: int | None = None,
        training: bool = False,
        forced_assignment: tuple[list[tuple[int, int]], set[int]] | None = None,
    ) -> dict[str, Any]:
        if task_type not in {"mot", "ovmot"}:
            raise ValueError("U2 core supports only mot and ovmot")
        required = ("boxes", "scores", "id_visual")
        missing = [name for name in required if name not in candidates]
        if missing:
            raise KeyError(f"missing candidate fields: {missing}")
        boxes = candidates["boxes"]
        scores = candidates["scores"]
        id_visual = candidates["id_visual"]
        semantic_visual = candidates.get("semantic_visual", id_visual)
        decoder_visual = candidates.get("decoder_query", semantic_visual)
        encoded = self.candidate_encoder(id_visual, boxes, scores, decoder_visual, semantic_visual)
        candidate_id = encoded["candidate_id_token"]
        candidate_sem = encoded["candidate_sem_token"]
        if state is None:
            state = UnifiedTrackState.empty(self.hidden_dim, device=boxes.device, dtype=candidate_id.dtype)
        if state.hidden_dim != self.hidden_dim:
            raise ValueError("state hidden dimension mismatch")
        if state.track_ids.device != boxes.device:
            state = state.to(boxes.device)
        current_frame = state.frame_index + 1 if frame_index is None else int(frame_index)
        if current_frame <= state.frame_index:
            raise ValueError("Track state can only advance to a later frame")
        frame_delta = current_frame - state.frame_index if state.frame_index >= 0 else 1
        if forced_assignment is not None and not (training and self.training):
            raise ValueError("Teacher assignments are available only during training")
        state = self.prepare_state(state, current_frame)

        # The temporal training wrapper augments the actual incoming state
        # using past unmatched proposals, before it constructs GT targets.
        pair_features = _pair_features(state, candidate_id, boxes, scores, frame_delta)
        track_tokens, candidate_tokens = self.set_interaction(state.id_memory, candidate_id, pair_features)
        identity = self.association(track_tokens, candidate_tokens, pair_features)
        semantic = self.ovmot(
            identity["identity_pair_logits"],
            state.semantic_memory,
            candidate_sem,
            task_type=task_type,
            class_text=class_text,
        )
        matched, births = self._assignment(
            semantic["pair_logits"], identity["no_match_logits"], identity["new_logits"]
        )
        model_matched, model_births = matched, births
        if forced_assignment is not None:
            matched, births = forced_assignment
        alive_logits = self.lifecycle(
            track_tokens,
            state.age,
            state.hits,
            state.lost_age + frame_delta,
            state.confidence,
        )
        next_state, current_indices, current_ids = self._update_state(
            state,
            candidate_id,
            candidate_sem,
            boxes,
            scores,
            track_tokens,
            matched,
            births,
            current_frame,
            alive_logits,
            frame_delta,
        )
        motion_delta = self.motion(track_tokens, state.velocity)
        return {
            "state": next_state,
            "track_ids": current_ids,
            "candidate_indices": current_indices,
            "pair_logits": semantic["pair_logits"],
            "identity_pair_logits": identity["identity_pair_logits"],
            "no_match_logits": identity["no_match_logits"],
            "new_logits": identity["new_logits"],
            "semantic_pair_residual": semantic["semantic_pair_residual"],
            "semantic_gate": semantic["semantic_gate"],
            "alive_logits": alive_logits,
            "motion_delta": motion_delta,
            "matched_pairs": matched,
            "birth_candidates": sorted(births),
            "candidate_id_tokens": candidate_id,
            "candidate_sem_tokens": candidate_sem,
            "track_tokens": track_tokens,
            "model_matched_pairs": model_matched,
            "model_birth_candidates": sorted(model_births),
            "teacher_forced": forced_assignment is not None,
            "frame_delta": frame_delta,
            "pair_feature_names": PAIR_FEATURE_NAMES,
            "task_type": task_type,
        }

    def _update_state(
        self,
        state: UnifiedTrackState,
        candidate_id: torch.Tensor,
        candidate_sem: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        track_tokens: torch.Tensor,
        matched: list[tuple[int, int]],
        births: set[int],
        frame_index: int,
        alive_logits: torch.Tensor,
        frame_delta: int = 1,
    ) -> tuple[UnifiedTrackState, torch.Tensor, torch.Tensor]:
        old = state
        t = old.num_tracks
        device = boxes.device
        predicted = _apply_motion(old.boxes, old.velocity * frame_delta)
        new_id = old.id_memory.clone()
        new_anchor = old.identity_anchor.clone()
        new_app = old.appearance_ema.clone()
        new_sem = old.semantic_memory.clone()
        new_boxes = predicted.clone()
        new_velocity = old.velocity.clone()
        new_age = old.age + frame_delta
        new_hits = old.hits.clone()
        new_lost = old.lost_age + frame_delta
        new_last = old.last_frame.clone()
        new_conf = old.confidence * (0.95 ** frame_delta)
        new_alive = old.alive.clone()
        retained = torch.zeros((t,), device=device, dtype=torch.bool)
        current: dict[int, int] = {}
        context = self.frame_context(track_tokens) if t else candidate_id.new_empty((0, self.hidden_dim))
        rows = torch.tensor([row for row, _ in matched], device=device, dtype=torch.long)
        cols = torch.tensor([col for _, col in matched], device=device, dtype=torch.long)
        if matched:
            # GRUCell(input, hidden): task text is deliberately absent.
            gru_input = torch.cat([candidate_id[cols], context[rows]], dim=-1)
            new_id[rows] = self.identity_gru(gru_input, old.id_memory[rows]).to(new_id.dtype)
            new_app[rows] = 0.9 * old.appearance_ema[rows] + 0.1 * candidate_id[cols]
            new_sem[rows] = self.semantic_gru(candidate_sem[cols], old.semantic_memory[rows]).to(new_sem.dtype)
            new_velocity[rows] = _motion_delta(old.boxes[rows], boxes[cols]) / frame_delta
            new_boxes[rows] = boxes[cols]
            new_hits[rows] = old.hits[rows] + 1
            new_lost[rows] = 0
            new_last[rows] = frame_index
            new_conf[rows] = scores[cols]
            new_alive[rows] = True
            retained[rows] = True
            for candidate_index, track_id in zip(cols.tolist(), old.track_ids[rows].tolist()):
                current[candidate_index] = track_id

        unmatched = torch.ones(t, device=device, dtype=torch.bool)
        unmatched[rows] = False
        hard_cap = old.lost_age + frame_delta <= self.max_age
        # First miss retained; later misses need learned alive, with a hard cap.
        retained[unmatched] = (hard_cap & ((old.lost_age == 0) | (alive_logits >= 0.0)))[unmatched]
        new_alive[unmatched] = retained[unmatched]
        new_id[unmatched] = old.id_memory[unmatched] * (0.995 ** frame_delta)

        # Births are sorted so ID assignment is deterministic under equal logits.
        new_ids = old.track_ids.clone()
        next_id = int(old.next_track_id)
        birth_indices = sorted(births)
        count = len(birth_indices)
        if count:
            birth_columns = torch.tensor(birth_indices, device=device, dtype=torch.long)
            new_ids = torch.cat([new_ids, torch.arange(next_id, next_id + count, device=device)])
            new_id = torch.cat([new_id, candidate_id[birth_columns]], dim=0)
            new_anchor = torch.cat([new_anchor, candidate_id[birth_columns]], dim=0)
            new_app = torch.cat([new_app, candidate_id[birth_columns]], dim=0)
            new_sem = torch.cat([new_sem, candidate_sem[birth_columns]], dim=0)
            new_boxes = torch.cat([new_boxes, boxes[birth_columns]], dim=0)
            new_velocity = torch.cat([new_velocity, boxes.new_zeros((count, 4))], dim=0)
            new_age = torch.cat([new_age, old.age.new_ones(count)], dim=0)
            new_hits = torch.cat([new_hits, old.hits.new_ones(count)], dim=0)
            new_lost = torch.cat([new_lost, old.lost_age.new_zeros(count)], dim=0)
            new_last = torch.cat([new_last, old.last_frame.new_full((count,), frame_index)], dim=0)
            new_conf = torch.cat([new_conf, scores[birth_columns]], dim=0)
            new_alive = torch.cat([new_alive, old.alive.new_ones(count)], dim=0)
            retained = torch.cat([retained, old.alive.new_ones(count)], dim=0)
            for offset, candidate_index in enumerate(birth_indices):
                current[candidate_index] = next_id + offset
            next_id += count

        keep = retained & (new_lost <= self.max_age)
        next_state = UnifiedTrackState(
            track_ids=new_ids[keep],
            id_memory=new_id[keep],
            identity_anchor=new_anchor[keep],
            appearance_ema=new_app[keep],
            semantic_memory=new_sem[keep],
            boxes=new_boxes[keep],
            velocity=new_velocity[keep],
            age=new_age[keep],
            hits=new_hits[keep],
            lost_age=new_lost[keep],
            last_frame=new_last[keep],
            confidence=new_conf[keep],
            alive=new_alive[keep],
            next_track_id=next_id,
            frame_index=frame_index,
            metadata={"max_age": self.max_age},
        )
        current_indices = torch.tensor(sorted(current), device=device, dtype=torch.long)
        current_ids = torch.tensor(
            [current[index] for index in sorted(current)], device=device, dtype=torch.long
        )
        return next_state, current_indices, current_ids
