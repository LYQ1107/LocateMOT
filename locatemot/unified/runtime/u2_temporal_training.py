"""Causal MOT clip rollout with external GT bookkeeping and scheduled forcing."""
from __future__ import annotations

import torch
from torch import nn

from locatemot.unified.losses.u2_association import association_targets, u2_association_losses
from locatemot.unified.losses.u2_semantic import u2_semantic_losses
from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
from locatemot.unified.models.u2.track_state import UnifiedTrackState


def teacher_forcing_probability(step: int) -> float:
    if step <= 500:
        return 1.
    return max(.4, 1. - .6 * (step - 500) / 2000)


def clip_length(step: int) -> int:
    return 2 if step < 1000 else 4 if step < 2500 else 6 if step < 4000 else 8


def update_identity_bookkeeping(prior: dict[int, int], output: dict, candidate_gt_ids: torch.Tensor) -> dict[int, int]:
    """Preserve the first known identity even when the model makes a switch.

    Duplicate births cannot adopt a GT while its canonical track remains active.
    Retired/dropped tracks can be replaced by a later runtime ID.
    """
    live = set(output["state"].track_ids.tolist())
    mapping = {track_id: identity for track_id, identity in prior.items() if track_id in live}
    owned = {identity for identity in mapping.values() if identity >= 0}
    for candidate, track_id in zip(output["candidate_indices"].tolist(), output["track_ids"].tolist()):
        identity = int(candidate_gt_ids[candidate])
        if mapping.get(track_id, -1) >= 0:
            continue
        if identity >= 0 and identity not in owned:
            mapping[track_id] = identity
            owned.add(identity)
        else:
            mapping.setdefault(track_id, -1)
    return mapping


class U2MOTTemporalTraining(nn.Module):
    """Connected full clips inside one DDP forward; foundation stays outside."""

    def __init__(self, core: SharedTrackCore | None = None) -> None:
        super().__init__()
        self.core = core if core is not None else SharedTrackCore()

    def forward(self, clips: list[list[dict]], step: int, *, generator: torch.Generator | None = None,
                forcing_decisions: list[list[bool]] | None = None, augment: bool = True,
                task_type: str = "mot", class_text: torch.Tensor | None = None,
                base_class_ids: torch.Tensor | None = None) -> dict:
        if not self.training:
            raise ValueError("Temporal supervision is training-only")
        if task_type not in {"mot", "ovmot"}:
            raise ValueError("U2 temporal tasks are mot and ovmot")
        if task_type == "ovmot" and (class_text is None or base_class_ids is None):
            raise ValueError("OVMOT requires the selected shared language features and base vocabulary")
        losses, diagnostics = [], []
        for clip_index, clip in enumerate(clips):
            state = None
            bookkeeping: dict[int, int] = {}
            certified_negative_tracks: set[int] = set()
            past_negatives = None
            for index, frame in enumerate(clip):
                candidates, gt = frame["candidates"], frame["supervision"]
                if state is None:
                    state = UnifiedTrackState.empty(self.core.hidden_dim, device=candidates["boxes"].device,
                                                    dtype=candidates["boxes"].dtype)
                state = self.core.prepare_state(state, frame["frame_index"])
                query_info = {"dropped_track_ids": [], "injected_track_ids": [], "injected_past_negative_indices": []}
                if augment:
                    state, query_info = self.core.query_interaction.augment_state(state, past_negatives, generator=generator)
                live = set(state.track_ids.tolist())
                bookkeeping = {key: value for key, value in bookkeeping.items() if key in live}
                certified_negative_tracks &= live
                certified_negative_tracks.update(query_info["injected_track_ids"])
                if task_type == "ovmot" and gt.category_ids is None:
                    raise ValueError("OVMOT requires external base-only original supervision")
                targets = association_targets(state, bookkeeping, candidates, gt, certified_negative_tracks)
                forced = (forcing_decisions[clip_index][index] if forcing_decisions is not None
                          else bool(torch.rand((), device=state.track_ids.device, generator=generator)
                                    < teacher_forcing_probability(step)))
                output = self.core(candidates, state, task_type=task_type, class_text=class_text,
                                   frame_index=frame["frame_index"], training=True,
                                   forced_assignment=(targets.matched_pairs, targets.births) if forced else None)
                frame_losses = u2_association_losses(output, state, targets, semantic=task_type == "ovmot")
                bookkeeping = update_identity_bookkeeping(bookkeeping, output, targets.candidate_gt_ids)
                if task_type == "ovmot":
                    semantic_losses = u2_semantic_losses(self.core, output, state, targets, gt, bookkeeping,
                                                        class_text, base_class_ids)
                    frame_losses["total"] = frame_losses["total"] + semantic_losses["semantic_total"]
                    frame_losses.update(semantic_losses)
                losses.append(frame_losses)
                for candidate, track_id in zip(output["candidate_indices"].tolist(), output["track_ids"].tolist()):
                    if targets.eligible[candidate] and targets.candidate_gt_ids[candidate] < 0:
                        certified_negative_tracks.add(track_id)
                    elif targets.candidate_gt_ids[candidate] >= 0:
                        certified_negative_tracks.discard(track_id)
                state = output["state"]
                negative = (targets.candidate_gt_ids < 0) & targets.eligible
                past_negatives = {"candidate_id_tokens": output["candidate_id_tokens"][negative],
                                  "candidate_sem_tokens": output["candidate_sem_tokens"][negative],
                                  "boxes": candidates["boxes"][negative], "scores": candidates["scores"][negative]}
                diagnostics.append({"clip": clip_index, "frame_index": frame["frame_index"],
                                    "teacher_forced": forced, "query_interaction": query_info,
                                    "active_tracks": state.num_tracks, "positive_candidates": int((targets.candidate_gt_ids >= 0).sum()),
                                    "negative_candidates": int(negative.sum()), "track_gt_ids": dict(bookkeeping),
                                    "model_matched_pairs": output["model_matched_pairs"],
                                    "model_birth_candidates": output["model_birth_candidates"]})
        mean = {key: torch.stack([loss[key] for loss in losses]).mean() for key in losses[0]}
        return {"losses": mean, "diagnostics": diagnostics,
                "teacher_forcing_probability": teacher_forcing_probability(step), "task_type": task_type}
