"""Registered U2 identity, lifecycle and motion supervision.

GT assignment/bookkeeping stays outside the causal core. Detector candidates
are assigned by current-frame IoU Hungarian, with official ignored regions
excluded from negative supervision. False proposals are never taught NEW.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from locatemot.unified.models.u2.shared_track_core import _iou_matrix, _motion_delta
from locatemot.unified.models.u2.track_state import UnifiedTrackState


@dataclass
class FrameSupervision:
    boxes: torch.Tensor
    identity_ids: torch.Tensor
    ignore_boxes: torch.Tensor
    # Visible now or an eligible observation within the next max_age source
    # frames. This is a loss label, never a runtime feature/state field.
    trackable_ids: set[int]
    distractor_boxes: torch.Tensor | None = None
    minimum_candidate_height: float = 0.
    # OVMOT labels remain external. None retains exhaustive ordinary MOT.
    category_ids: torch.Tensor | None = None
    negative_category_ids: set[int] | None = None

    def to(self, device: torch.device | str) -> "FrameSupervision":
        return FrameSupervision(self.boxes.to(device), self.identity_ids.to(device),
                                self.ignore_boxes.to(device), self.trackable_ids,
                                None if self.distractor_boxes is None else self.distractor_boxes.to(device),
                                self.minimum_candidate_height,
                                None if self.category_ids is None else self.category_ids.to(device),
                                self.negative_category_ids)


@dataclass
class AssociationTargets:
    candidate_gt_ids: torch.Tensor
    eligible: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    alive: torch.Tensor
    matched_pairs: list[tuple[int, int]]
    births: set[int]
    gt_boxes: torch.Tensor
    row_supervised: torch.Tensor | None = None
    candidate_category_ids: torch.Tensor | None = None


def candidate_gt_assignment(boxes: torch.Tensor, gt: FrameSupervision) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((len(boxes),), -1, device=boxes.device, dtype=torch.long)
    eligible = torch.ones(len(boxes), device=boxes.device, dtype=torch.bool)
    distractors = gt.boxes.new_empty((0, 4)) if gt.distractor_boxes is None else gt.distractor_boxes
    matching_boxes = torch.cat([gt.boxes, distractors])
    matched = torch.zeros(len(boxes), device=boxes.device, dtype=torch.bool)
    if len(boxes) and len(matching_boxes):
        iou = _iou_matrix(boxes, matching_boxes)
        iou = torch.where(iou >= .5, iou, 0.)
        iou_values = iou.detach().float().cpu().numpy()
        rows, cols = linear_sum_assignment(-iou_values)
        positive_rows, positive_cols, ignored_rows, matched_rows = [], [], [], []
        for row, col in zip(rows.tolist(), cols.tolist()):
            if iou_values[row, col] >= .5:
                matched_rows.append(row)
                if col < len(gt.boxes):
                    positive_rows.append(row)
                    positive_cols.append(col)
                else:
                    ignored_rows.append(row)
        row_tensor = torch.tensor(positive_rows, device=boxes.device, dtype=torch.long)
        col_tensor = torch.tensor(positive_cols, device=boxes.device, dtype=torch.long)
        ids[row_tensor] = gt.identity_ids[col_tensor]
        matched[torch.tensor(matched_rows, device=boxes.device, dtype=torch.long)] = True
        eligible[torch.tensor(ignored_rows, device=boxes.device, dtype=torch.long)] = False
    if len(boxes) and len(gt.ignore_boxes):
        lo = torch.maximum(boxes[:, None, :2], gt.ignore_boxes[None, :, :2])
        hi = torch.minimum(boxes[:, None, 2:], gt.ignore_boxes[None, :, 2:])
        extent = (hi - lo).clamp_min(0)
        intersection = extent[..., 0] * extent[..., 1]
        size = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0)
        area = (size[:, 0] * size[:, 1]).clamp_min(1e-6)
        ignored = (intersection / area[:, None]).amax(dim=1) > .5
        eligible &= ~(ignored & ~matched)
    if gt.minimum_candidate_height:
        eligible &= ~((boxes[:, 3] - boxes[:, 1] <= gt.minimum_candidate_height) & ~matched)
    return ids, eligible


def association_targets(
    state: UnifiedTrackState,
    track_gt_ids: dict[int, int],
    candidates: dict[str, torch.Tensor],
    gt: FrameSupervision,
    certified_negative_track_ids: set[int] | None = None,
) -> AssociationTargets:
    ids, eligible = candidate_gt_assignment(candidates["boxes"], gt)
    n, t = len(ids), state.num_tracks
    row_values, col_values, alive_values = [n] * t, [t] * n, [0.] * t
    matches: list[tuple[int, int]] = []
    canonical: dict[int, int] = {}
    for index, track_id in enumerate(state.track_ids.tolist()):
        identity = track_gt_ids.get(track_id, -1)
        if identity >= 0:
            if identity in canonical:
                raise ValueError("Two runtime tracks own the same canonical training identity")
            canonical[identity] = index
            alive_values[index] = float(identity in gt.trackable_ids)
    candidate_ids = ids.tolist()
    for candidate, identity in enumerate(candidate_ids):
        if identity in canonical:
            track = canonical[identity]
            row_values[track] = candidate
            col_values[candidate] = track
            matches.append((track, candidate))
    births = {index for index, identity in enumerate(candidate_ids) if identity >= 0 and identity not in canonical}
    gt_box_by_id = {identity: index for index, identity in enumerate(gt.identity_ids.tolist())}
    box_indices = torch.tensor([gt_box_by_id[candidate_ids[candidate]] for _, candidate in matches], device=ids.device, dtype=torch.long)
    matched_gt_boxes = gt.boxes[box_indices]
    row = torch.tensor(row_values, device=ids.device, dtype=torch.long)
    col = torch.tensor(col_values, device=ids.device, dtype=torch.long)
    alive = torch.tensor(alive_values, device=ids.device, dtype=torch.float32)
    row_supervised, category_ids = None, None
    if gt.category_ids is not None:
        # Partial OVMOT annotations: unknown objects remain candidates but
        # cannot become negative identity/alive/NEW labels merely by omission.
        category_ids = torch.full_like(ids, -1)
        for identity, category in zip(gt.identity_ids.tolist(), gt.category_ids.tolist()):
            category_ids[ids == identity] = category
        certified = torch.zeros(n, device=ids.device, dtype=torch.bool)
        for category in gt.negative_category_ids or set():
            certified |= candidates["predicted_class"] == category
        if n and len(gt.boxes):
            # Duplicate proposals of a labeled base object are real hard
            # negatives for identity births, even when Hungarian chose another.
            certified |= _iou_matrix(candidates["boxes"], gt.boxes).amax(dim=1) >= .5
        eligible &= (ids >= 0) | certified
        row_supervised = torch.tensor(
            [track_gt_ids.get(track_id, -1) >= 0 or track_id in (certified_negative_track_ids or set())
             for track_id in state.track_ids.tolist()], device=ids.device, dtype=torch.bool)
    return AssociationTargets(ids, eligible, row, col, alive, matches, births, matched_gt_boxes,
                              row_supervised, category_ids)


def u2_association_losses(output: dict, state: UnifiedTrackState, targets: AssociationTargets,
                          *, semantic: bool = False) -> dict[str, torch.Tensor]:
    pair = output["pair_logits" if semantic else "identity_pair_logits"].float()
    no_match, new = output["no_match_logits"].float(), output["new_logits"].float()
    t, n = pair.shape
    zero = output["candidate_id_tokens"].sum() * 0 + new.sum() * 0 + no_match.sum() * 0
    row_logits = torch.cat([pair.masked_fill(~targets.eligible[None], -1e4), no_match[:, None]], dim=1)
    supervised = (torch.ones(t, device=pair.device, dtype=torch.bool) if targets.row_supervised is None
                  else targets.row_supervised)
    row = F.cross_entropy(row_logits[supervised], targets.row[supervised]) if supervised.any() else zero
    positive = targets.candidate_gt_ids >= 0
    col_pair = pair if targets.row_supervised is None else pair.masked_fill(~supervised[:, None], -1e4)
    col_logits = torch.cat([col_pair.T, new[:, None]], dim=1)
    col_ce = F.cross_entropy(col_logits[positive], targets.col[positive]) if positive.any() else zero
    birth_target = (positive & (targets.col == t)).float()
    # NEW has a fixed zero reject boundary in the runtime core. BCE additionally
    # trains that boundary on true births versus existing/false proposals, even
    # on the first frame where a one-class column CE alone has zero gradient.
    birth = (F.binary_cross_entropy_with_logits(new[targets.eligible], birth_target[targets.eligible])
             if targets.eligible.any() else zero)
    switch_terms, contrast_terms = [], []
    if targets.matched_pairs:
        rows = torch.tensor([r for r, _ in targets.matched_pairs], device=pair.device)
        cols = torch.tensor([c for _, c in targets.matched_pairs], device=pair.device)
        competing_row = row_logits[rows].clone()
        competing_row.scatter_(1, cols[:, None], -1e4)
        competing_col = col_logits[cols].clone()
        competing_col.scatter_(1, rows[:, None], -1e4)
        competitor = torch.maximum(competing_row.amax(dim=1), competing_col.amax(dim=1))
        switch_terms.append(F.relu(.5 + competitor - pair[rows, cols]).mean())
        cand = F.normalize(output["candidate_id_tokens"][cols].float(), dim=-1)
        for reference in (state.id_memory, state.identity_anchor):
            similarity = cand @ F.normalize(reference.float(), dim=-1).T / .07
            if targets.row_supervised is not None:
                similarity = similarity.masked_fill(~supervised[None], -1e4)
            contrast_terms.append(F.cross_entropy(similarity, rows))
        motion_target = _motion_delta(state.boxes[rows].float(), targets.gt_boxes.float()) / output["frame_delta"]
        motion = F.smooth_l1_loss(output["motion_delta"][rows].float(), motion_target)
    else:
        motion = output["motion_delta"].sum() * 0
    switch = torch.stack(switch_terms).mean() if switch_terms else zero
    contrast = torch.stack(contrast_terms).mean() if contrast_terms else zero
    alive = (F.binary_cross_entropy_with_logits(output["alive_logits"][supervised].float(), targets.alive[supervised].float())
             if supervised.any() else zero)
    col = col_ce + birth
    total = row + col + .25 * switch + .25 * alive + .50 * motion + .25 * contrast
    return {"total": total, "row": row, "col": col, "switch": switch, "alive": alive,
            "motion": motion, "identity_contrast": contrast, "col_ce": col_ce, "birth_reject": birth}
