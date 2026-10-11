"""Base-only OVMOT classification and persistent semantic consistency."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from locatemot.unified.losses.u2_association import AssociationTargets, FrameSupervision
from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
from locatemot.unified.models.u2.track_state import UnifiedTrackState


def u2_semantic_losses(core: SharedTrackCore, output: dict, state: UnifiedTrackState,
                       targets: AssociationTargets, gt: FrameSupervision,
                       bookkeeping: dict[int, int], class_text: torch.Tensor,
                       base_class_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    # The softmax denominator is also base-only: novel classes never serve as
    # class loss targets or negative class labels. Inference uses all 1203.
    local_ids = torch.full((len(class_text),), -1, device=class_text.device, dtype=torch.long)
    local_ids[base_class_ids] = torch.arange(len(base_class_ids), device=class_text.device)
    positive = targets.candidate_category_ids >= 0
    zero = output["candidate_sem_tokens"].sum() * 0
    class_terms = []
    if positive.any():
        labels = local_ids[targets.candidate_category_ids[positive]]
        if (labels < 0).any():
            raise ValueError("Novel class cannot enter U2-B supervised classification")
        logits = core.ovmot.class_logits(output["candidate_sem_tokens"][positive], class_text[base_class_ids])
        class_terms.append(F.cross_entropy(logits, labels))
    # Supervise observed canonical memory after its actual causal GRU update.
    # External labels never become persistent runtime state fields.
    visible_classes = dict(zip(gt.identity_ids.tolist(), gt.category_ids.tolist()))
    indices, labels = [], []
    for index, track_id in enumerate(output["state"].track_ids.tolist()):
        identity = bookkeeping.get(track_id, -1)
        if identity in visible_classes:
            indices.append(index)
            labels.append(visible_classes[identity])
    if indices:
        labels_tensor = local_ids[torch.tensor(labels, device=class_text.device)]
        if (labels_tensor < 0).any():
            raise ValueError("Novel memory label cannot enter U2-B classification")
        memory = output["state"].semantic_memory[indices]
        class_terms.append(F.cross_entropy(core.ovmot.class_logits(memory, class_text[base_class_ids]), labels_tensor))
    classification = torch.stack(class_terms).mean() if class_terms else zero
    if targets.matched_pairs:
        rows = torch.tensor([row for row, _ in targets.matched_pairs], device=class_text.device)
        cols = torch.tensor([col for _, col in targets.matched_pairs], device=class_text.device)
        consistency = (1 - F.cosine_similarity(state.semantic_memory[rows].float(),
                                               output["candidate_sem_tokens"][cols].float(), dim=-1)).mean()
    else:
        consistency = zero
    # Restrict regularization to supervised pairs; unlabeled novel tracks do
    # not receive a pairwise semantic label merely for appearing in the clip.
    row_mask = targets.row_supervised
    gate = output["semantic_gate"][row_mask][:, targets.eligible]
    gate_reg = gate.float().square().mean() if gate.numel() else zero
    total = .50 * classification + .25 * consistency + .10 * gate_reg
    return {"semantic_total": total, "open_vocab_class": classification,
            "semantic_consistency": consistency, "semantic_gate_reg": gate_reg}
