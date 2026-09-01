"""Train Stage L20 SINT-Set on the frozen dual observation bank.

Phase A trains source-invariant adapters, grouped observations, null-aware
set retrieval, and hard-negative ranking.  Fragment-pair graph supervision is
kept as a separate Phase B because it is only meaningful after set retrieval
has passed its detection gate.  All outputs are RMOT-only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from locatemot.models.l20_source_invariant_set_correspondence import (  # noqa: E402
    L20SourceInvariantSetCorrespondence,
)
from tools.l20_common import (  # noqa: E402
    BankStore, TextStore, build_l20_buckets, l20_frame_features,
    l20_frame_targets,
)
from tools.train_l18_carr import load_items  # noqa: E402
from tools.train_l19 import choose_item, choose_verified_swap, detach_state  # noqa: E402


VARIANTS = {
    # A0 is the frozen L19 checkpoint and is not retrained here.
    "A1": {"adapters": False, "grouping": False, "alignment": False,
           "hard": False, "null": False},
    "A2": {"adapters": True, "grouping": False, "alignment": False,
           "hard": False, "null": False},
    "A3": {"adapters": True, "grouping": False, "alignment": True,
           "hard": False, "null": False},
    "A4": {"adapters": True, "grouping": True, "alignment": True,
           "hard": False, "null": False},
    "A5": {"adapters": True, "grouping": True, "alignment": True,
           "hard": True, "null": False},
    "A6": {"adapters": True, "grouping": True, "alignment": True,
           "hard": True, "null": True},
}


AUDIT_SCHEMA_VERSION = "locatemot-l20-blocking-sanity-audit-v2"
AUDIT_CONFIG_ID = "strict-mutual-nearest-iou0.80-app0.82-grid0.70-0.80-v1"
CURRENT_AUDIT = ROOT / "outputs/l20/protocol/l20_blocking_sanity_audit.current.json"
AUDIT_RUN_DIR = ROOT / "outputs/l20/protocol/blocking_audit_runs"
AUDIT_LOCK_PATH = ROOT / "outputs/l20/protocol/l20_blocking_sanity_audit.lock"


def _audit_time(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError("L20 blocking audit has no completed_at")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"invalid L20 audit timestamp: {value!r}") from error


def load_current_blocking_audit() -> tuple[Path, dict]:
    """Accept only a self-consistent, latest promoted L20 audit snapshot."""
    try:
        import fcntl
        with AUDIT_LOCK_PATH.open("a+") as lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "another L20 blocking audit is writing; trainer refuses stale/current race"
                ) from error
            audit = json.loads(CURRENT_AUDIT.read_text()) if CURRENT_AUDIT.exists() else None
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"run the new promoted L20 blocking audit before training: {CURRENT_AUDIT}"
        ) from error
    if audit is None:
        raise RuntimeError(
            f"run the new promoted L20 blocking audit before training: {CURRENT_AUDIT}")
    required = ("schema_version", "audit_code", "audit_config_id", "run_id",
                "started_at", "completed_at", "audit_status",
                "current_audit_path", "run_report_path")
    missing = [key for key in required if key not in audit]
    if missing:
        raise RuntimeError(
            f"L20 current audit is stale/missing required fields: {missing}")
    if audit["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise RuntimeError(
            f"L20 current audit schema mismatch: {audit['schema_version']}")
    if audit["audit_code"] != "tools/audit_l20_blocking.py":
        raise RuntimeError("L20 current audit code identifier mismatch")
    if audit["audit_config_id"] != AUDIT_CONFIG_ID:
        raise RuntimeError("L20 current audit configuration mismatch")
    if audit["audit_status"] != "complete" or audit.get("blocking_passed") is not True:
        raise RuntimeError("L20 current audit is not a completed passing audit")
    if audit["current_audit_path"] != str(CURRENT_AUDIT):
        raise RuntimeError("L20 current audit path is not the promoted current file")
    started = _audit_time(audit["started_at"])
    completed = _audit_time(audit["completed_at"])
    if completed < started:
        raise RuntimeError("L20 audit completed_at precedes started_at")

    decision = audit.get("grouping_decision", {})
    if (decision.get("use_grouping") is not True or
            float(decision.get("selected_iou_threshold", -1.0)) != 0.80 or
            float(decision.get("selected_appearance_threshold", -1.0)) != 0.82):
        raise RuntimeError(
            "L20 current audit does not select the required strict IoU=0.80/appearance=0.82 rule")
    selected = [value for value in audit.get("strict_group_threshold_audit", [])
                if float(value.get("iou_threshold", -1.0)) == 0.80 and
                float(value.get("appearance_threshold", -1.0)) == 0.82]
    if len(selected) != 1 or selected[0].get("aggregate", {}).get("passed") is not True:
        raise RuntimeError("L20 current audit lacks a passing strict .80/.82 threshold record")

    run_path = Path(audit["run_report_path"])
    if not run_path.is_absolute():
        run_path = (ROOT / run_path).resolve()
    if run_path.parent.resolve() != AUDIT_RUN_DIR.resolve() or not run_path.exists():
        raise RuntimeError("L20 current audit run_report_path is missing or outside blocking_audit_runs")
    run = json.loads(run_path.read_text())
    for key in ("schema_version", "audit_config_id", "run_id", "completed_at",
                "audit_status"):
        if run.get(key) != audit.get(key):
            raise RuntimeError(f"L20 current/run audit mismatch in {key}")
    if run.get("audit_status") != "complete":
        raise RuntimeError("L20 referenced run report is not complete")

    # A newer successful run must always have promoted itself.  Refuse a
    # current snapshot that is older than such a run, which is the stale-audit
    # failure mode this gate is intended to prevent.
    for candidate in AUDIT_RUN_DIR.glob("*.json"):
        if candidate == run_path:
            continue
        try:
            candidate_report = json.loads(candidate.read_text())
            if (candidate_report.get("schema_version") == AUDIT_SCHEMA_VERSION and
                    candidate_report.get("audit_config_id") == AUDIT_CONFIG_ID and
                    candidate_report.get("audit_status") == "complete" and
                    _audit_time(candidate_report["completed_at"]) > completed):
                raise RuntimeError(
                    f"L20 current audit is older than successful run {candidate}")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return CURRENT_AUDIT, audit


def expression_text(entry: dict) -> str:
    return str(entry.get("sentence", entry.get("expression", "")))


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not len(logits):
        return logits.new_zeros(())
    target = target.to(logits.dtype)
    positive = target > 0.5
    negative = ~positive
    weights = torch.ones_like(target)
    if positive.any():
        weights[positive] = (weights.new_tensor(0.5) /
                             positive.float().sum().clamp_min(1.0)).to(weights.dtype)
    if negative.any():
        weights[negative] = (weights.new_tensor(0.5) /
                             negative.float().sum().clamp_min(1.0)).to(weights.dtype)
    return (weights * F.binary_cross_entropy_with_logits(
        logits, target, reduction="none")).sum()


def set_ranking_loss(logits: torch.Tensor, target: torch.Tensor,
                     null_logit: torch.Tensor, null_target: float,
                     use_null: bool, margin: float = 0.35) -> torch.Tensor:
    """Multi-positive listwise ranking with an explicit NULL alternative."""
    if not len(logits):
        return logits.new_zeros(())
    positive = target > 0.5
    negative = ~positive
    terms = []
    if positive.any() and negative.any():
        negatives = logits[negative]
        negatives = torch.topk(
            negatives, min(len(negatives), max(8, 4 * int(positive.sum())))
        ).values
        terms.append(F.softplus(
            margin - logits[positive, None] + negatives[None, :]).mean())
    if use_null:
        if positive.any():
            # Covered frame: every positive group must beat NULL.
            terms.append(F.softplus(
                margin + null_logit - logits[positive]).mean())
        elif null_target > 0.5:
            # ABSENT/PRESENT_UNCOVERED frame: NULL must beat every group.
            terms.append(F.softplus(
                margin + logits.max() - null_logit).mean())
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def hard_negative_margin(logits: torch.Tensor, target: torch.Tensor,
                         source: torch.Tensor, enabled: bool,
                         margin: float = 0.45) -> torch.Tensor:
    """Mine source-internal and cross-source top negatives in the frame."""
    if not enabled or not len(logits):
        return logits.new_zeros(())
    positive = target > 0.5
    terms = []
    for source_id in (0, 1, 2):
        pos = positive & (source == source_id)
        neg = (~positive) & (source == source_id)
        if pos.any() and neg.any():
            hard = torch.topk(neg.float() * logits +
                              (~neg).float() * logits.detach().min() -
                              (~neg).float() * 1e6,
                              min(int(neg.sum()), max(2, 2 * int(pos.sum())))).values
            terms.append(F.softplus(margin - logits[pos, None] + hard[None, :]).mean())
    if positive.any() and (~positive).any():
        hard = logits[~positive]
        hard = torch.topk(hard, min(len(hard), max(8, 3 * int(positive.sum())))).values
        terms.append(F.softplus(
            margin - logits[positive, None] + hard[None, :]).mean())
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def hard_negative_bucket_count(target: torch.Tensor, enabled: bool) -> int:
    """Count the actual frame-level top-negative bucket used by the loss."""
    if not enabled or not len(target):
        return 0
    positive = target > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        return 0
    return min(int(negative.sum()), max(8, 3 * int(positive.sum())))


def cross_pool_alignment_loss(output: dict, enabled: bool) -> torch.Tensor:
    """Align true same-group cross-pool views while retaining hard negatives."""
    if not enabled:
        return output["logits"].new_zeros(())
    values = output.get("row_features")
    groups = output.get("row_group_ids")
    source = output.get("row_source")
    if values is None or len(values) < 2:
        return output["logits"].new_zeros(())
    values = F.normalize(values, dim=-1)
    terms = []
    for group_id in torch.unique(groups).detach().cpu().tolist():
        indices = torch.nonzero(groups == int(group_id),
                                as_tuple=False).flatten()
        main = indices[source[indices] == 0]
        reserve = indices[source[indices] == 1]
        if not len(main) or not len(reserve):
            continue
        anchor = values[main].mean(0, keepdim=True)
        positive = values[reserve].mean(0, keepdim=True)
        positive_score = (anchor * positive).sum(-1)
        other = torch.nonzero(groups != int(group_id),
                              as_tuple=False).flatten()
        if len(other):
            negative_scores = (anchor @ values[other].transpose(0, 1)).reshape(-1)
            hard = torch.topk(negative_scores, min(len(negative_scores), 8)).values
            terms.append(F.softplus(0.25 - positive_score[:, None] + hard[None, :]).mean())
        terms.append(1.0 - positive_score.mean())
    return torch.stack(terms).mean() if terms else output["logits"].new_zeros(())


def l20_item_episode(item: dict, store: BankStore, text_store: TextStore,
                     rng: random.Random, sequence_length: int, burn_in: int,
                     device: torch.device,
                     training_conflict_singletons: bool = True):
    bank = store.get(item["bank_dataset"], item["video"])
    if "l19_track_membership" not in bank:
        from tools.train_l19 import l19_track_membership_index
        bank["l19_track_membership"] = l19_track_membership_index(bank)
    entry = item["entry"]
    frame_count = len(bank["tensors"]["frame_ids"])
    total_length = min(frame_count, int(sequence_length))
    loss_length = max(1, total_length - int(burn_in))
    positive_frames = [
        bank["frame_to_index"][int(frame)]
        for frame, ids in entry.get("label", {}).items()
        if ids and int(frame) in bank["frame_to_index"]
    ]
    if positive_frames and rng.random() < 0.75:
        anchor = rng.choice(positive_frames)
        start = max(0, min(frame_count - total_length,
                           anchor - rng.randrange(loss_length)))
        mode = "positive_covered"
    else:
        start = rng.randrange(max(1, frame_count - total_length + 1))
        mode = "random_or_empty"
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    text = expression_text(entry)
    family = expression_family_vector(text).to(device)
    tokens, mask = text_store.get(text, device)
    episode = []
    for frame_index in range(start, start + total_length):
        features, track_ids, begin, end = l20_frame_features(
            bank, frame_index, device,
            training_conflict_singletons=training_conflict_singletons)
        frame_id = int(bank["tensors"]["frame_ids"][frame_index])
        targets = l20_frame_targets(
            bank, begin, end, entry, frame_id, bank["l19_track_membership"],
            training_conflict_singletons=training_conflict_singletons)
        episode.append({
            "features": features, "track_ids": track_ids,
            "target": targets, "frame": frame_id,
        })
    return item, query, family, tokens, mask, episode, mode


def _group_targets(row: dict, output: dict) -> dict[str, torch.Tensor]:
    target = row["target"]
    member_rows = output.get("group_member_rows")
    if member_rows is not None:
        # Output groups are refined by the model and therefore need not have
        # the same IDs or cardinality as the target's frame groups.  Labels
        # are row labels; aggregate over exactly the rows emitted by each
        # refined group.
        values = {}
        for name in ("membership", "observation", "presence", "group_target"):
            row_values = np.asarray(target["row_" + {
                "membership": "membership", "observation": "match",
                "presence": "presence", "group_target": "match",
            }[name]], np.float32)
            values[name] = [float(row_values[members.detach().cpu().numpy()].max())
                            if len(members) else 0.0
                            for members in member_rows]
        row_source = np.asarray(target["source"], np.int64)
        # Target source is already group-level for strict groups, but deriving
        # it from member rows keeps the mapping correct when grouping is off.
        source_values = []
        row_pool = row.get("features", {}).get("pool_id")
        if row_pool is not None:
            row_pool = row_pool.detach().cpu().numpy()
        for members in member_rows:
            indices = members.detach().cpu().numpy()
            if row_pool is not None and len(indices):
                sources = set(int(value) for value in row_pool[indices].tolist())
                source_values.append(next(iter(sources)) if len(sources) == 1 else 2)
            elif len(indices):
                source_values.append(int(row_source[indices[0]]) if len(row_source)
                                    else 0)
            else:
                source_values.append(0)
        return {
            "membership": torch.as_tensor(values["membership"],
                                          device=output["logits"].device),
            "observation": torch.as_tensor(values["observation"],
                                           device=output["logits"].device),
            "presence": torch.as_tensor(values["presence"],
                                        device=output["logits"].device),
            "target": torch.as_tensor(values["group_target"],
                                       device=output["logits"].device),
            "source": torch.as_tensor(source_values,
                                      device=output["logits"].device,
                                      dtype=torch.long),
        }
    indices = output["group_row_indices"].detach().cpu().tolist()
    return {
        "membership": torch.as_tensor(target["membership"][indices],
                                      device=output["logits"].device),
        "observation": torch.as_tensor(target["observation"][indices],
                                        device=output["logits"].device),
        "presence": torch.as_tensor(target["presence"][indices],
                                     device=output["logits"].device),
        "target": torch.as_tensor(target["group_target"][indices],
                                   device=output["logits"].device),
        "source": torch.as_tensor(target["source"][indices],
                                   device=output["logits"].device, dtype=torch.long),
    }


def run_episode(model, query, family, tokens, mask, episode: list[dict],
                burn_in: int, bptt_chunk: int, options: dict):
    state = {}
    last_seen = {}
    context = model.query_context(tokens, query, family, mask)
    chunk_losses, frame_losses = [], []
    stats = Counter()
    for index, row in enumerate(episode):
        if index < int(burn_in):
            with torch.no_grad():
                output = model(row["features"], query, family, row["track_ids"],
                               state, query_tokens=tokens, query_mask=mask,
                               query_context=context)
            state = output["state"]
            last_seen.update({int(track_id): index for track_id in
                              row["track_ids"].detach().cpu().tolist()})
            continue
        output = model(row["features"], query, family, row["track_ids"], state,
                       query_tokens=tokens, query_mask=mask,
                       query_context=context)
        state = output["state"]
        last_seen.update({int(track_id): index for track_id in
                          row["track_ids"].detach().cpu().tolist()})
        state = {key: value for key, value in state.items()
                 if index - last_seen.get(int(key), index) <= 12}
        labels = _group_targets(row, output)
        null_target = float(row["target"]["null_target"])
        membership = balanced_bce(output["membership_logits"], labels["membership"])
        observation = balanced_bce(output["observation_logits"], labels["observation"])
        presence = balanced_bce(output["presence_logits"], labels["presence"])
        set_loss = set_ranking_loss(
            output["logits"], labels["target"], output["null_logit"],
            null_target, options["null"])
        null_bce = balanced_bce(
            output["null_logit"].reshape(1),
            output["null_logit"].new_tensor([null_target])) if options["null"] else \
            output["logits"].new_zeros(())
        hard = hard_negative_margin(
            output["logits"], labels["target"], labels["source"],
            options["hard"])
        align = cross_pool_alignment_loss(output, options["alignment"])
        row_match = torch.as_tensor(
            row["target"]["row_match"], device=output["logits"].device)
        view = balanced_bce(output["row_quality_logits"], row_match)
        loss = (1.00 * set_loss + 0.60 * membership +
                0.50 * observation + 0.10 * presence +
                0.70 * null_bce + 0.40 * hard +
                0.30 * align + 0.15 * view)
        frame_losses.append(loss)
        positive_groups = int((labels["target"] > 0.5).sum())
        negative_groups = int((labels["target"] <= 0.5).sum())
        hard_groups = hard_negative_bucket_count(labels["target"], options["hard"])
        stats.update({
            "frames": 1, "groups": int(len(output["logits"])),
            "positive_groups": positive_groups,
            "negative_groups": negative_groups,
            "main_positive": int(((labels["target"] > 0.5) &
                                   (labels["source"] == 0)).sum()),
            "reserve_positive": int(((labels["target"] > 0.5) &
                                      (labels["source"] == 1)).sum()),
            "null_frames": int(null_target > 0.5),
            "covered_frames": int(null_target <= 0.5),
            "hard_negative_frames": int(hard_groups > 0),
            "hard_negative_groups": hard_groups,
        })
        for name, value in (("set", set_loss), ("membership", membership),
                            ("observation", observation), ("presence", presence),
                            ("null", null_bce), ("hard", hard),
                            ("alignment", align), ("view", view)):
            stats[name] += float(value.detach())
        if len(frame_losses) >= int(bptt_chunk):
            chunk_losses.append(torch.stack(frame_losses).mean())
            frame_losses = []
            state = detach_state(state)
    if frame_losses:
        chunk_losses.append(torch.stack(frame_losses).mean())
    if not chunk_losses:
        chunk_losses = [next(model.parameters()).sum() * 0.0]
    frames = max(1, int(stats["frames"]))
    for name in ("set", "membership", "observation", "presence", "null",
                 "hard", "alignment", "view"):
        stats[name] /= frames
    return torch.stack(chunk_losses).mean(), stats


@torch.no_grad()
def proxy_validate(model, items, store, text_store, rng, episodes,
                   sequence_length, burn_in, bptt_chunk, device, options):
    model.eval()
    losses, stats = [], Counter()
    for _ in range(int(episodes)):
        _item, query, family, tokens, mask, episode, _mode = l20_item_episode(
            rng.choice(items), store, text_store, rng, sequence_length,
            burn_in, device, training_conflict_singletons=False)
        loss, row_stats = run_episode(
            model, query, family, tokens, mask, episode, burn_in,
            bptt_chunk, options)
        losses.append(float(loss))
        stats.update(row_stats)
    return {"loss": float(np.mean(losses)) if losses else None,
            "episodes": int(episodes), "stats": dict(stats)}


def gradient_check(model, device):
    model.zero_grad(set_to_none=True)
    n = 4
    features = {
        "clip": torch.randn(n, 512, device=device),
        "history_clip": torch.randn(n, 512, device=device),
        "pbd": torch.randn(n, 2048, device=device),
        "uidm_h": torch.randn(n, 384, device=device),
        "geometry": torch.randn(n, 7, device=device),
        "motion": torch.randn(n, 8, device=device),
        "context": torch.randn(n, 8, device=device),
        "lifecycle": torch.randn(n, 8, device=device),
        "objectness": torch.rand(n, device=device),
        "pool_id": torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device),
        "observation_group_id": torch.tensor([1, 1, 2, 3], dtype=torch.long,
                                              device=device),
    }
    query = torch.randn(512, device=device)
    family = torch.zeros(8, device=device)
    ids = torch.tensor([11, 12, 13, 14], dtype=torch.long, device=device)
    model.detach_state = False
    first = model(features, query, family, ids, {})
    for value in first["state"].values():
        value["memory"].retain_grad()
    second = model(features, query, family, ids, first["state"])
    second["logits"].sum().backward()
    gradients = [value["memory"].grad for value in first["state"].values()
                 if value["memory"].grad is not None]
    norm = float(sum(value.abs().sum() for value in gradients)) if gradients else 0.0
    model.zero_grad(set_to_none=True)
    model.detach_state = True
    return {"nonzero": bool(norm > 1e-9), "state_gradient_l1": norm,
            "checked_tracks": len(gradients)}


def save_checkpoint(path, model, optimizer, scheduler, args, protocol, options,
                    step, validation, sampler_stats, gradient):
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "hidden": args.hidden, "heads": args.heads, "dropout": args.dropout,
        "token_dim": 512, "temporal_points": args.temporal_points,
        "hook_points": args.hook_points, "sequence_length": args.sequence_length,
        "burn_in": args.burn_in, "bptt_chunk": args.bptt_chunk,
        "seed": args.seed, "variant": args.variant,
        "use_source_adapters": options["adapters"],
        "use_grouping": options["grouping"], "use_null": options["null"],
        "enable_alignment": options["alignment"],
        "enable_hard_negative": options["hard"],
        "source_invariant_shared_head": True,
        "carr_coverage_gate": False, "heuristic_linker": False,
        "rmot_only": True,
        "bank_root": str((ROOT / args.bank_root).resolve()),
    }
    checkpoint = {
        "model": model.state_dict(), "model_name": "l20_sint_set",
        "cfg": cfg, "protocol": protocol, "step": int(step),
        "validation_proxy": validation,
        "bank_root": str((ROOT / args.bank_root).resolve()),
        "text_root": str((ROOT / args.text_root).resolve()),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(), "sampler_stats": sampler_stats,
        "temporal_gradient_check": gradient,
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def checkpoint_loadability(path: Path, args, options: dict) -> dict:
    """Reload the just-written state dict with the exact recorded topology."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["cfg"]
        reloaded = L20SourceInvariantSetCorrespondence(
            cfg["hidden"], cfg["heads"], dropout=cfg["dropout"],
            temporal_points=cfg["temporal_points"],
            hook_points=cfg["hook_points"],
            use_source_adapters=cfg["use_source_adapters"],
            use_grouping=cfg["use_grouping"], use_null=cfg["use_null"])
        reloaded.load_state_dict(checkpoint["model"])
        return {"passed": True, "model_name": checkpoint["model_name"],
                "step": int(checkpoint["step"]),
                "parameter_count": sum(value.numel()
                                        for value in reloaded.parameters())}
    except Exception as error:
        return {"passed": False, "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="A6")
    parser.add_argument("--out", default="outputs/l20/checkpoints/l20_a6_diag.pt")
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--text-root", default="outputs/l18/data/text_cache")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--burn-in", type=int, default=24)
    parser.add_argument("--bptt-chunk", type=int, default=24)
    parser.add_argument("--temporal-points", type=int, default=8)
    parser.add_argument("--hook-points", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--cache-size", type=int, default=1)
    parser.add_argument("--swap-prob", type=float, default=0.20)
    args = parser.parse_args()
    if args.sequence_length <= args.burn_in:
        raise ValueError("sequence length must exceed burn-in")
    options = dict(VARIANTS[args.variant])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items, protocol = load_items()
    audit_path, audit = load_current_blocking_audit()
    # Honor the audited grouping decision.  If the audit ever selects the
    # mandated raw-row fallback, A4--A6 automatically remain ungrouped.
    options["grouping"] = bool(options["grouping"] and
                                audit["grouping_decision"]["use_grouping"])
    store = BankStore((ROOT / args.bank_root).resolve(), args.cache_size)
    text_store = TextStore((ROOT / args.text_root).resolve())
    buckets, _bucket_metadata = build_l20_buckets(items["train"], store)
    required = {"reserve_positive", "present_uncovered", "hard_negative"}
    if not required.intersection(buckets):
        raise RuntimeError("L20 training buckets lack reserve/uncovered examples")
    by_video = defaultdict(list)
    for domain in ("kitti", "dance"):
        for item in items["train"][domain]:
            by_video[item["video"]].append(item)
    model = L20SourceInvariantSetCorrespondence(
        args.hidden, args.heads, dropout=args.dropout,
        temporal_points=args.temporal_points, hook_points=args.hook_points,
        use_source_adapters=options["adapters"],
        use_grouping=options["grouping"], use_null=options["null"],
    ).to(device)
    model.detach_state = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates = max(1, args.steps // args.accumulate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=args.lr * 0.05)
    gradient = gradient_check(model, device)
    sampler_stats = {
        "variant": args.variant,
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "sampled": Counter(), "swap_samples": [], "gradient_check": gradient,
    }
    rng = random.Random(args.seed + 1009)
    history = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        model.train()
        item, bucket = choose_item(items["train"], buckets, rng)
        sampler_stats["sampled"][bucket] += 1
        original_item = item
        swap = None
        if args.swap_prob > 0 and rng.random() < args.swap_prob:
            item, swap = choose_verified_swap(item, by_video, rng)
            if swap is not None:
                sampler_stats["sampled"]["true_query_swap"] += 1
                if len(sampler_stats["swap_samples"]) < 64:
                    sampler_stats["swap_samples"].append(swap)
        item, query, family, tokens, mask, episode, mode = l20_item_episode(
            item, store, text_store, rng, args.sequence_length, args.burn_in,
            device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, stats = run_episode(
                model, query, family, tokens, mask, episode, args.burn_in,
                args.bptt_chunk, options)
            (loss / max(1, args.accumulate)).backward()
        if step % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        row = {
            "step": step, "loss": float(loss.detach()), "bucket": bucket,
            "mode": mode, "query": expression_text(item["entry"]),
            "original_query": expression_text(original_item["entry"]),
            "stats": dict(stats), "true_query_swap": swap is not None,
        }
        history.append(row)
        if step == 1 or step % 25 == 0:
            print(f"[l20] variant={args.variant} step={step}/{args.steps} "
                  f"loss={row['loss']:.4f} bucket={bucket} "
                  f"groups={int(stats['groups'])} null={int(stats['null_frames'])}",
                  flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            validation = proxy_validate(
                model, items["train_val"]["kitti"] + items["train_val"]["dance"],
                store, text_store, random.Random(args.seed + step),
                args.validation_episodes, args.sequence_length, args.burn_in,
                args.bptt_chunk, device, options)
            row["validation"] = validation
            out = (ROOT / args.out).resolve()
            save_checkpoint(
                out.with_name(f"{out.stem}_step{step}.pt"), model, optimizer,
                scheduler, args, protocol, options, step, validation,
                sampler_stats, gradient)
            print(f"[l20-val] step={step} loss={validation['loss']:.5f} "
                  f"stats={validation['stats']}", flush=True)
    out = (ROOT / args.out).resolve()
    save_checkpoint(
        out, model, optimizer, scheduler, args, protocol, options, args.steps,
        history[-1].get("validation") if history else None,
        sampler_stats, gradient)
    loadability = checkpoint_loadability(out, args, options)
    serial_sampler = dict(sampler_stats)
    serial_sampler["sampled"] = dict(serial_sampler["sampled"])
    out.with_suffix(out.suffix + ".json").write_text(json.dumps({
        "args": vars(args), "options": options, "steps": args.steps,
        "wall_seconds": time.time() - started,
        "bucket_metadata_count": len(_bucket_metadata),
        "sampler_stats": serial_sampler, "history": history,
        "checkpoint_loadability": loadability,
        "blocking_audit": str(audit_path),
    }, indent=2) + "\n")
    print(f"[l20] done seconds={time.time() - started:.1f} "
          f"gradient={gradient} checkpoint_loadability={loadability}", flush=True)


if __name__ == "__main__":
    main()
