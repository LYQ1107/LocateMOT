#!/usr/bin/env python3
"""Train the L50 single-variable semantic generalization package.

The model is exactly the frozen-input, hidden=256 L49 semantic core.  This
entry point changes only sampling/loss/feature-level temporal augmentation.
It consumes the train-only L49 fit units and the immutable L29 fit score cache;
no official test path is opened.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l50_domain_balanced_semantic import (  # noqa: E402
    L50DomainBalancedSemanticMatcher,
)
from locatemot.rmot.l49_data import (  # noqa: E402
    L49_SPLITS,
    TEXT_CACHE,
    load_bank,
    sha256_file,
    unit_features,
)

DATA = ROOT / "outputs/l49/data"
CONTRACT = ROOT / "outputs/l49/audit/kitti_data_contract.json"
L48_INIT = ROOT / "outputs/l48/train/semantic_smoke100/checkpoint_semantic_step100.pt"
L29_CACHE = ROOT / "outputs/l49/val/fit_baseline_scores_selected.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
SAVE_STEPS = (100, 250, 500, 1000, 2500, 5000, 10000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class BankStore:
    """Small LRU: no new bank/cache is written by L50."""

    def __init__(self, limit: int = 2):
        self.limit = int(limit)
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, dataset: str, video: str):
        key = f"{dataset}|{video}"
        if key not in self.cache:
            self.cache[key] = load_bank(dataset, video)
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]


def unit_key(row: dict) -> tuple[str, str, int, int]:
    return (str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"]))


def load_teacher_cache(path: Path) -> dict[tuple[str, str, int, int], torch.Tensor]:
    """Load only the immutable train-fit L29 score cache, with key checks."""
    result = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("split") not in (None, "fit"):
            raise AssertionError(f"unexpected split in L29 fit cache: {row.get('split')}")
        key = unit_key(row)
        if key in result:
            raise AssertionError(f"duplicate L29 teacher cache key: {key}")
        values = torch.as_tensor(row["score"], dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"nonfinite L29 teacher score: {key}")
        result[key] = values
    if len(result) != 5314:
        raise AssertionError(f"expected 5314 train teacher records, found {len(result)}")
    return result


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not len(logits):
        return logits.new_zeros(())
    target = target.bool()
    terms = []
    if target.any():
        terms.append(F.binary_cross_entropy_with_logits(logits[target], torch.ones_like(logits[target])))
    if (~target).any():
        terms.append(F.binary_cross_entropy_with_logits(logits[~target], torch.zeros_like(logits[~target])))
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def box_iou(box: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    left = torch.maximum(box[:, None, 0], other[None, :, 0])
    top = torch.maximum(box[:, None, 1], other[None, :, 1])
    right = torch.minimum(box[:, None, 2], other[None, :, 2])
    bottom = torch.minimum(box[:, None, 3], other[None, :, 3])
    inter = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)
    area = (box[:, 2] - box[:, 0]).clamp_min(0) * (box[:, 3] - box[:, 1]).clamp_min(0)
    other_area = (other[:, 2] - other[:, 0]).clamp_min(0) * (other[:, 3] - other[:, 1]).clamp_min(0)
    return inter / (area[:, None] + other_area[None, :] - inter).clamp_min(1e-6)


def choose_hard_negatives(score: torch.Tensor, teacher: torch.Tensor,
                          values: dict[str, torch.Tensor], bank: dict,
                          unit: dict, target: torch.Tensor, progress: float,
                          stats: Counter) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a same-frame hard set without source/pool/group semantics.

    The selection is detached and uses only current-frame candidate evidence:
    current score, L29 score, objectness, frozen pooled appearance similarity,
    and box adjacency.  The curriculum starts with a smaller hard subset and
    grows it; all candidates still enter the balanced membership/listwise loss.
    """
    negative = torch.nonzero(~target, as_tuple=False).flatten()
    if not len(negative):
        return negative, negative
    n = len(score)
    hard_fraction = 0.25 + 0.60 * float(np.clip(progress, 0.0, 1.0))
    hard_limit = min(24, max(4, int(math.ceil(len(negative) * hard_fraction))))
    candidates: set[int] = set()

    def add_top(values_tensor: torch.Tensor, limit: int):
        if len(values_tensor):
            # Some auxiliary rankings (visual/geometry) are already indexed
            # by ``negative``; model/objectness/teacher rankings are full-N.
            ranked = values_tensor if len(values_tensor) == len(negative) else values_tensor[negative]
            order = torch.argsort(ranked.detach(), descending=True)[:min(limit, len(negative))]
            candidates.update(int(x) for x in negative[order].tolist())

    add_top(score, hard_limit)
    add_top(teacher, max(4, hard_limit // 2))
    add_top(values["objectness"].reshape(-1), max(4, hard_limit // 2))

    pos = torch.nonzero(target, as_tuple=False).flatten()
    if len(pos):
        with torch.no_grad():
            normalized = F.normalize(values["clip"].float(), dim=-1)
            visual = normalized[negative] @ normalized[pos].T
            add_top(visual.max(-1).values, max(4, hard_limit // 2))
            boxes = bank["tensors"]["box"][int(unit["begin"]):int(unit["end"])].float().to(score.device)
            overlap = box_iou(boxes[negative], boxes[pos]).max(-1).values
            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            distance = (centers[negative, None] - centers[pos][None]).square().sum(-1).sqrt().min(-1).values
            add_top(overlap, max(4, hard_limit // 2))
            add_top(-distance, max(4, hard_limit // 2))
        stats["visual_or_geometry_hard_candidates"] += min(len(negative), hard_limit)

    hard = torch.as_tensor(sorted(candidates), dtype=torch.long, device=score.device)
    if len(hard) > hard_limit:
        combined = score.detach()[hard] + 0.25 * teacher.detach()[hard]
        hard = hard[torch.argsort(combined, descending=True)[:hard_limit]]
    hard = hard.to(score.device)
    easy = torch.as_tensor([int(x) for x in negative.tolist() if int(x) not in set(hard.tolist())],
                           dtype=torch.long, device=score.device)
    stats["hard_negative_selected"] += int(len(hard))
    stats["easy_negative_selected"] += int(len(easy))
    stats[f"hard_fraction_bucket_{int(hard_fraction * 10):02d}"] += 1
    return hard, easy


def augment_temporal_features(values: dict[str, torch.Tensor], rng: random.Random,
                              step: int, total_steps: int, stats: Counter) -> dict[str, torch.Tensor]:
    """Apply provenance-preserving feature-level temporal augmentation.

    Labels and frame/candidate rows are untouched.  The model still receives
    the same pooled streams; only history/motion/context observations are
    stochastically masked or mildly time-jittered in memory.
    """
    out = {key: value.clone() for key, value in values.items()}
    progress = min(1.0, float(step) / max(1, total_steps))
    history_drop = 0.05 + 0.15 * progress
    candidate_occlusion = 0.02 + 0.06 * progress
    if rng.random() < history_drop:
        out["history_clip"].zero_()
        stats["history_stream_dropout_units"] += 1
    if rng.random() < 0.35:
        scale = 0.90 + 0.20 * rng.random()
        out["motion"] = out["motion"] * scale
        stats["motion_gap_jitter_units"] += 1
    if len(out["clip"]) and rng.random() < candidate_occlusion:
        count = max(1, int(round(len(out["clip"]) * (0.05 + 0.05 * progress))))
        indices = rng.sample(range(len(out["clip"])), min(count, len(out["clip"])))
        index = torch.as_tensor(indices, dtype=torch.long)
        out["clip"][index] = 0.0
        out["context"][index] = 0.0
        stats["candidate_occlusion_rows"] += len(indices)
        stats["candidate_occlusion_units"] += 1
    stats["augmented_units"] += 1
    return out


def teacher_rank_terms(score: torch.Tensor, teacher: torch.Tensor,
                       target: torch.Tensor, hard: torch.Tensor):
    zero = score.new_zeros(())
    pos = torch.nonzero(target, as_tuple=False).flatten()
    if not len(pos) or not len(hard):
        return zero, zero, 0, 0, 0
    teacher_margin = teacher[pos, None] - teacher[hard][None, :]
    student_margin = score[pos, None] - score[hard][None, :]
    teacher_correct = teacher_margin >= 0
    teacher_error = ~teacher_correct
    preserve = F.softplus(0.10 - student_margin[teacher_correct]).mean() if teacher_correct.any() else zero
    correction = F.softplus(0.10 - student_margin[teacher_error]).mean() if teacher_error.any() else zero
    return preserve, correction, int(teacher_correct.sum()), int(teacher_error.sum()), int((teacher_correct & (student_margin < 0)).sum())


def unit_loss(model, values: dict[str, torch.Tensor], teacher_cpu: torch.Tensor,
              bank: dict, unit: dict, device: torch.device, progress: float,
              stats: Counter):
    target = values["target"].to(device, non_blocking=True)
    teacher = teacher_cpu.to(device, non_blocking=True)
    moved = {key: value.to(device, non_blocking=True) for key, value in values.items()
             if key != "target"}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(moved["clip"], moved["history_clip"], moved["geometry"], moved["motion"],
                       moved["context"], moved["lifecycle"], moved["objectness"], moved["text"],
                       moved["text_mask"], moved["relation"])
        score = output["semantic_logit"]
        score.retain_grad()
        hard, easy = choose_hard_negatives(score, teacher, moved, bank, unit, target, progress, stats)
        pos = torch.nonzero(target, as_tuple=False).flatten()
        membership = balanced_bce(score, target)
        hard_bce = F.binary_cross_entropy_with_logits(score[hard], torch.zeros_like(score[hard])) if len(hard) else score.new_zeros(())
        easy_bce = F.binary_cross_entropy_with_logits(score[easy], torch.zeros_like(score[easy])) if len(easy) else score.new_zeros(())
        if len(pos) and len(hard):
            pairwise = F.softplus(0.20 + score[hard][None, :] - score[pos][:, None]).mean()
            listwise = torch.logsumexp(score, 0) - torch.logsumexp(score[pos], 0)
            min_positive = F.softplus(0.20 + score[hard].max() - score[pos]).mean()
        else:
            pairwise = listwise = min_positive = score.new_zeros(())
        preserve, correction, correct_pairs, error_pairs, correct_flip = teacher_rank_terms(score, teacher, target, hard)
        centered = score - score.mean()
        teacher_centered = teacher - teacher.mean()
        scale_loss = F.smooth_l1_loss(
            centered / teacher_centered.std().clamp_min(0.25),
            teacher_centered / teacher_centered.std().clamp_min(0.25))
        mean_drift = F.smooth_l1_loss(score.mean(), teacher.mean())
        drift = scale_loss + 0.25 * mean_drift
        inactive = balanced_bce(score, torch.zeros_like(target)) if not target.any() else score.new_zeros(())
        # Unit losses are already means over rows/pairs.  This explicit factor
        # records the degree normalization without allowing large candidate
        # frames to dominate the optimizer.
        degree_factor = math.sqrt(32.0 / max(1, len(score)))
        degree_factor = float(np.clip(degree_factor, 0.75, 1.25))
        total = degree_factor * (
            membership + hard_bce + 0.10 * easy_bce + pairwise + 0.50 * listwise
            + 0.50 * min_positive + 1.25 * preserve + 0.25 * correction
            + 0.25 * drift + 0.25 * inactive
        )
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite L50 unit loss")
    parts = {
        "total": float(total.detach()), "membership_bce": float(membership.detach()),
        "hard_negative_bce": float(hard_bce.detach()), "easy_negative_bce": float(easy_bce.detach()),
        "pairwise": float(pairwise.detach()), "listwise_all_positive": float(listwise.detach()),
        "min_positive": float(min_positive.detach()), "teacher_preservation": float(preserve.detach()),
        "teacher_error_correction": float(correction.detach()), "score_drift": float(drift.detach()),
        "inactive": float(inactive.detach()), "degree_factor": degree_factor,
        "positive_count": int(len(pos)), "negative_count": int((~target).sum()),
        "hard_negative_count": int(len(hard)), "easy_negative_count": int(len(easy)),
        "teacher_correct_pairs": correct_pairs, "teacher_error_pairs": error_pairs,
        "teacher_correct_flip_pairs": correct_flip,
    }
    return total, parts, output, pos, hard


def save_checkpoint(out: Path, model, optimizer, step: int, args, trace, counts,
                    started: float, device: torch.device, contract_sha: str,
                    text_sha: str, teacher_sha: str):
    checkpoint = out / f"checkpoint_l50_step{step}.pt"
    payload = {
        "format": "locatemot-l50-domain-balanced-semantic-v1",
        "stage": "L50-B-targeted-or-long",
        "seed": args.seed, "steps": int(step), "model_config": model.config(),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "init_l48_checkpoint": str(L48_INIT.resolve()), "init_l48_checkpoint_sha256": sha256(L48_INIT),
        "l29_teacher_fit_cache": str(L29_CACHE.resolve()), "l29_teacher_fit_cache_sha256": teacher_sha,
        "data_contract": str(CONTRACT.resolve()), "data_contract_sha256": contract_sha,
        "text_cache": str(TEXT_CACHE.resolve()), "text_cache_sha256": text_sha,
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
        "train_only": True, "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hard_negative_contract": {
            "full_candidate_set_retained": True,
            "sources": ["current_student_score", "L29_score", "objectness", "pooled_clip_similarity", "box_adjacency"],
            "curriculum": "hard_fraction=.25->.85; maximum 24; detached selection",
            "teacher_correct_weight": 1.25, "teacher_error_weight": 0.25,
        },
        "temporal_augmentation_contract": {
            "feature_level_only": True, "labels_changed": False, "new_backbone_cache": False,
            "history_dropout": ".05->.20", "motion_gap_jitter": ".90-1.10",
            "candidate_occlusion_simulation": ".02->.08 unit probability",
        },
        "elapsed_sec": time.time() - started,
    }
    torch.save(payload, checkpoint)
    reloaded = L50DomainBalancedSemanticMatcher(hidden=256, heads=4).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True)
    reloaded.eval()
    finite = all(torch.isfinite(v).all().item() for v in reloaded.state_dict().values()
                 if torch.is_floating_point(v))
    if not finite:
        raise FloatingPointError(f"nonfinite state after reload at step {step}")
    window = [row for row in trace if int(row["step"]) > max(0, step - 100)]
    keys = ("total", "membership_bce", "hard_negative_bce", "easy_negative_bce", "pairwise",
            "listwise_all_positive", "min_positive", "teacher_preservation",
            "teacher_error_correction", "score_drift", "inactive")
    metrics = {
        "format": "locatemot-l50-training-checkpoint-metrics-v1", "step": int(step),
        "seed": args.seed, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_reload": True, "reload_state_finite": bool(finite), "window_steps": len(window),
        "loss_mean": {key: float(np.mean([row[key] for row in window])) for key in keys},
        "gradient_norm": {
            "mean": float(np.mean([row["gradient_norm"] for row in window])),
            "max": float(np.max([row["gradient_norm"] for row in window])),
            "nonzero_steps": int(sum(row["gradient_norm"] > 0 for row in window)),
        },
        "gradient_audit": {
            "positive_nonzero_mean": float(np.mean([row["positive_grad_nonzero"] for row in window])),
            "hard_nonzero_mean": float(np.mean([row["hard_grad_nonzero"] for row in window])),
        },
        "sampling": {key: int(value) for key, value in sorted(counts.items())},
        "elapsed_sec": time.time() - started, "train_only": True,
        "screening_gt_used": False, "official_test_labels_read": False,
    }
    (out / f"metrics_l50_step{step}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    del reloaded
    gc.collect()
    return checkpoint, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/l50/train/targeted500")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    try:
        contract = json.loads(CONTRACT.read_text())
        if contract.get("stage") != "L49-A":
            raise RuntimeError("unexpected L49 data contract stage")
        units = load_jsonl(DATA / "train_units.jsonl")
        if len(units) != 5314 or {x["split"] for x in units} != {"fit"}:
            raise RuntimeError("L50 requires the immutable 5314 train-fit units")
        if {x["dataset"] for x in units} != set(L49_SPLITS):
            raise RuntimeError("L50 train units are not exactly V1/V2")
        text_payload = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        if not {x["sentence"] for x in units}.issubset(text_payload["sentence_to_index"]):
            raise RuntimeError("missing frozen text representation for a train unit")
        teacher_cache = load_teacher_cache(L29_CACHE)
        expected_keys = {unit_key(x) for x in units}
        if expected_keys != set(teacher_cache):
            raise AssertionError("L29 teacher cache keys do not exactly match train units")
        contract_sha = sha256_file(CONTRACT); text_sha = sha256_file(TEXT_CACHE); teacher_sha = sha256(L29_CACHE)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        model = L50DomainBalancedSemanticMatcher(hidden=256, heads=4, dropout=0.1).to(device)
        init = torch.load(L48_INIT, map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"], strict=True)
        del init
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        rng = random.Random(args.seed)
        by_domain_video: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for unit in units:
            by_domain_video[str(unit["dataset"])][str(unit["video"])].append(unit)
        domains = tuple(sorted(by_domain_video))
        if domains != ("refer_kitti_v1", "refer_kitti_v2"):
            raise AssertionError(f"unexpected domains {domains}")
        store = BankStore(limit=2)
        trace = []
        counts = Counter()
        saved = {}
        model.train()
        for step in range(1, int(args.steps) + 1):
            domain = domains[(step - 1) % len(domains)]
            videos = sorted(by_domain_video[domain])
            video = videos[rng.randrange(len(videos))]
            unit = rng.choice(by_domain_video[domain][video])
            bank = store.get(domain, video)
            values = unit_features(unit, bank, text_payload, history=8)
            values = augment_temporal_features(values, rng, step, args.steps, counts)
            teacher = teacher_cache[unit_key(unit)]
            counts[f"sampled_domain|{domain}"] += 1
            counts[f"sampled_video|{domain}|{video}"] += 1
            counts[f"category|{unit['category']}"] += 1
            optimizer.zero_grad(set_to_none=True)
            progress = float(step - 1) / max(1, int(args.steps) - 1)
            loss, parts, output, pos, hard = unit_loss(
                model, values, teacher, bank, unit, device, progress, counts)
            loss.backward()
            grad = output["semantic_logit"].grad if output["semantic_logit"].grad is not None else None
            if grad is None:
                raise RuntimeError("semantic output gradient was not retained")
            else:
                positive_grad_nonzero = float((grad[pos].abs() > 1e-12).float().mean().detach().cpu()) if len(pos) else 0.0
                hard_grad_nonzero = float((grad[hard].abs() > 1e-12).float().mean().detach().cpu()) if len(hard) else 0.0
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if not np.isfinite(grad_norm) or grad_norm <= 0:
                raise FloatingPointError(f"invalid gradient norm at step {step}: {grad_norm}")
            optimizer.step()
            parts.update({
                "step": int(step), "domain": domain, "video": video,
                "category": str(unit["category"]), "gradient_norm": grad_norm,
                "positive_grad_nonzero": positive_grad_nonzero,
                "hard_grad_nonzero": hard_grad_nonzero,
                "score_mean": float(output["semantic_logit"].detach().mean().cpu()),
                "score_std": float(output["semantic_logit"].detach().std().cpu()),
                "teacher_mean": float(teacher.mean()), "teacher_std": float(teacher.std()),
            })
            trace.append(parts)
            if step in SAVE_STEPS and step <= args.steps:
                saved[step], _ = save_checkpoint(out, model, optimizer, step, args, trace, counts,
                                                  started, device, contract_sha, text_sha, teacher_sha)
                print(json.dumps({"event": "checkpoint", "step": step,
                                  "checkpoint": str(saved[step])}, ensure_ascii=False), flush=True)
        (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        summary_keys = ("total", "membership_bce", "hard_negative_bce", "easy_negative_bce", "pairwise",
                        "listwise_all_positive", "min_positive", "teacher_preservation",
                        "teacher_error_correction", "score_drift", "inactive")
        summary = {
            "format": "locatemot-l50-domain-balanced-training-summary-v1", "stage": "L50-B",
            "seed": args.seed, "steps": int(args.steps),
            "checkpoints": {str(k): str(v.resolve()) for k, v in saved.items()},
            "checkpoint_sha256": {str(k): sha256(v) for k, v in saved.items()},
            "train_units": len(units), "domain_sequence": list(domains),
            "sample_counts": {key: int(value) for key, value in sorted(counts.items())},
            "loss_mean": {key: float(np.mean([row[key] for row in trace])) for key in summary_keys},
            "gradient_norm": {"mean": float(np.mean([row["gradient_norm"] for row in trace])),
                              "max": float(np.max([row["gradient_norm"] for row in trace])),
                              "nonzero_steps": int(sum(row["gradient_norm"] > 0 for row in trace))},
            "gradient_audit": {
                "positive_nonzero_mean": float(np.mean([row["positive_grad_nonzero"] for row in trace])),
                "hard_nonzero_mean": float(np.mean([row["hard_grad_nonzero"] for row in trace])),
            },
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "elapsed_sec": time.time() - started, "data_contract_sha256": contract_sha,
            "text_cache_sha256": text_sha, "l29_teacher_cache_sha256": teacher_sha,
            "manifest_sha256": sha256(MANIFEST), "train_only": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "local_visual_stream": False, "parameter_expansion": False,
            "identity_sequence_enabled": False,
        }
        (out / "metrics_l50_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text(
            "# L50-B training incomplete\n\n"
            f"First actionable error: `{type(exc).__name__}: {exc}`\n"
            "Official test labels were not read; historical outputs were not modified.\n"
        )
        raise


if __name__ == "__main__":
    main()
