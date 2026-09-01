#!/usr/bin/env python3
"""B0: small three-domain L48 semantic matcher smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from locatemot.models.l48_joint_rmot import L48SemanticMatcher  # noqa: E402
from locatemot.rmot.l48_data import sha256_file, load_bank  # noqa: E402

DATA = ROOT / "outputs/l48/data"
CONTRACT = ROOT / "outputs/l48/audit/joint_data_contract.json"
TEXT_CACHE = DATA / "text_cache.pt"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class BankStore:
    def __init__(self, limit=3):
        self.limit = int(limit)
        self.cache = OrderedDict()

    def get(self, dataset, video):
        key = (dataset, str(video))
        if key not in self.cache:
            self.cache[key] = load_bank(dataset, str(video))
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]


def relation_features(boxes: torch.Tensor, image_size):
    """Fixed query-independent nearest-neighbour relation, no GT/source use."""
    boxes = boxes.float()
    if len(image_size) >= 2:
        width, height = float(image_size[0]), float(image_size[1])
    else:
        width = max(1.0, float(boxes[:, 2].max().item()))
        height = max(1.0, float(boxes[:, 3].max().item()))
    scale = boxes.new_tensor([width, height, width, height])
    norm = boxes / scale
    centers = (norm[:, :2] + norm[:, 2:]) * 0.5
    if len(boxes) <= 1:
        return boxes.new_zeros((len(boxes), 4))
    delta = centers[:, None, :] - centers[None, :, :]
    distance = delta.square().sum(-1)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.argmin(-1)
    n_delta = delta[torch.arange(len(boxes)), nearest]
    other = norm[nearest]
    left = torch.maximum(norm[:, None, 0], other[None, :, 0])
    top = torch.maximum(norm[:, None, 1], other[None, :, 1])
    right = torch.minimum(norm[:, None, 2], other[None, :, 2])
    bottom = torch.minimum(norm[:, None, 3], other[None, :, 3])
    # The nearest candidate's normalized IoU is query-independent.
    inter = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)
    area = (norm[:, 2] - norm[:, 0]).clamp_min(0) * (norm[:, 3] - norm[:, 1]).clamp_min(0)
    other_area = (other[:, 2] - other[:, 0]).clamp_min(0) * (other[:, 3] - other[:, 1]).clamp_min(0)
    # Above pairwise tensors have shape N,N; select the same nearest index.
    iou = inter[torch.arange(len(boxes)), nearest] / (
        area + other_area[nearest] - inter[torch.arange(len(boxes)), nearest]).clamp_min(1e-6)
    return torch.cat((n_delta, iou[:, None], distance[torch.arange(len(boxes)), nearest, None].sqrt()), -1)


def unit_tensors(unit, store, text_payload):
    bank = store.get(unit["dataset"], unit["video"])
    tensors = bank["tensors"]
    begin, end = int(unit["begin"]), int(unit["end"])
    sl = slice(begin, end)
    text_index = text_payload["sentence_to_index"][unit["sentence"]]
    return {
        "clip": tensors["clip"][sl].float(),
        "history_clip": tensors["history_clip"][sl].float(),
        "geometry": tensors["geometry"][sl].float(),
        "motion": tensors["motion"][sl].float(),
        "context": tensors["context"][sl].float(),
        "lifecycle": tensors["lifecycle"][sl].float(),
        "objectness": tensors["objectness"][sl].float(),
        "relation": relation_features(tensors["box"][sl], unit.get("image_size", [])),
        "text": text_payload["token_hidden"][text_index].float(),
        "text_mask": text_payload["attention_mask"][text_index].bool(),
        "target": torch.tensor([i in set(unit["positive_indices"]) for i in range(end - begin)], dtype=torch.bool),
    }


def balanced_bce(logits, target):
    if not len(logits):
        return logits.new_zeros(())
    pos = target.bool()
    neg = ~pos
    pieces = []
    if pos.any():
        pieces.append(F.binary_cross_entropy_with_logits(logits[pos], torch.ones_like(logits[pos])))
    if neg.any():
        pieces.append(F.binary_cross_entropy_with_logits(logits[neg], torch.zeros_like(logits[neg])))
    return torch.stack(pieces).mean() if pieces else logits.new_zeros(())


def unit_loss(model, values, device):
    move = {key: value.to(device, non_blocking=True)
            for key, value in values.items() if key not in ("target", "text_mask")}
    move["text_mask"] = values["text_mask"].to(device, non_blocking=True)
    target = values["target"].to(device, non_blocking=True)
    out = model(move["clip"], move["history_clip"], move["geometry"],
                move["motion"], move["context"], move["lifecycle"],
                move["objectness"], move["text"], move["text_mask"],
                move["relation"])
    score = out["semantic_logit"]
    score.retain_grad()
    pos = torch.nonzero(target, as_tuple=False).flatten()
    neg = torch.nonzero(~target, as_tuple=False).flatten()
    zero = score.new_zeros(())
    with torch.no_grad():
        hard = neg[torch.argsort(score.detach()[neg], descending=True)[:min(24, len(neg))]] if len(neg) else neg
    bce = balanced_bce(score, target)
    hard_bce = F.binary_cross_entropy_with_logits(score[hard], torch.zeros_like(score[hard])) if len(hard) else zero
    pair = F.softplus(.2 + score[hard][None, :] - score[pos][:, None]).mean() if len(pos) and len(hard) else zero
    listwise = torch.logsumexp(score, 0) - torch.logsumexp(score[pos], 0) if len(pos) else zero
    min_positive = F.softplus(.2 + score[hard].max() - score[pos]).mean() if len(pos) and len(hard) else zero
    inactive = balanced_bce(score, torch.zeros_like(target)) if not len(pos) else zero
    total = bce + .5 * hard_bce + pair + .5 * listwise + .5 * min_positive + .25 * inactive
    return total, {"total": float(total.detach()), "membership_bce": float(bce.detach()),
                   "hard_bce": float(hard_bce.detach()), "pairwise": float(pair.detach()),
                   "listwise_all_positive": float(listwise.detach()),
                   "min_positive": float(min_positive.detach()), "inactive": float(inactive.detach()),
                   "positive_count": int(len(pos)), "negative_count": int(len(neg)),
                   "hard_count": int(len(hard)), "positive_grad_nonzero": 0.0,
                   "hard_grad_nonzero": 0.0, "stream_norms": out["stream_norms"]}, score, pos, hard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/l48/train/semantic_smoke100")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    try:
        contract = json.loads(CONTRACT.read_text())
        if contract.get("decision") != "enter_B0":
            raise RuntimeError("data contract is not enter_B0")
        units = load_jsonl(DATA / "train_units.jsonl")
        if not units or len({u["dataset"] for u in units}) != 3:
            raise RuntimeError("B0 train units do not cover all three domains")
        text_payload = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        required_sentences = {u["sentence"] for u in units}
        if not required_sentences.issubset(text_payload["sentence_to_index"]):
            raise RuntimeError("text cache misses a train unit sentence")
        by_domain_category = defaultdict(list)
        for unit in units:
            by_domain_category[(unit["dataset"], unit["category"])].append(unit)
        domains = sorted({u["dataset"] for u in units})
        categories = sorted({u["category"] for u in units})
        store = BankStore(limit=3)
        device = torch.device(args.device)
        model = L48SemanticMatcher(hidden=args.hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        rng = random.Random(args.seed)
        trace, grad_trace = [], []
        sample_counts = Counter()
        last_values = None
        model.train()
        autocast_enabled = device.type == "cuda"
        for step in range(1, args.steps + 1):
            domain = domains[(step - 1) % len(domains)]
            available = [c for c in categories if by_domain_category[(domain, c)]]
            category = available[(step - 1) % len(available)]
            unit = rng.choice(by_domain_category[(domain, category)])
            sample_counts[(domain, category)] += 1
            values = unit_tensors(unit, store, text_payload)
            last_values = values
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss, parts, score, pos, hard = unit_loss(model, values, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            if score.grad is not None:
                if len(pos):
                    parts["positive_grad_nonzero"] = float((score.grad[pos].abs() > 1e-12).float().mean().detach().cpu())
                if len(hard):
                    parts["hard_grad_nonzero"] = float((score.grad[hard].abs() > 1e-12).float().mean().detach().cpu())
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if not np.isfinite(norm):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            optimizer.step()
            parts["step"] = step
            parts["domain"] = domain
            parts["category"] = category
            trace.append(parts)
            grad_trace.append(norm)
        checkpoint = out / f"checkpoint_semantic_step{args.steps}.pt"
        payload = {"format": "locatemot-l48-semantic-matcher-v1", "stage": "B0-semantic-smoke",
                   "model": model.state_dict(), "config": model.config(),
                   "seed": args.seed, "steps": args.steps,
                   "data_contract_sha256": sha256_file(CONTRACT),
                   "text_cache": str(TEXT_CACHE.resolve()),
                   "text_cache_sha256": sha256_file(TEXT_CACHE),
                   "screening_gt_used": False, "ordinary_mot_ovmot_touched": False}
        torch.save(payload, checkpoint)
        reload_model = L48SemanticMatcher(hidden=args.hidden).to(device)
        reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
        reload_model.eval()
        with torch.inference_mode():
            test = unit_tensors(units[0], store, text_payload)
            moved = {key: value.to(device) for key, value in test.items()}
            reload_out = reload_model(moved["clip"], moved["history_clip"], moved["geometry"],
                                      moved["motion"], moved["context"], moved["lifecycle"],
                                      moved["objectness"], moved["text"], moved["text_mask"],
                                      moved["relation"])
        reload_finite = bool(torch.isfinite(reload_out["semantic_logit"]).all())
        if not reload_finite:
            raise FloatingPointError("reloaded semantic logits are nonfinite")
        (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        metrics = {
            "format": "locatemot-l48-semantic-smoke-metrics-v1", "stage": "B0",
            "seed": args.seed, "steps": args.steps, "device": str(device),
            "hidden": args.hidden, "train_units_available": len(units),
            "sample_counts": {f"{d}|{c}": n for (d, c), n in sorted(sample_counts.items())},
            "loss_mean": {key: float(np.mean([row[key] for row in trace]))
                          for key in ("total", "membership_bce", "hard_bce", "pairwise",
                                      "listwise_all_positive", "min_positive", "inactive")},
            "loss_last": {key: trace[-1][key] for key in
                          ("total", "membership_bce", "hard_bce", "pairwise",
                           "listwise_all_positive", "min_positive", "inactive")},
            "gradient_norm": {"mean": float(np.mean(grad_trace)), "max": float(np.max(grad_trace)),
                              "nonzero_steps": int(sum(x > 0 for x in grad_trace))},
            "positive_grad_nonzero_mean": float(np.mean([x["positive_grad_nonzero"] for x in trace])),
            "hard_grad_nonzero_mean": float(np.mean([x["hard_grad_nonzero"] for x in trace])),
            "finite_loss": all(np.isfinite(x["total"]) for x in trace),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_reload": reload_finite,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "elapsed_sec": time.time() - start, "data_contract_sha256": sha256_file(CONTRACT),
            "text_cache_sha256": sha256_file(TEXT_CACHE),
            "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id_as_feature"],
            "token_span_region_alignment": "UNALIGNED", "static_motion_language_mask": "UNALIGNED/not claimed",
            "screening_gt_used": False, "ordinary_mot_ovmot_touched": False,
        }
        (out / f"metrics_semantic_step{args.steps}.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text(
            "# L48 B0 semantic smoke incomplete\n\n"
            f"First actionable error: `{type(exc).__name__}: {exc}`\n"
        )
        raise


if __name__ == "__main__":
    main()
