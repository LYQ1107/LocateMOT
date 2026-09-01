"""Fair no-tracker linear-probe comparison for the old bank and L22 bank v2.

The probe is the frozen-feature upper-limit test requested by Stage L22.  It
uses identical sampled query-frame units, labels, bank rows, optimizer steps,
and evaluation code for every feature variant.  It never invokes a tracker or
TrackEval and never writes a checkpoint outside its independent output root.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.train_rmot_candidate_scorer import (  # noqa: E402
    auc, average_precision, load_bank, load_metadata, make_refs, scalar_stats,
)


VARIANTS = {
    "old_feature": ("clip", "geometry", "objectness"),
    "crop": ("crop_tight", "geometry", "objectness"),
    "crop_context": ("crop_tight", "crop_context_1p5", "geometry", "objectness"),
    "crop_context_geometry": ("crop_tight", "crop_context_1p5", "geometry_v2", "objectness"),
    "crop_context_geometry_motion": ("crop_tight", "crop_context_1p5", "geometry_v2", "motion_v2", "objectness"),
    "all_v2": ("clip", "history_clip", "crop_tight", "crop_context_1p5", "crop_local_context",
               "crop_full_context", "geometry_v2", "neighbor_v2", "motion_v2", "lifecycle_v2", "objectness"),
}


def fixed_refs(refs: list[dict], count: int, seed: int) -> list[dict]:
    if len(refs) <= count:
        return list(refs)
    return [refs[i] for i in sorted(random.Random(seed).sample(range(len(refs)), count))]


def feature_array(ref: dict, bank: dict, relative: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
    t = bank["tensors"]; absolute = ref["begin"] + np.asarray(relative, np.int64)
    pieces = [np.repeat(ref["spec"][None, :], len(absolute), axis=0).astype(np.float32)]
    for field in fields:
        if field == "objectness":
            value = t[field][absolute].float().numpy().reshape(-1, 1)
        else:
            value = t[field][absolute].float().numpy().reshape(len(absolute), -1)
        pieces.append(value.astype(np.float32))
    return np.concatenate(pieces, axis=1)


def selected_rows(ref: dict, bank: dict, rng: random.Random, training: bool = True) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(ref["positive"])
    negative = np.flatnonzero(~ref["positive"])
    if training and len(positive) > 8:
        positive = np.asarray(rng.sample(positive.tolist(), 8), np.int64)
    objectness = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
    pre = negative[np.argsort(-objectness[negative], kind="stable")[:min(negative.size, 96)]]
    rest = np.setdiff1d(negative, pre, assume_unique=False)
    if training and len(rest) > 16:
        rest = np.asarray(rng.sample(rest.tolist(), 16), np.int64)
    rows = np.concatenate((positive, pre, rest)).astype(np.int64)
    return rows, ref["positive"][rows].astype(np.float32)


def train_probe(model: nn.Module, refs: list[dict], bank: dict, fields: tuple[str, ...],
                device: torch.device, seed: int, steps: int, batch_frames: int) -> dict:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses, gradients = [], []
    model.train()
    for _step in range(steps):
        selected = [refs[rng.randrange(len(refs))] for _ in range(batch_frames)]
        xs, ys, frame_sizes = [], [], []
        for ref in selected:
            rows, labels = selected_rows(ref, bank, rng, training=True)
            xs.append(feature_array(ref, bank, rows, fields)); ys.append(labels); frame_sizes.append(len(labels))
        x = torch.as_tensor(np.concatenate(xs), device=device)
        y = torch.as_tensor(np.concatenate(ys), device=device)
        logits = model(x).squeeze(1)
        # Equal weight per frame prevents a large candidate frame dominating
        # the probe; this is still only a linear frozen-feature upper bound.
        offset, terms = 0, []
        for size in frame_sizes:
            terms.append(nn.functional.binary_cross_entropy_with_logits(
                logits[offset:offset + size], y[offset:offset + size]))
            offset += size
        loss = torch.stack(terms).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step()
        losses.append(float(loss.detach().cpu())); gradients.append(grad)
    return {"steps": steps, "loss": scalar_stats(losses), "gradient_norm": scalar_stats(gradients)}


def eval_probe(model: nn.Module, refs: list[dict], bank: dict, fields: tuple[str, ...],
               device: torch.device, batch_refs: int = 32) -> dict:
    scores_all, labels_all = [], []
    margins, online_margins = [], []; violation = []; online_violation = []
    top1 = top5 = positive_frames = 0; null_max = []
    source_selected = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
    zero_pred = zero_pos = 0
    model.eval()
    for chunk_start in range(0, len(refs), batch_refs):
        chunk = refs[chunk_start:chunk_start + batch_refs]
        xs, boundaries = [], []
        for ref in chunk:
            rows = np.arange(ref["end"] - ref["begin"], dtype=np.int64)
            xs.append(feature_array(ref, bank, rows, fields)); boundaries.append((ref, len(rows)))
        flat = np.concatenate(xs, axis=0)
        with torch.inference_mode():
            flat_score = model(torch.as_tensor(flat, device=device)).squeeze(1).cpu().numpy()
        offset = 0
        for ref, size in boundaries:
            score = flat_score[offset:offset + size]; label = ref["positive"].astype(bool)
            offset += size
            scores_all.append(score); labels_all.append(label)
            positive = np.flatnonzero(label); negative = np.flatnonzero(~label)
            zero_pred += int(np.sum(score >= 0.0)); zero_pos += int(np.sum((score >= 0.0) & label))
            if len(positive) and len(negative):
                positive_frames += 1
                order = np.argsort(-score, kind="stable")
                top1 += int(label[order[:1]].any()); top5 += int(label[order[:5]].any())
                margin = float(score[positive].min() - score[negative].max())
                margins.append(margin); violation.append(margin < 0)
                objectness = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
                pre = negative[np.argsort(-objectness[negative], kind="stable")[:min(96, len(negative))]]
                hard = pre[np.argsort(-score[pre], kind="stable")[:min(24, len(pre))]]
                if len(hard):
                    online_margin = float(score[positive].min() - score[hard].max())
                    online_margins.append(online_margin); online_violation.append(online_margin < 0)
                for source_id in (0, 1):
                    pool = bank["tensors"]["pool_id"][ref["begin"]:ref["end"]].numpy() == source_id
                    source_rows = np.flatnonzero(pool)
                    if len(source_rows):
                        source_order = source_rows[np.argsort(-score[source_rows], kind="stable")]
                        source_selected[source_id][0] += 1
                        source_selected[source_id][1] += int(label[source_order[:1]].any())
                        source_selected[source_id][2] += min(5, len(source_order))
                        source_selected[source_id][3] += int(label[source_order[:5]].sum())
            elif len(positive) == 0:
                null_max.append(float(score.max()) if len(score) else 0.0)
    score = np.concatenate(scores_all); label = np.concatenate(labels_all)
    return {
        "candidate_count": int(len(label)), "positive_count": int(label.sum()),
        "roc_auc": auc(score, label), "pr_auc": average_precision(score, label),
        "positive_score": scalar_stats(score[label]), "negative_score": scalar_stats(score[~label]),
        "positive_model_hard_margin": scalar_stats(margins),
        "model_hard_violation_rate": float(np.mean(violation)) if violation else None,
        "positive_online_hard_margin": scalar_stats(online_margins),
        "online_hard_violation_rate": float(np.mean(online_violation)) if online_violation else None,
        "positive_frame_count": positive_frames,
        "top1_frame_recall": float(top1 / max(1, positive_frames)),
        "top5_frame_recall": float(top5 / max(1, positive_frames)),
        "source_internal_precision": {
            "main": {"top1": source_selected[0][1] / max(1, source_selected[0][0]),
                      "top5": source_selected[0][3] / max(1, source_selected[0][2]),
                      "selected_frames": source_selected[0][0]},
            "reserve": {"top1": source_selected[1][1] / max(1, source_selected[1][0]),
                        "top5": source_selected[1][3] / max(1, source_selected[1][2]),
                        "selected_frames": source_selected[1][0]}},
        "null_highest_candidate_score": scalar_stats(null_max),
        "zero_threshold": {"predictions": zero_pred, "positive": zero_pos,
                            "predictions_per_positive": zero_pred / max(1, int(label.sum()))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--old-bank-root", default="outputs/l19/dual_banks_features")
    ap.add_argument("--v2-bank-root", default="outputs/l22/candidate_bank_v2")
    ap.add_argument("--out-root", default="outputs/l22/eval/bank_v2_linear_probe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--frames-per-split", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    manifest = Path(args.manifest); old_root = Path(args.old_bank_root); v2_root = Path(args.v2_bank_root); out_root = Path(args.out_root)
    if not manifest.is_absolute(): manifest = ROOT / manifest
    if not old_root.is_absolute(): old_root = ROOT / old_root
    if not v2_root.is_absolute(): v2_root = ROOT / v2_root
    if not out_root.is_absolute(): out_root = ROOT / out_root
    if out_root.exists(): raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    data = json.loads(manifest.read_text()); rows = sorted(data["queries"], key=lambda r: int(r["query_index"]))
    metadata = load_metadata(); videos = sorted({str(r["video"]) for r in rows})
    old = {v: load_bank(old_root / "kitti" / f"{v}.pt") for v in videos}
    v2 = {v: load_bank(v2_root / "kitti" / f"{v}.pt") for v in videos}
    old_refs = make_refs(rows, metadata, old); v2_refs = make_refs(rows, metadata, v2)
    # Structural identity is asserted before any probe is fit.
    for a, b in zip(old_refs, v2_refs):
        for key in ("query_index", "video", "frame_id", "begin", "end"):
            if a[key] != b[key]: raise AssertionError(f"reference mismatch {key}: {a[key]} vs {b[key]}")
        if not np.array_equal(a["positive"], b["positive"]): raise AssertionError("label mismatch")
    train_old = fixed_refs([r for r in old_refs if r["split"] == "calibration"], args.frames_per_split, args.seed)
    val_old = fixed_refs([r for r in old_refs if r["split"] == "screening"], args.frames_per_split, args.seed + 1)
    train_v2 = fixed_refs([r for r in v2_refs if r["split"] == "calibration"], args.frames_per_split, args.seed)
    val_v2 = fixed_refs([r for r in v2_refs if r["split"] == "screening"], args.frames_per_split, args.seed + 1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    result = {"format": "locatemot-l22-bank-linear-probe-v1", "manifest": str(manifest),
              "manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
              "query_count": len(rows), "calibration_queries": 64, "screening_queries": 96,
              "train_frame_units": len(train_old), "val_frame_units": len(val_old),
              "seed": args.seed, "steps": args.steps, "device": str(device),
              "old_bank_root": str(old_root), "v2_bank_root": str(v2_root), "variants": {}}
    for name, fields in VARIANTS.items():
        print(f"[L22 linear probe] {name}", flush=True)
        probe_bank = v2 if name != "old_feature" else old
        train_refs = train_v2 if name != "old_feature" else train_old
        val_refs = val_v2 if name != "old_feature" else val_old
        sample_x = feature_array(train_refs[0], probe_bank[train_refs[0]["video"]], np.arange(train_refs[0]["end"] - train_refs[0]["begin"]), fields)
        model = nn.Linear(sample_x.shape[1], 1, device=device)
        started = time.time()
        train_report = train_probe(model, train_refs, probe_bank[train_refs[0]["video"]], fields, device, args.seed, args.steps, 16) if False else None
        # Refs span two videos; train one frame at a time against its matching bank.
        # The wrapper below keeps exactly the same optimizer while selecting the
        # appropriate bank for each ref.
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); rng = random.Random(args.seed)
        losses, grads = [], []; model.train()
        for _ in range(args.steps):
            selected = [train_refs[rng.randrange(len(train_refs))] for _ in range(16)]
            xs, ys, sizes = [], [], []
            for ref in selected:
                rows_sel, labels_sel = selected_rows(ref, probe_bank[ref["video"]], rng, True)
                xs.append(feature_array(ref, probe_bank[ref["video"]], rows_sel, fields)); ys.append(labels_sel); sizes.append(len(labels_sel))
            x = torch.as_tensor(np.concatenate(xs), device=device); y = torch.as_tensor(np.concatenate(ys), device=device)
            logits = model(x).squeeze(1); offset = 0; terms = []
            for size in sizes:
                terms.append(nn.functional.binary_cross_entropy_with_logits(logits[offset:offset + size], y[offset:offset + size])); offset += size
            loss = torch.stack(terms).mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step(); losses.append(float(loss.detach().cpu())); grads.append(grad)
        train_report = {"steps": args.steps, "loss": scalar_stats(losses), "gradient_norm": scalar_stats(grads)}
        # Evaluate video-by-video because a bank is keyed by video.
        val_parts = []
        train_parts = []
        for split_refs, destination in ((train_refs, train_parts), (val_refs, val_parts)):
            for video in videos:
                subset = [r for r in split_refs if r["video"] == video]
                if subset: destination.append(eval_probe(model, subset, probe_bank[video], fields, device))
        def merge(parts):
            keys = ("candidate_count", "positive_count", "positive_frame_count")
            out = {k: int(sum(p[k] for p in parts)) for k in keys}
            for metric in ("roc_auc", "pr_auc", "top1_frame_recall", "top5_frame_recall", "model_hard_violation_rate", "online_hard_violation_rate"):
                vals = [p[metric] for p in parts if p[metric] is not None]; out[metric] = float(np.mean(vals)) if vals else None
            for metric in ("positive_score", "negative_score", "positive_model_hard_margin", "positive_online_hard_margin", "null_highest_candidate_score"):
                vals = [p[metric] for p in parts]; out[metric] = {"count": sum(v.get("count", 0) for v in vals), "mean": float(np.average([v.get("mean", 0.) for v in vals], weights=[max(1,v.get("count",0)) for v in vals])) if vals else None}
            out["source_internal_precision"] = {name: {k: float(np.mean([p["source_internal_precision"][name][k] for p in parts])) for k in ("top1", "top5")} for name in ("main", "reserve")}
            out["zero_threshold"] = {k: int(sum(p["zero_threshold"][k] for p in parts)) for k in ("predictions", "positive")}; out["zero_threshold"]["predictions_per_positive"] = out["zero_threshold"]["predictions"] / max(1, out["positive_count"])
            return out
        result["variants"][name] = {"fields": list(fields), "dimension": int(sample_x.shape[1]), "train": train_report, "train_metrics": merge(train_parts), "validation_metrics": merge(val_parts), "elapsed_sec": time.time() - started}
        del model
    (out_root / "linear_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_root / "README.md").write_text("# Stage L22 bank v2 linear probe\n\nFrozen old-bank versus candidate-bank-v2 upper-limit comparison. No tracker or TrackEval was run.\n")
    print(json.dumps({"output": str(out_root / "linear_probe.json"), "variants": list(result["variants"])}, indent=2))


if __name__ == "__main__":
    main()
