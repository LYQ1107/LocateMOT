"""C0-C5 frozen dense-feature upper-limit probe for Stage L23.

All variants share the fixed L19 manifest, calibration/screening query-frame
units, optimizer, seed and evaluator.  This is an offline probe only: no
tracker, TrackEval, checkpoint selection from screening labels, or GT-driven
feature construction is performed.
"""
from __future__ import annotations

import argparse
import hashlib
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
    average_precision, auc, load_bank, load_metadata, make_refs, scalar_stats,
)

VARIANTS = {
    "C0_old_pooled": ("clip", "geometry", "objectness"),
    "C1_v2_crop_context": ("crop_tight", "crop_context_1p5", "geometry_v2", "motion_v2", "objectness"),
    "C2_v3_roi": ("dense_roi", "geometry_v2", "objectness"),
    "C3_v3_multipoint": ("dense_points", "geometry_v2", "objectness"),
    "C4_v3_roi_context_motion": ("dense_roi", "dense_context_1p5", "dense_context_3", "geometry_v2", "motion_v2", "objectness"),
    "C5_v3_all_dense": ("dense_roi", "dense_points", "dense_context_1p5", "dense_context_3", "dense_prev_roi",
                         "geometry_v2", "neighbor_v2", "motion_v2", "lifecycle_v2", "objectness"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_refs(refs: list[dict], count: int, seed: int) -> list[dict]:
    if len(refs) <= count:
        return list(refs)
    return [refs[i] for i in sorted(random.Random(seed).sample(range(len(refs)), count))]


def feature_array(ref: dict, bank: dict, relative: np.ndarray, fields: tuple[str, ...]) -> np.ndarray:
    t = bank["tensors"]
    absolute = ref["begin"] + np.asarray(relative, np.int64)
    pieces = [np.repeat(ref["spec"][None, :], len(absolute), axis=0).astype(np.float32)]
    for field in fields:
        value = t[field][absolute].float().numpy()
        if field == "objectness":
            value = value.reshape(-1, 1)
        else:
            value = value.reshape(len(absolute), -1)
        pieces.append(value.astype(np.float32))
    return np.concatenate(pieces, axis=1)


def selected_rows(ref: dict, bank: dict, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(ref["positive"])
    negative = np.flatnonzero(~ref["positive"])
    if len(positive) > 8:
        positive = np.asarray(rng.sample(positive.tolist(), 8), np.int64)
    objectness = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
    hard = negative[np.argsort(-objectness[negative], kind="stable")[:min(96, len(negative))]]
    rest = np.setdiff1d(negative, hard, assume_unique=False)
    if len(rest) > 16:
        rest = np.asarray(rng.sample(rest.tolist(), 16), np.int64)
    rows = np.concatenate((positive, hard, rest)).astype(np.int64)
    return rows, ref["positive"][rows].astype(np.float32)


def train_probe(model: nn.Module, refs: list[dict], banks: dict[str, dict], fields: tuple[str, ...],
                device: torch.device, seed: int, steps: int, batch_frames: int) -> dict:
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses, gradients = [], []
    model.train()
    for _ in range(steps):
        selected = [refs[rng.randrange(len(refs))] for _ in range(batch_frames)]
        xs, ys, sizes = [], [], []
        for ref in selected:
            rows, labels = selected_rows(ref, banks[ref["video"]], rng)
            xs.append(feature_array(ref, banks[ref["video"]], rows, fields)); ys.append(labels); sizes.append(len(labels))
        x = torch.as_tensor(np.concatenate(xs), device=device); y = torch.as_tensor(np.concatenate(ys), device=device)
        logits = model(x).squeeze(1); offset = 0; terms = []
        for size in sizes:
            terms.append(nn.functional.binary_cross_entropy_with_logits(logits[offset:offset + size], y[offset:offset + size]))
            offset += size
        loss = torch.stack(terms).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step()
        losses.append(float(loss.detach().cpu())); gradients.append(grad)
    return {"steps": steps, "loss": scalar_stats(losses), "gradient_norm": scalar_stats(gradients)}


def evaluate(model: nn.Module, refs: list[dict], banks: dict[str, dict], fields: tuple[str, ...],
             device: torch.device, batch_refs: int = 24) -> dict:
    all_scores, all_labels = [], []
    margins, violations, top1_margins, hard_scores = [], [], [], []
    top1 = top5 = positive_frames = multi_frames = multi_top1 = multi_top5 = 0
    null_max, source = [], {0: [0, 0, 0], 1: [0, 0, 0]}
    zero_predictions = zero_positive = 0
    model.eval()
    for start in range(0, len(refs), batch_refs):
        chunk = refs[start:start + batch_refs]; arrays = []; bounds = []
        for ref in chunk:
            size = ref["end"] - ref["begin"]
            arrays.append(feature_array(ref, banks[ref["video"]], np.arange(size, dtype=np.int64), fields))
            bounds.append((ref, size))
        flat = np.concatenate(arrays)
        with torch.inference_mode():
            scores_flat = model(torch.as_tensor(flat, device=device)).squeeze(1).cpu().numpy()
        offset = 0
        for ref, size in bounds:
            scores = scores_flat[offset:offset + size]; offset += size
            labels = ref["positive"].astype(bool); positive = np.flatnonzero(labels); negative = np.flatnonzero(~labels)
            all_scores.append(scores); all_labels.append(labels)
            zero_predictions += int((scores >= 0).sum()); zero_positive += int(((scores >= 0) & labels).sum())
            if not len(positive):
                null_max.append(float(scores.max()) if len(scores) else 0.0); continue
            positive_frames += 1
            order = np.argsort(-scores, kind="stable")
            top1 += int(labels[order[:1]].any()); top5 += int(labels[order[:5]].any())
            if len(positive) > 1:
                multi_frames += 1; multi_top1 += int(labels[order[:1]].any()); multi_top5 += int(labels[order[:5]].any())
            objectness = banks[ref["video"]]["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
            pre = negative[np.argsort(-objectness[negative], kind="stable")[:min(96, len(negative))]]
            hard = pre[np.argsort(-scores[pre], kind="stable")[:min(24, len(pre))]]
            if len(hard):
                margin = float(scores[positive].min() - scores[hard].max())
                margins.append(margin); violations.append(margin < 0); hard_scores.extend(scores[hard].tolist())
                top1_margins.append(float(scores[positive].max() - scores[hard].max()))
            for sid in (0, 1):
                pool = banks[ref["video"]]["tensors"]["pool_id"][ref["begin"]:ref["end"]].numpy() == sid
                rows = np.flatnonzero(pool)
                if not len(rows): continue
                source[sid][0] += 1; source[sid][1] += int(labels[rows[np.argmax(scores[rows])]])
                source[sid][2] += int(labels[rows[np.argsort(-scores[rows], kind="stable")[:5]]].sum())
    scores = np.concatenate(all_scores); labels = np.concatenate(all_labels)
    return {
        "candidate_count": int(len(labels)), "positive_count": int(labels.sum()),
        "positive_frame_count": positive_frames, "roc_auc": auc(scores, labels),
        "pr_auc": average_precision(scores, labels), "top1_frame_recall": top1 / max(1, positive_frames),
        "top5_frame_recall": top5 / max(1, positive_frames), "model_hard_margin": scalar_stats(margins),
        "model_hard_violation_rate": float(np.mean(violations)) if violations else None,
        "per_frame_top1_margin": scalar_stats(top1_margins), "model_hard_score": scalar_stats(hard_scores),
        "multi_positive_frame_count": multi_frames,
        "multi_positive_top1_recall": multi_top1 / max(1, multi_frames),
        "multi_positive_top5_recall": multi_top5 / max(1, multi_frames),
        "source_internal_precision": {
            "main": {"top1": source[0][1] / max(1, source[0][0]), "top5": source[0][2] / max(1, source[0][0] * 5), "frames": source[0][0]},
            "reserve": {"top1": source[1][1] / max(1, source[1][0]), "top5": source[1][2] / max(1, source[1][0] * 5), "frames": source[1][0]}},
        "null_highest_candidate_score": scalar_stats(null_max),
        "zero_threshold": {"predictions": zero_predictions, "positive": zero_positive,
                            "predictions_per_positive": zero_predictions / max(1, int(labels.sum()))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--old-bank-root", default="outputs/l19/dual_banks_features")
    ap.add_argument("--v3-root", default="outputs/l23/candidate_bank_v3")
    ap.add_argument("--out-root", default="outputs/l23/eval/dense_linear_probe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--frames-per-split", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    def p(value: str) -> Path:
        value = Path(value); return value if value.is_absolute() else ROOT / value
    manifest, old_root, v3_root, out_root = map(p, (args.manifest, args.old_bank_root, args.v3_root, args.out_root))
    if out_root.exists(): raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    queries = sorted(json.loads(manifest.read_text())["queries"], key=lambda x: int(x["query_index"]))
    metadata = load_metadata(); videos = sorted({str(q["video"]) for q in queries})
    old = {v: load_bank(old_root / "kitti" / f"{v}.pt") for v in videos}
    v3 = {v: load_bank(v3_root / "kitti" / f"{v}.pt") for v in videos}
    old_refs = make_refs(queries, metadata, old); v3_refs = make_refs(queries, metadata, v3)
    for a, b in zip(old_refs, v3_refs):
        for key in ("query_index", "video", "frame_id", "begin", "end"):
            if a[key] != b[key]: raise AssertionError(f"reference mismatch {key}")
        if not np.array_equal(a["positive"], b["positive"]): raise AssertionError("label mismatch")
    train = fixed_refs([r for r in v3_refs if r["split"] == "calibration"], args.frames_per_split, args.seed)
    screening = fixed_refs([r for r in v3_refs if r["split"] == "screening"], args.frames_per_split, args.seed + 1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    result = {"format": "locatemot-l23-dense-linear-probe-v1", "manifest": str(manifest),
              "manifest_sha256": sha256(manifest), "query_count": len(queries),
              "calibration_queries": 64, "screening_queries": 96,
              "train_frame_units": len(train), "screening_frame_units": len(screening),
              "steps": args.steps, "seed": args.seed, "device": str(device),
              "v3_root": str(v3_root), "old_bank_root": str(old_root),
              "screening_gt_used_for_selection": False, "variants": {}}
    for name, fields in VARIANTS.items():
        print(f"[L23 dense probe] {name}", flush=True)
        probe_banks = old if name == "C0_old_pooled" else v3
        refs = old_refs if name == "C0_old_pooled" else v3_refs
        train_refs = [r for r in refs if r["split"] == "calibration"]
        screening_refs = [r for r in refs if r["split"] == "screening"]
        sample = feature_array(train_refs[0], probe_banks[train_refs[0]["video"]], np.arange(train_refs[0]["end"] - train_refs[0]["begin"]), fields)
        model = nn.Linear(sample.shape[1], 1, device=device)
        started = time.time()
        train_report = train_probe(model, train, probe_banks, fields, device, args.seed, args.steps, 16)
        train_metrics = evaluate(model, train_refs, probe_banks, fields, device)
        screening_metrics = evaluate(model, screening, probe_banks, fields, device)
        result["variants"][name] = {"fields": list(fields), "dimension": int(sample.shape[1]),
                                    "train": train_report, "calibration_metrics": train_metrics,
                                    "screening_metrics": screening_metrics,
                                    "elapsed_sec": time.time() - started}
        del model
    (out_root / "dense_linear_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_root / "README.md").write_text("# Stage L23 dense-feature linear probe\n\nC0-C5 use the fixed L19 fast manifest and an identical light linear probe. Screening labels are reporting-only and were not used for model selection. No tracker or TrackEval was run.\n")
    print(json.dumps({"result": str(out_root / "dense_linear_probe.json"),
                      "variants": {k: {m: v["screening_metrics"].get(m) for m in ("roc_auc", "pr_auc", "top1_frame_recall", "model_hard_margin", "model_hard_violation_rate")} for k, v in result["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
