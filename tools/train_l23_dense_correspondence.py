"""Train/evaluate one independent L23 dense-correspondence ablation.

The training side is calibration-only and uses online hard mining selected by
the current scorer after an objectness top-96 efficiency prefilter. The same
rule is used for screening metrics. No grouping, source acceptance, NULL
subtraction, GRU, tracker, or old checkpoint is touched.
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
from locatemot.models.rmot_dense_correspondence_scorer import DenseQueryCorrespondenceScorer  # noqa: E402
from tools.train_rmot_candidate_scorer import average_precision, auc, load_bank, load_metadata, make_refs, scalar_stats  # noqa: E402

ROI_ONLY = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_refs(refs: list[dict], count: int, seed: int) -> list[dict]:
    if len(refs) <= count: return list(refs)
    return [refs[i] for i in sorted(random.Random(seed).sample(range(len(refs)), count))]


def arrays_for(ref: dict, bank: dict, rows: np.ndarray) -> dict[str, np.ndarray]:
    t = bank["tensors"]; absolute = ref["begin"] + np.asarray(rows, np.int64)
    query = np.repeat(ref["spec"][None, :], len(rows), axis=0).astype(np.float32)
    dense_roi = t["dense_roi"][absolute].float().numpy()
    dense_points = dense_roi[:, None, :] if ROI_ONLY else t["dense_points"][absolute].float().numpy()
    return {"query": query, "dense_points": dense_points,
            "dense_roi": dense_roi,
            "geometry": t["geometry_v2"][absolute].float().numpy(),
            "objectness": t["objectness"][absolute].float().numpy().reshape(-1, 1),
            "dense_context_1p5": t["dense_context_1p5"][absolute].float().numpy(),
            "dense_context_3": t["dense_context_3"][absolute].float().numpy(),
            "dense_prev_roi": t["dense_prev_roi"][absolute].float().numpy(),
            "motion": t["motion_v2"][absolute].float().numpy()}


def tensors(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


def model_score(model: nn.Module, value: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(query=value["query"], dense_points=value["dense_points"], dense_roi=value["dense_roi"],
                 geometry=value["geometry"], objectness=value["objectness"],
                 dense_context_1p5=value["dense_context_1p5"], dense_context_3=value["dense_context_3"],
                 dense_prev_roi=value["dense_prev_roi"], motion=value["motion"])


def mine_hard(ref: dict, bank: dict, model: nn.Module, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = ref["end"] - ref["begin"]; rows = np.arange(size, dtype=np.int64)
    labels = ref["positive"].astype(bool); negative = np.flatnonzero(~labels)
    objectness = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
    pre = negative[np.argsort(-objectness[negative], kind="stable")[:min(96, len(negative))]]
    if len(pre):
        with torch.inference_mode():
            scores = model_score(model, tensors(arrays_for(ref, bank, pre), device)).detach().cpu().numpy()
        hard = pre[np.argsort(-scores, kind="stable")[:min(24, len(pre))]]
    else:
        hard = np.zeros(0, dtype=np.int64)
    easy = np.setdiff1d(negative, pre, assume_unique=False)
    if len(easy) > 16:
        easy = easy[:16]
    return np.flatnonzero(labels), hard, easy


def frame_loss(model: nn.Module, ref: dict, bank: dict, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    size = ref["end"] - ref["begin"]; rows = np.arange(size, dtype=np.int64); labels = ref["positive"].astype(bool)
    positive, hard, easy = mine_hard(ref, bank, model, device)
    full = model_score(model, tensors(arrays_for(ref, bank, rows), device))
    zero = full.new_zeros(())
    if len(positive):
        positive_logits = full[torch.as_tensor(positive, device=device)]
        positive_bce = nn.functional.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
        if len(hard):
            hard_logits = full[torch.as_tensor(hard, device=device)]
            hard_bce = nn.functional.binary_cross_entropy_with_logits(hard_logits, torch.zeros_like(hard_logits))
            pairwise = nn.functional.softplus(1.0 - positive_logits[:, None] + hard_logits[None, :]).mean()
            violation = nn.functional.softplus(hard_logits.max() - positive_logits.max())
        else:
            hard_bce = pairwise = violation = zero
        if len(easy):
            easy_logits = full[torch.as_tensor(easy, device=device)]
            easy_bce = nn.functional.binary_cross_entropy_with_logits(easy_logits, torch.zeros_like(easy_logits))
        else:
            easy_bce = zero
        listwise = -(torch.logsumexp(positive_logits, dim=0) - torch.logsumexp(full, dim=0))
    else:
        positive_bce = hard_bce = pairwise = listwise = violation = zero
        easy_rows = rows[:min(32, len(rows))]
        easy_logits = full[torch.as_tensor(easy_rows, device=device)]
        easy_bce = nn.functional.softplus(easy_logits.max()) if len(easy_logits) else zero
    total = positive_bce + hard_bce + 0.1 * easy_bce + pairwise + 0.5 * listwise + 0.2 * violation
    return total, {"total": float(total.detach().cpu()), "positive_bce": float(positive_bce.detach().cpu()),
                   "hard_negative_bce": float(hard_bce.detach().cpu()), "easy_negative_bce": float(easy_bce.detach().cpu()),
                   "pairwise_margin": float(pairwise.detach().cpu()), "listwise": float(listwise.detach().cpu()),
                   "hard_violation_penalty": float(violation.detach().cpu()),
                   "positive_count": float(len(positive)), "hard_negative_count": float(len(hard)),
                   "easy_negative_count": float(len(easy)), "null_frame": float(not len(positive))}


def train(model: nn.Module, refs: list[dict], banks: dict[str, dict], device: torch.device,
          steps: int, seed: int, batch_frames: int) -> dict:
    rng = random.Random(seed); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = {key: [] for key in ("total", "positive_bce", "hard_negative_bce", "easy_negative_bce", "pairwise_margin", "listwise", "hard_violation_penalty", "positive_count", "hard_negative_count", "easy_negative_count", "null_frame")}
    gradients = []; started = time.time(); model.train()
    for _ in range(steps):
        selected = [refs[rng.randrange(len(refs))] for _ in range(batch_frames)]
        losses, reports = [], []
        for ref in selected:
            loss, report = frame_loss(model, ref, banks[ref["video"]], device); losses.append(loss); reports.append(report)
        loss = torch.stack(losses).mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
        gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))); optimizer.step()
        for key in history: history[key].append(float(np.mean([x[key] for x in reports])))
    return {"steps": steps, "elapsed_sec": time.time() - started,
            "loss": {key: scalar_stats(value) for key, value in history.items() if key not in ("positive_count", "hard_negative_count", "easy_negative_count", "null_frame")},
            "bucket_counts": {key: float(np.sum(value)) for key, value in history.items() if key in ("positive_count", "hard_negative_count", "easy_negative_count", "null_frame")},
            "gradient_norm": scalar_stats(gradients)}


def evaluate(model: nn.Module, refs: list[dict], banks: dict[str, dict], device: torch.device) -> dict:
    model.eval(); all_scores, all_labels = [], []; margins = []; violations = []; positive_values = []; hard_values = []; easy_values = []
    top1 = top5 = positive_frames = null_frames = 0; zero_predictions = zero_positive = 0; source = {0: [0, 0, 0], 1: [0, 0, 0]}
    for ref in refs:
        bank = banks[ref["video"]]; size = ref["end"] - ref["begin"]; rows = np.arange(size, dtype=np.int64); labels = ref["positive"].astype(bool)
        with torch.inference_mode(): scores = model_score(model, tensors(arrays_for(ref, bank, rows), device)).cpu().numpy()
        all_scores.append(scores); all_labels.append(labels); zero_predictions += int((scores >= 0).sum()); zero_positive += int(((scores >= 0) & labels).sum())
        positive = np.flatnonzero(labels); negative = np.flatnonzero(~labels)
        if not len(positive): null_frames += 1; continue
        positive_frames += 1; order = np.argsort(-scores, kind="stable"); top1 += int(labels[order[:1]].any()); top5 += int(labels[order[:5]].any()); positive_values.extend(scores[positive].tolist())
        objectness = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
        pre = negative[np.argsort(-objectness[negative], kind="stable")[:min(96, len(negative))]]
        hard = pre[np.argsort(-scores[pre], kind="stable")[:min(24, len(pre))]]
        easy = np.setdiff1d(negative, pre, assume_unique=False)[:16]
        if len(hard):
            margins.append(float(scores[positive].min() - scores[hard].max())); violations.append(margins[-1] < 0); hard_values.extend(scores[hard].tolist())
        easy_values.extend(scores[easy].tolist())
        for sid in (0, 1):
            pool = bank["tensors"]["pool_id"][ref["begin"]:ref["end"]].numpy() == sid; pool_rows = np.flatnonzero(pool)
            if len(pool_rows):
                source[sid][0] += 1; source[sid][1] += int(labels[pool_rows[np.argmax(scores[pool_rows])]]); source[sid][2] += int(labels[pool_rows[np.argsort(-scores[pool_rows])[:5]]].sum())
    scores = np.concatenate(all_scores); labels = np.concatenate(all_labels)
    return {"candidate_count": int(len(labels)), "positive_count": int(labels.sum()), "positive_frame_count": positive_frames,
            "null_frame_count": null_frames, "roc_auc": auc(scores, labels), "pr_auc": average_precision(scores, labels),
            "top1_frame_recall": top1 / max(1, positive_frames), "top5_frame_recall": top5 / max(1, positive_frames),
            "positive_score": scalar_stats(positive_values), "online_hard_score": scalar_stats(hard_values), "easy_negative_score": scalar_stats(easy_values),
            "positive_online_hard_margin": scalar_stats(margins), "hard_negative_violation_rate": float(np.mean(violations)) if violations else None,
            "source_internal_precision": {"main": {"top1": source[0][1] / max(1, source[0][0]), "top5": source[0][2] / max(1, source[0][0] * 5), "frames": source[0][0]},
                                           "reserve": {"top1": source[1][1] / max(1, source[1][0]), "top5": source[1][2] / max(1, source[1][0] * 5), "frames": source[1][0]}},
            "null_highest_candidate_score": scalar_stats([float(np.max(x)) for x, y in zip(all_scores, all_labels) if not y.any()]),
            "zero_threshold": {"predictions": zero_predictions, "positive": zero_positive, "predictions_per_positive": zero_predictions / max(1, int(labels.sum()))}}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json"); ap.add_argument("--v3-root", default="outputs/l23/candidate_bank_v3"); ap.add_argument("--out-root", required=True); ap.add_argument("--stage", default="D0", choices=("D0", "D1", "D2", "D3", "D4")); ap.add_argument("--steps", type=int, default=50); ap.add_argument("--seed", type=int, default=17); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--frames-per-split", type=int, default=6000); ap.add_argument("--batch-frames", type=int, default=4); ap.add_argument("--roi-only", action="store_true")
    args = ap.parse_args()
    global ROI_ONLY
    ROI_ONLY = bool(args.roi_only)
    def p(x: str) -> Path:
        x = Path(x); return x if x.is_absolute() else ROOT / x
    manifest, v3_root, out_root = map(p, (args.manifest, args.v3_root, args.out_root))
    if out_root.exists(): raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    try:
        queries = sorted(json.loads(manifest.read_text())["queries"], key=lambda x: int(x["query_index"])); metadata = load_metadata(); videos = sorted({str(q["video"]) for q in queries})
        banks = {v: load_bank(v3_root / "kitti" / f"{v}.pt") for v in videos}; refs = make_refs(queries, metadata, banks)
        calibration = fixed_refs([r for r in refs if r["split"] == "calibration"], args.frames_per_split, args.seed); screening = fixed_refs([r for r in refs if r["split"] == "screening"], args.frames_per_split, args.seed + 1)
        device = torch.device(args.device); model = DenseQueryCorrespondenceScorer(stage=args.stage).to(device)
        report = {"format": "locatemot-l23-dense-correspondence-train-v1", "stage": args.stage, "manifest": str(manifest), "manifest_sha256": sha256(manifest), "v3_root": str(v3_root), "device": str(device), "seed": args.seed, "steps": args.steps, "calibration_queries": 64, "screening_queries": 96, "calibration_frame_units": len(calibration), "screening_frame_units": len(screening), "loss_weights": {"positive_bce": 1.0, "online_hard_bce": 1.0, "easy_bce": 0.1, "pairwise_margin": 1.0, "listwise": 0.5, "hard_violation": 0.2}, "hard_rule": {"objectness_prefilter": 96, "current_scorer_topk": 24, "training_validation_same_rule": True}, "excluded": ["grouping", "membership", "source_acceptance", "scalar_null_subtraction", "temporal_gru", "tracker"], "motion_language_decomposition": "not claimed; current query is static-only and motion_v2 is visual track motion"}
        report["train"] = train(model, calibration, banks, device, args.steps, args.seed, args.batch_frames)
        report["calibration_metrics"] = evaluate(model, calibration, banks, device); report["screening_metrics"] = evaluate(model, screening, banks, device)
        ckpt = out_root / f"checkpoint_{args.stage.lower()}_step{args.steps}.pt"; torch.save({"model": model.state_dict(), "config": {"stage": args.stage, "class": "DenseQueryCorrespondenceScorer"}, "manifest_sha256": report["manifest_sha256"], "v3_root": str(v3_root), "step": args.steps}, ckpt)
        report["checkpoint"] = str(ckpt); report["checkpoint_reload"] = False
        reload_model = DenseQueryCorrespondenceScorer(stage=args.stage).to(device); reload_model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"]); reload_model.eval(); report["checkpoint_reload"] = True
        (out_root / f"metrics_{args.stage.lower()}_step{args.steps}.json").write_text(json.dumps(report, indent=2) + "\n")
        (out_root / "README.md").write_text(f"# L23 {args.stage} dense correspondence\n\nIndependent {args.steps}-step run on calibration only; screening is reporting-only.\n")
        print(json.dumps({"out_root": str(out_root), "stage": args.stage, "checkpoint": str(ckpt), "screening": {k: report["screening_metrics"].get(k) for k in ("roc_auc", "pr_auc", "top1_frame_recall", "positive_online_hard_margin", "hard_negative_violation_rate", "zero_threshold")}}, indent=2))
    except Exception as exc:
        (out_root / "INCOMPLETE.md").write_text(f"# INCOMPLETE\n\nL23 {args.stage} run stopped at first error: `{type(exc).__name__}: {exc}`\n"); raise


if __name__ == "__main__": main()
