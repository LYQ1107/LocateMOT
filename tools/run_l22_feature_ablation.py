"""Bounded Stage L22 feature and hard/loss ablations."""
from __future__ import annotations

import hashlib
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
    "F1_crop": (("tight", "geometry", "objectness"), 96, 24, 12, 1.0, 0.5),
    "F2_context": (("tight", "context", "geometry", "objectness"), 96, 24, 12, 1.0, 0.5),
    "F3_motion": (("tight", "context", "geometry", "motion", "objectness"), 96, 24, 12, 1.0, 0.5),
    "F4_neighbor": (("tight", "context", "geometry", "motion", "neighbor", "objectness"), 96, 24, 12, 1.0, 0.5),
    "F5_hard_top48_top12": (("tight", "context", "geometry", "motion", "neighbor", "objectness"), 48, 12, 12, 1.0, 0.5),
    "F6_listwise_only": (("tight", "context", "geometry", "motion", "neighbor", "objectness"), 96, 24, 12, 0.0, 1.0),
}
FIELD_MAP = {"tight": "crop_tight", "context": "crop_context_1p5",
             "geometry": "geometry_v2", "motion": "motion_v2",
             "neighbor": "neighbor_v2", "objectness": "objectness"}


class AblationScorer(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 256),
                                 nn.GELU(), nn.Dropout(.1), nn.Linear(256, 256),
                                 nn.GELU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(torch.nan_to_num(x.float())).squeeze(1)


def vector(ref, bank, relative, fields):
    t = bank["tensors"]; absolute = ref["begin"] + np.asarray(relative, np.int64)
    parts = [np.repeat(ref["spec"][None, :], len(absolute), axis=0).astype(np.float32)]
    for field in fields:
        value = t[FIELD_MAP[field]][absolute].float().numpy().reshape(len(absolute), -1)
        parts.append(value.astype(np.float32))
    return np.concatenate(parts, axis=1)


def select_rows(ref, bank, rng, model, device, fields, prefilter, topk):
    pos = np.flatnonzero(ref["positive"]); neg = np.flatnonzero(~ref["positive"])
    if len(pos) > 8: pos = np.asarray(rng.sample(pos.tolist(), 8), np.int64)
    obj = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
    pre = neg[np.argsort(-obj[neg], kind="stable")[:min(prefilter, len(neg))]]
    if len(pre):
        with torch.no_grad(): hs = model(torch.as_tensor(vector(ref, bank, pre, fields), device=device)).cpu().numpy()
        hard = pre[np.argsort(-hs, kind="stable")[:min(topk, len(pre))]]
    else:
        hard = np.zeros(0, np.int64)
    rest = np.setdiff1d(neg, hard, assume_unique=False)
    easy = np.asarray(rng.sample(rest.tolist(), min(16, len(rest))), np.int64) if len(rest) else np.zeros(0, np.int64)
    rows = np.concatenate((pos, hard, easy)); mask = np.zeros(len(rows), bool)
    mask[len(pos):len(pos) + len(hard)] = True
    return rows, mask


def train(model, train_refs, banks, fields, prefilter, topk, pair_topk,
          pair_weight, listwise_weight, device, seed, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=2e-5)
    rng = random.Random(seed); loss_rows = []; model.train()
    for _ in range(steps):
        refs = [train_refs[rng.randrange(len(train_refs))] for _ in range(8)]
        chunks = [select_rows(r, banks[r["video"]], rng, model, device, fields, prefilter, topk) for r in refs]
        x = torch.as_tensor(np.concatenate([vector(r, banks[r["video"]], rows, fields) for r, (rows, _) in zip(refs, chunks)]), device=device)
        y = torch.as_tensor(np.concatenate([r["positive"][rows].astype(np.float32) for r, (rows, _) in zip(refs, chunks)]), device=device)
        hard = torch.as_tensor(np.concatenate([mask for _, mask in chunks]), device=device).bool()
        logits = model(x); offset = 0; groups = []
        for rows, _ in chunks: groups.append((offset, offset + len(rows))); offset += len(rows)
        pos_terms = []; hard_terms = []; easy_terms = []; pair_terms = []; list_terms = []
        for begin, end in groups:
            local, target = logits[begin:end], y[begin:end]; hm = hard[begin:end]
            pos = local[target > .5]; neg = local[target <= .5]
            hn = local[(target <= .5) & hm]; easy = local[(target <= .5) & ~hm]
            if len(pos): pos_terms.append(nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)))
            if len(hn): hard_terms.append(nn.functional.binary_cross_entropy_with_logits(hn, torch.zeros_like(hn)))
            if len(easy): easy_terms.append(nn.functional.binary_cross_entropy_with_logits(easy, torch.zeros_like(easy)))
            pair_neg = hn[torch.topk(hn, min(pair_topk, len(hn)), largest=True).indices] if len(hn) else neg
            if len(pos) and len(pair_neg):
                pair_terms.append(nn.functional.softplus(.5 - (pos[:, None] - pair_neg[None, :])).mean())
                list_terms.append(torch.logsumexp(torch.cat((pos, pair_neg)), 0) - torch.logsumexp(pos, 0))
        zero = logits.sum() * 0.; pb = torch.stack(pos_terms).mean() if pos_terms else zero
        hb = torch.stack(hard_terms).mean() if hard_terms else zero; eb = torch.stack(easy_terms).mean() if easy_terms else zero
        pair = torch.stack(pair_terms).mean() if pair_terms else zero; lst = torch.stack(list_terms).mean() if list_terms else zero
        total = pb + hb + .1 * eb + pair_weight * pair + listwise_weight * lst
        optimizer.zero_grad(set_to_none=True); total.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)); optimizer.step(); scheduler.step()
        loss_rows.append({"total": float(total.detach().cpu()), "positive_bce": float(pb.detach().cpu()),
                          "hard_negative_bce": float(hb.detach().cpu()), "easy_negative_bce": float(eb.detach().cpu()),
                          "pairwise_margin": float(pair.detach().cpu()), "listwise_ranking": float(lst.detach().cpu()),
                          "gradient_norm": grad, "sampled_candidates": int(len(y)),
                          "sampled_positive": int(y.sum().item()), "sampled_hard": int(hard.sum().item())})
    return {key: scalar_stats([row[key] for row in loss_rows]) for key in loss_rows[0]}, loss_rows


def evaluate(model, refs, bank, fields, prefilter, topk, device):
    scores = []; labels = []; margins = []; online = []; violations = []; online_violations = []
    null_max = []; top1 = top5 = frame_count = zero_pred = zero_pos = 0; source = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
    model.eval()
    for ref in refs:
        n = ref["end"] - ref["begin"]; value = vector(ref, bank, np.arange(n), fields)
        with torch.inference_mode(): out = model(torch.as_tensor(value, device=device)).cpu().numpy()
        y = ref["positive"].astype(bool); scores.append(out); labels.append(y); zero_pred += int((out >= 0).sum()); zero_pos += int(((out >= 0) & y).sum())
        pos = np.flatnonzero(y); neg = np.flatnonzero(~y)
        if len(pos) and len(neg):
            frame_count += 1; order = np.argsort(-out, kind="stable"); top1 += int(y[order[:1]].any()); top5 += int(y[order[:5]].any())
            margin = float(out[pos].min() - out[neg].max()); margins.append(margin); violations.append(margin < 0)
            obj = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
            pre = neg[np.argsort(-obj[neg], kind="stable")[:min(prefilter, len(neg))]]
            hard = pre[np.argsort(-out[pre], kind="stable")[:min(topk, len(pre))]]
            online_margin = float(out[pos].min() - out[hard].max()) if len(hard) else margin
            online.append(online_margin); online_violations.append(online_margin < 0)
            pool = bank["tensors"]["pool_id"][ref["begin"]:ref["end"]].numpy()
            for sid in (0, 1):
                rows = np.flatnonzero(pool == sid)
                if len(rows):
                    so = rows[np.argsort(-out[rows], kind="stable")]; source[sid][0] += 1; source[sid][1] += int(y[so[:1]].any()); source[sid][2] += min(5, len(so)); source[sid][3] += int(y[so[:5]].sum())
        elif ref["null"]:
            null_max.append(float(out.max()) if len(out) else 0.)
    scores = np.concatenate(scores); labels = np.concatenate(labels)
    return {"candidate_count": int(len(labels)), "positive_count": int(labels.sum()), "roc_auc": auc(scores, labels), "pr_auc": average_precision(scores, labels),
            "positive_model_hard_margin": scalar_stats(margins), "model_hard_violation_rate": float(np.mean(violations)) if violations else None,
            "positive_online_hard_margin": scalar_stats(online), "online_hard_violation_rate": float(np.mean(online_violations)) if online_violations else None,
            "positive_frame_count": frame_count, "top1_frame_recall": top1 / max(1, frame_count), "top5_frame_recall": top5 / max(1, frame_count),
            "source_internal_precision": {"main": {"top1": source[0][1] / max(1, source[0][0]), "top5": source[0][3] / max(1, source[0][2])}, "reserve": {"top1": source[1][1] / max(1, source[1][0]), "top5": source[1][3] / max(1, source[1][2])}},
            "null_highest_candidate_score": scalar_stats(null_max), "zero_threshold": {"predictions": zero_pred, "positive": zero_pos, "predictions_per_positive": zero_pred / max(1, int(labels.sum()))}}


def merge(parts):
    result = {key: int(sum(p[key] for p in parts)) for key in ("candidate_count", "positive_count", "positive_frame_count")}
    for key in ("roc_auc", "pr_auc", "top1_frame_recall", "top5_frame_recall", "model_hard_violation_rate", "online_hard_violation_rate"):
        result[key] = float(np.mean([p[key] for p in parts if p[key] is not None]))
    for key in ("positive_model_hard_margin", "positive_online_hard_margin", "null_highest_candidate_score"):
        result[key] = {"count": sum(p[key].get("count", 0) for p in parts), "mean": float(np.mean([p[key].get("mean", 0.) for p in parts]))}
    result["source_internal_precision"] = {s: {k: float(np.mean([p["source_internal_precision"][s][k] for p in parts])) for k in ("top1", "top5")} for s in ("main", "reserve")}
    result["zero_threshold"] = {k: int(sum(p["zero_threshold"][k] for p in parts)) for k in ("predictions", "positive")}; result["zero_threshold"]["predictions_per_positive"] = result["zero_threshold"]["predictions"] / max(1, result["positive_count"])
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json"); ap.add_argument("--bank-root", default="outputs/l22/candidate_bank_v2"); ap.add_argument("--out-root", default="outputs/l22/train/feature_ablations"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=17); args = ap.parse_args()
    manifest = Path(args.manifest); root = Path(args.bank_root); out = Path(args.out_root)
    if not manifest.is_absolute(): manifest = ROOT / manifest
    if not root.is_absolute(): root = ROOT / root
    if not out.is_absolute(): out = ROOT / out
    if out.exists(): raise FileExistsError(out)
    rows = sorted(json.loads(manifest.read_text())["queries"], key=lambda x: int(x["query_index"])); metadata = load_metadata(); videos = sorted({str(r["video"]) for r in rows}); banks = {v: load_bank(root / "kitti" / f"{v}.pt") for v in videos}; refs = make_refs(rows, metadata, banks); train_refs = [r for r in refs if r["split"] == "calibration"]; val_refs = [r for r in refs if r["split"] == "screening"]; device = torch.device(args.device)
    result = {"format": "locatemot-l22-feature-ablation-v1", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "bank_root": str(root), "seed": args.seed, "steps": args.steps, "official_eval_used": False, "trackeval_used": False, "variants": {}}
    for name, (fields, pre, top, pair, pair_weight, list_weight) in VARIANTS.items():
        print("[L22 ablation]", name, flush=True); sample = vector(train_refs[0], banks[train_refs[0]["video"]], np.arange(train_refs[0]["end"] - train_refs[0]["begin"]), fields); model = AblationScorer(sample.shape[1]).to(device); started = time.time(); loss, loss_rows = train(model, train_refs, banks, fields, pre, top, pair, pair_weight, list_weight, device, args.seed, args.steps)
        train_metrics = merge([evaluate(model, [r for r in train_refs if r["video"] == v], banks[v], fields, pre, top, device) for v in videos]); val_metrics = merge([evaluate(model, [r for r in val_refs if r["video"] == v], banks[v], fields, pre, top, device) for v in videos]); target = out / name; target.mkdir(parents=True, exist_ok=False); checkpoint = target / f"checkpoint_step{args.steps}.pt"; torch.save({"format": "locatemot-l22-ablation", "step": args.steps, "seed": args.seed, "model": model.state_dict(), "fields": list(fields), "hard_rule": {"prefilter": pre, "topk": top, "pairwise_topk": pair}}, checkpoint); metrics = {"fields": list(fields), "dimension": sample.shape[1], "hard_rule": {"prefilter": pre, "topk": top, "pairwise_topk": pair}, "loss": loss, "train": train_metrics, "validation": val_metrics, "elapsed_sec": time.time() - started, "checkpoint": str(checkpoint)}; (target / f"metrics_step{args.steps}.json").write_text(json.dumps(metrics, indent=2) + "\n"); result["variants"][name] = metrics; del model
    out.mkdir(parents=True, exist_ok=True); (out / "ablation_summary.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"output": str(out / "ablation_summary.json"), "variants": list(result["variants"])}, indent=2), flush=True)


if __name__ == "__main__": main()
