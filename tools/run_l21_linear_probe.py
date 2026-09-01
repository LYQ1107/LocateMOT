"""F5 frozen-feature linear probe for RMOT candidate separability."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.train_rmot_candidate_scorer import (  # noqa: E402
    HARD_PREFILTER, HARD_TOPK, auc, average_precision, load_bank,
    load_metadata, make_refs, scalar_stats,
)


def sampled_split(refs, banks, rng: random.Random, max_frames: int = 6000):
    selected_refs = refs if len(refs) <= max_frames else rng.sample(refs, max_frames)
    values, labels, frame_chunks = [], [], []
    for ref in selected_refs:
        bank, tensors = banks[ref["video"]], banks[ref["video"]]["tensors"]
        objectness = tensors["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
        positive = np.flatnonzero(ref["positive"])
        negative = np.flatnonzero(~ref["positive"])
        if len(positive) > 8:
            positive = np.asarray(rng.sample(positive.tolist(), 8), dtype=np.int64)
        prefilter = negative[np.argsort(-objectness[negative], kind="stable")[:min(len(negative), HARD_PREFILTER)]]
        hard = prefilter[:min(len(prefilter), HARD_TOPK)]
        remaining = np.setdiff1d(negative, hard, assume_unique=False)
        if len(remaining) > 16:
            remaining = np.asarray(rng.sample(remaining.tolist(), 16), dtype=np.int64)
        rows = np.concatenate([positive, hard, remaining]).astype(np.int64)
        current = tensors["clip"][ref["begin"] + rows].float().numpy()
        geometry = tensors["geometry"][ref["begin"] + rows].float().numpy()
        obj = tensors["objectness"][ref["begin"] + rows].float().numpy().reshape(-1, 1)
        query = np.repeat(ref["spec"][None, :], len(rows), axis=0)
        feature = np.concatenate([query, current, geometry, obj], axis=1)
        values.append(feature); labels.append(ref["positive"][rows])
        frame_chunks.append(np.empty(len(rows), np.float32))
    return np.concatenate(values).astype(np.float32), np.concatenate(labels).astype(bool), frame_chunks


def report(model, x, y, frame_chunks, device):
    with torch.inference_mode():
        score = model(torch.as_tensor(x, device=device)).squeeze(1).cpu().numpy()
    top1 = top5 = positive_frames = 0
    margins = []
    offset = 0
    for chunk in frame_chunks:
        end = offset + len(chunk)
        s, l = score[offset:end], y[offset:end]
        pos, neg = np.flatnonzero(l), np.flatnonzero(~l)
        if len(pos) and len(neg):
            positive_frames += 1
            order = np.argsort(-s, kind="stable")
            top1 += int(l[order[:1]].any()); top5 += int(l[order[:5]].any())
            margins.append(float(s[pos].min() - s[neg].max()))
        offset = end
    return {
        "candidate_count": int(len(y)), "positive_count": int(y.sum()),
        "roc_auc": auc(score, y), "pr_auc": average_precision(score, y),
        "positive_score": scalar_stats(score[y]), "negative_score": scalar_stats(score[~y]),
        "positive_hard_negative_margin": scalar_stats(margins),
        "hard_negative_violation_rate": float(np.mean(np.asarray(margins) < 0)) if margins else None,
        "positive_frame_count": positive_frames,
        "top1_frame_recall": float(top1 / max(1, positive_frames)),
        "top5_frame_recall": float(top5 / max(1, positive_frames)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    manifest_path, bank_root, out_root = map(Path, (args.manifest, args.bank_root, args.out_root))
    if not manifest_path.is_absolute(): manifest_path = ROOT / manifest_path
    if not bank_root.is_absolute(): bank_root = ROOT / bank_root
    if not out_root.is_absolute(): out_root = ROOT / out_root
    if out_root.exists(): raise FileExistsError(out_root)
    manifest = json.loads(manifest_path.read_text())
    rows = sorted(manifest["queries"], key=lambda row: int(row["query_index"]))
    if len(rows) != 160: raise ValueError("expected fixed 160-query manifest")
    metadata = load_metadata(); videos = sorted({str(row["video"]) for row in rows})
    banks = {video: load_bank(bank_root / "kitti" / f"{video}.pt") for video in videos}
    refs = make_refs(rows, metadata, banks)
    train_refs = [ref for ref in refs if ref["split"] == "calibration"]
    val_refs = [ref for ref in refs if ref["split"] == "screening"]
    rng = random.Random(args.seed)
    train_x, train_y, train_chunks = sampled_split(train_refs, banks, rng)
    val_x, val_y, val_chunks = sampled_split(val_refs, banks, random.Random(args.seed + 1))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed)
    model = nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x_tensor, y_tensor = torch.as_tensor(train_x), torch.as_tensor(train_y.astype(np.float32))
    model.train(); loss_rows=[]
    for step in range(1, args.steps + 1):
        indices = torch.randint(0, len(y_tensor), (min(4096, len(y_tensor)),))
        logits = model(x_tensor[indices].to(device)).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[indices].to(device))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)); optimizer.step()
        loss_rows.append((float(loss.detach().cpu()), grad_norm))
    model.eval()
    train_report = report(model, train_x, train_y, train_chunks, device)
    val_report = report(model, val_x, val_y, val_chunks, device)
    out_root.mkdir(parents=True, exist_ok=False)
    checkpoint = out_root / "linear_probe_step100.pt"
    torch.save({"format":"locatemot-l21-linear-probe-v1","step":args.steps,"seed":args.seed,
                "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "model":model.state_dict()}, checkpoint)
    payload={"format":"locatemot-l21-linear-probe-v1","provenance":{"manifest":str(manifest_path.resolve()),
        "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"checkpoint":str(checkpoint),
        "official_eval_used":False,"trackeval_used":False,"training_data":"calibration sampled frames",
        "validation_data":"screening sampled frames","features":"query+current_clip+geometry+objectness"},
        "sampling":{"train_candidates":int(len(train_y)),"train_positives":int(train_y.sum()),
        "val_candidates":int(len(val_y)),"val_positives":int(val_y.sum()),"train_frames":len(train_chunks),"val_frames":len(val_chunks)},
        "train":train_report,"validation":val_report,
        "loss":{"mean":float(np.mean([x[0] for x in loss_rows])),"last":loss_rows[-1][0],"mean_grad_norm":float(np.mean([x[1] for x in loss_rows]))}}
    (out_root/"linear_probe.json").write_text(json.dumps(payload,indent=2)+"\n")
    (out_root/"linear_probe.md").write_text("# F5 frozen-feature linear probe\n\n"+json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
