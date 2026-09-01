#!/usr/bin/env python3
"""Evaluate the frozen L32 gate on the fixed 100-unit screening audit cache."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l32_agreement_gate import L32AgreementGate


def metrics(unit, score, reject=None):
    y = unit["label"].astype(bool)
    chosen = score >= 0.0
    if reject is not None:
        chosen = chosen & (reject >= 0.5)
    tp, fp, fn = chosen & y, chosen & ~y, ~chosen & y
    top1, top5, set_recall, strict, null_accept, empty, mp = [], [], [], [], 0, 0, []
    for uid in np.unique(unit["unit_id"]):
        idx = np.flatnonzero(unit["unit_id"] == uid)
        order = idx[np.argsort(-score[idx], kind="stable")]
        pos = y[idx]; neg = ~pos
        if not pos.any():
            null_accept += int(chosen[idx].any()); empty += int(not chosen[idx].any()); continue
        top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any()))
        k = min(len(order), int(pos.sum()))
        set_recall.append(float(y[order[:k]].sum() / max(1, int(pos.sum()))))
        mp.append(float(pos.sum() > 1) * float(y[order[:1]].any()))
        if neg.any(): strict.append(float(score[idx][pos].min() - score[idx][neg].max()))
    source = {}
    for source_id, name in ((0, "main"), (1, "reserve")):
        pool = unit["source"] == source_id
        source[name] = {"selected": int((chosen & pool).sum()), "positive": int((y & pool).sum()),
                        "precision": float((chosen & pool & y).sum() / max(1, (chosen & pool).sum())),
                        "recall": float((chosen & pool & y).sum() / max(1, (y & pool).sum()))}
    return {"rows": int(len(y)), "positive_rows": int(y.sum()), "selected": int(chosen.sum()),
            "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()),
            "precision": float(tp.sum() / max(1, chosen.sum())), "recall": float(tp.sum() / max(1, y.sum())),
            "fp_per_frame": float(fp.sum() / max(1, len(np.unique(unit["unit_id"])))),
            "predictions_per_positive": float(chosen.sum() / max(1, y.sum())),
            "top1": float(np.mean(top1)) if top1 else None, "top5": float(np.mean(top5)) if top5 else None,
            "multi_positive_top1": float(np.mean(mp)) if mp else None,
            "multi_positive_count": int(sum(x > 0 for x in mp)),
            "topk_set_recall": float(np.mean(set_recall)) if set_recall else None,
            "hard_violation": float(np.mean(np.asarray(strict) < 0)) if strict else None,
            "strict_margin": float(np.mean(strict)) if strict else None,
            "empty_rate": float(empty / max(1, len(np.unique(unit["unit_id"])))),
            "null_false_acceptance": float(null_accept / max(1, len(np.unique(unit["unit_id"])))),
            "source_precision": source, "selection": "score>=0 plus optional reject>=0.5"}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--audit-root", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audit = Path(args.audit_root); z = np.load(audit / "score_cache.npz", allow_pickle=False)
    split = np.asarray(z["split"]); keep = split == 1
    unit = {key: np.asarray(z[key])[keep] for key in ("unit_id", "source", "label", "membership", "association", "recency", "motion_norm", "lifecycle_norm")}
    features = np.stack((unit["membership"], unit["association"], unit["recency"] / 8.0,
                         unit["motion_norm"], unit["lifecycle_norm"],
                         np.tanh(unit["association"] / 3.0)), axis=1).astype(np.float32)
    gate = L32AgreementGate(); state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    gate.load_state_dict(state); gate.eval()
    with torch.inference_mode():
        output = gate(torch.as_tensor(features), torch.as_tensor(unit["membership"]))
    residual = output["residual"].numpy(); final = output["final"].numpy(); reject = torch.sigmoid(output["reject_logit"]).numpy()
    result = {"membership_only": metrics(unit, unit["membership"]),
              "bounded_residual": metrics(unit, final),
              "reject_gated": metrics(unit, final, reject)}
    payload = {"format": "locatemot-l32-agreement-gate-eval-v1", "audit_root": str(audit.resolve()),
               "checkpoint": str(Path(args.checkpoint).resolve()), "screening_units": int(len(np.unique(unit["unit_id"]))),
               "screening_gt_used_for_model_or_selection": False, "residual_abs_max": float(np.max(np.abs(residual))),
               "reject_rate": float(np.mean(reject >= 0.5)), "strategies": result}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
