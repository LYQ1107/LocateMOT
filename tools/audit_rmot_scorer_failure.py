"""CPU-only failure decomposition for the 250-step candidate scorer."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.rmot_candidate_scorer import RMOTCandidateScorer  # noqa: E402
from tools.train_rmot_candidate_scorer import (  # noqa: E402
    auc, average_precision, load_bank, load_metadata, make_refs, scalar_stats,
)


def margin_loss(positive: torch.Tensor, negative: torch.Tensor,
                margin: float = 0.5) -> torch.Tensor:
    return torch.nn.functional.softplus(margin - (positive - negative))


def direction_check() -> dict:
    high = float(margin_loss(torch.tensor(2.0), torch.tensor(0.0)))
    low = float(margin_loss(torch.tensor(0.0), torch.tensor(2.0)))
    positive = torch.tensor(0.0, requires_grad=True)
    negative = torch.tensor(0.0, requires_grad=True)
    value = margin_loss(positive, negative)
    value.backward()
    result = {
        "positive_high_negative_low_loss": high,
        "positive_low_negative_high_loss": low,
        "loss_decreases_when_positive_exceeds_negative": bool(high < low),
        "gradient_positive_at_equal": float(positive.grad),
        "gradient_negative_at_equal": float(negative.grad),
    }
    if not result["loss_decreases_when_positive_exceeds_negative"] or \
            not (result["gradient_positive_at_equal"] < 0 and
                 result["gradient_negative_at_equal"] > 0):
        raise AssertionError(result)
    return result


def score_frame(model, query: np.ndarray, tensors: dict, begin: int,
                end: int, device: torch.device, batch_size: int) -> np.ndarray:
    scores = []
    query_tensor = torch.as_tensor(query, dtype=torch.float32).reshape(1, -1)
    for start in range(begin, end, batch_size):
        stop = min(end, start + batch_size)
        count = stop - start
        index = torch.arange(start, stop, dtype=torch.long)
        values = {
            "query": query_tensor.expand(count, -1).to(device),
            "static_query": query_tensor.expand(count, -1).to(device),
            "motion_query": query_tensor.expand(count, -1).to(device),
            "current": tensors["clip"].float().index_select(0, index).to(device),
            "history": tensors["history_clip"].float().index_select(0, index).to(device),
            "geometry": tensors["geometry"].float().index_select(0, index).to(device),
            "motion": tensors["motion"].float().index_select(0, index).to(device),
            "objectness": tensors["objectness"].float().index_select(0, index).reshape(count, 1).to(device),
            "delta": torch.zeros(count, 1, device=device),
        }
        output = model(**values) if False else model(
            values["query"], values["static_query"], values["motion_query"],
            values["current"], values["history"], values["geometry"],
            frame_delta=values["delta"], motion_feature=values["motion"],
            objectness=values["objectness"],
        )
        scores.append(output["final_candidate_logit"].float().cpu().numpy())
    return np.concatenate(scores)


def report_split(model, refs: list[dict], banks: dict[str, dict],
                 device: torch.device, batch_size: int) -> dict:
    positive_scores, train_hard_scores, easy_scores = [], [], []
    model_hard_scores, top1_margins, positive_hard_margins = [], [], []
    train_hard_model_max, model_hard_in_train_hard = [], []
    labels_all, scores_all = [], []
    frame_count = positive_frame_count = 0
    query_ids = sorted({ref["query_index"] for ref in refs})
    for query_index in query_ids:
        query_refs = sorted(
            [ref for ref in refs if ref["query_index"] == query_index],
            key=lambda ref: ref["frame_index"])
        bank = banks[query_refs[0]["video"]]
        tensors = bank["tensors"]
        for ref in query_refs:
            frame_count += 1
            scores = score_frame(model, ref["spec"], tensors,
                                 ref["begin"], ref["end"], device, batch_size)
            labels = ref["positive"]
            objectness = tensors["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
            positives = np.flatnonzero(labels)
            negatives = np.flatnonzero(~labels)
            labels_all.append(labels)
            scores_all.append(scores)
            if len(positives):
                positive_frame_count += 1
                positive_scores.extend(scores[positives].tolist())
            if not len(negatives):
                continue
            hard_count = min(len(negatives), 12) if len(negatives) > 24 else len(negatives)
            train_hard = negatives[np.argsort(-objectness[negatives], kind="stable")[:hard_count]]
            train_hard_set = set(train_hard.tolist())
            easy = np.asarray([index for index in negatives if int(index) not in train_hard_set], dtype=np.int64)
            train_hard_scores.extend(scores[train_hard].tolist())
            easy_scores.extend(scores[easy].tolist())
            model_hard_index = int(negatives[np.argmax(scores[negatives])])
            model_hard_scores.append(float(scores[model_hard_index]))
            train_hard_model_max.append(float(scores[train_hard].max()))
            model_hard_in_train_hard.append(int(model_hard_index in train_hard_set))
            if len(positives):
                positive_hard_margins.append(float(scores[positives].min() - scores[negatives].max()))
                top1_margins.append(float(scores[positives].max() - scores[negatives].max()))
    labels = np.concatenate(labels_all).astype(bool)
    scores = np.concatenate(scores_all).astype(np.float32)
    return {
        "query_count": len(query_ids), "candidate_count": int(len(labels)),
        "positive_count": int(labels.sum()), "frame_count": frame_count,
        "positive_frame_count": positive_frame_count,
        "pooled_auc": auc(scores, labels), "pooled_pr_auc": average_precision(scores, labels),
        "positive_score": scalar_stats(positive_scores),
        "hard_negative_score_train_objectness_top12": scalar_stats(train_hard_scores),
        "easy_negative_score": scalar_stats(easy_scores),
        "hard_negative_score_validation_model_max": scalar_stats(model_hard_scores),
        "positive_minus_hard_negative_margin": scalar_stats(positive_hard_margins),
        "top1_positive_minus_hard_negative_margin": scalar_stats(top1_margins),
        "train_hard_score_max": scalar_stats(train_hard_model_max),
        "validation_model_hard_in_train_objectness_hard_rate": float(
            np.mean(model_hard_in_train_hard)) if model_hard_in_train_hard else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    bank_root = Path(args.bank_root)
    checkpoint_path = Path(args.checkpoint)
    out_root = Path(args.out_root)
    for value in (manifest_path, bank_root, checkpoint_path, out_root):
        if not value.is_absolute(): value = ROOT / value
    manifest_path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
    bank_root = bank_root if bank_root.is_absolute() else ROOT / bank_root
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    if out_root.exists():
        raise FileExistsError(out_root)
    manifest = json.loads(manifest_path.read_text())
    rows = sorted(manifest["queries"], key=lambda row: int(row["query_index"]))
    if len(rows) != 160 or manifest.get("selection_uses_model_scores", True):
        raise ValueError("not the fixed score-independent 160-query manifest")
    metadata = load_metadata()
    videos = sorted({str(row["video"]) for row in rows})
    banks = {video: load_bank(bank_root / "kitti" / f"{video}.pt") for video in videos}
    refs = make_refs(rows, metadata, banks)
    train_refs = [ref for ref in refs if ref["split"] == "calibration"]
    val_refs = [ref for ref in refs if ref["split"] == "screening"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RMOTCandidateScorer().cpu()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    direction = direction_check()
    with torch.inference_mode():
        train = report_split(model, train_refs, banks, torch.device("cpu"), args.batch_size)
        validation = report_split(model, val_refs, banks, torch.device("cpu"), args.batch_size)
    payload = {
        "format": "locatemot-rmot-scorer-failure-decomposition-v1",
        "provenance": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": checkpoint["manifest_sha256"],
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_step": checkpoint["step"], "seed": checkpoint["seed"],
            "device": "cpu", "trackeval_used": False, "training_used": False,
            "official_eval_used": False,
        },
        "implementation_audit": {
            "training_hard_negative": "per frame, negative rows ranked by frozen objectness; top 12 when negative_count>24, otherwise all negatives; remaining sampled negatives are random",
            "validation_hard_negative": "per frame, highest final scorer logit among every negative row",
            "hard_negative_definitions_match": False,
            "pairwise_expression": "softplus(pair_margin - (positive_logit - negative_logit))",
            "pairwise_margin": checkpoint["config"]["pair_margin"],
            "pairwise_detached": False, "pairwise_gradient_path": True,
            "source_in_score": False, "grouping": False, "membership": False,
            "null_scalar_subtraction": False,
        },
        "manual_pairwise_direction_check": direction,
        "train": train, "validation": validation,
    }
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "failure_decomposition.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# RMOT scorer 250-step failure decomposition", "",
        "CPU-only, frozen bank/checkpoint, no training and no TrackEval.", "",
        f"Train pooled AUC/PR-AUC: {train['pooled_auc']:.6f}/{train['pooled_pr_auc']:.6f}.",
        f"Validation pooled AUC/PR-AUC: {validation['pooled_auc']:.6f}/{validation['pooled_pr_auc']:.6f}.",
        "", "## Hard-negative contract", "",
        "Training hard negatives are frozen-objectness top-12 negatives plus random negatives; validation hard negatives are the scorer-max negative over the full frame. These definitions do not match.",
        f"Validation model-hard rows that are also training objectness-hard: {validation['validation_model_hard_in_train_objectness_hard_rate']:.6f}.",
        "", "## Manual pairwise check", "",
        f"Loss(high positive=2, negative=0)={direction['positive_high_negative_low_loss']:.6f}; loss(low positive=0, negative=2)={direction['positive_low_negative_high_loss']:.6f}; direction PASS.",
        "", "The JSON contains positive, training-hard, easy-negative, validation-model-hard, positive-minus-hard, and top-1 margin distributions for both splits.", "",
    ]
    (out_root / "failure_decomposition.md").write_text("\n".join(lines))
    print(json.dumps({"status": "complete", "output": str(out_root), "manual_pairwise": direction, "train": train, "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
