#!/usr/bin/env python3
"""Evaluate native YesTrack rows with the frozen base or a LoRA adapter.

Reports AUROC, average precision (PR-AUC), accuracy, Brier score, ECE and
false-rejection/false-acceptance rates from the actual yes/no slot
probability. It never reads GT into the model prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "external/OneJev")]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank_sum += (i + 1 + j) * sum(labels[k] for k in order[i:j]) / 2.0
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    pos = sum(labels)
    if not pos:
        return None
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    hit = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            hit += 1
            total += hit / rank
    return total / pos


def ece(labels: list[int], scores: list[float], bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, score in enumerate(scores) if lo <= score < hi or (b == bins - 1 and score == hi)]
        if idx:
            total += len(idx) / len(scores) * abs(sum(scores[i] for i in idx) / len(idx) -
                                                   sum(labels[i] for i in idx) / len(idx))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path(
        "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training/onejev_unified600_native_v2/ckpt-step600"))
    parser.add_argument("--runtime-root", type=Path, default=Path("/data2/user/LocateMOT_GroundJEV"),
                        help="checkout containing external/OneJev and its native train package")
    parser.add_argument("--rows", type=Path, default=Path(
        "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/native_yesno_v2_phaseb_val.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional PEFT adapter directory; omit for the frozen base")
    parser.add_argument("--media-root", type=Path, default=Path("/data2/user/LocateMOT_GroundJEV"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=32768)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    onejev_root = runtime_root / "external/OneJev"
    if onejev_root.is_dir() and str(onejev_root) not in sys.path:
        sys.path.insert(0, str(onejev_root))
    import torch
    from peft import PeftModel
    from train import common, mm

    rows = common.load_rows(args.rows)
    if not rows:
        raise ValueError("validation rows are empty")
    mm.set_root(args.media_root)
    bundle = common.load_bundle(str(args.base_model), device=args.device, dtype="bfloat16",
                                train=False, multimodal=True, load_model=True)
    model = bundle.model
    if args.checkpoint is not None:
        model = PeftModel.from_pretrained(model, str(args.checkpoint), is_trainable=False)
    model.eval()

    examples = [common.render_example(row, bundle.template_for(row), i, seed=None, augment=False)
                for i, row in enumerate(rows)]
    lengths = [bundle.prompt_length(example.prompt, rows[example.idx]) for example in examples]
    if any(length > args.max_len for length in lengths):
        raise ValueError(f"validation row exceeds max_len={args.max_len}")
    batches = common.length_bucket_batches(lengths, args.batch_tokens, args.max_rows, shuffle_seed=None)
    labels: list[int] = []
    scores: list[float] = []
    records: list[dict] = []
    with torch.inference_mode():
        for batch in batches:
            selected = [examples[j] for j in batch]
            inputs, ends = mm.mm_inputs(bundle.processor, [example.prompt for example in selected],
                                        [rows[example.idx].get("media") or [] for example in selected], args.device)
            logits = mm.mm_readout_logits(model, inputs, ends)
            slots, _ = common.slot_view(logits, bundle.slot_ids, [len(example.labels) for example in selected])
            probs = torch.softmax(slots, dim=-1)[:, 0].float().cpu().tolist()
            for example, score in zip(selected, probs):
                label = 1 if float(rows[example.idx]["target"].get("yes", 0.0)) >= 0.5 else 0
                labels.append(label); scores.append(float(score))
                records.append({"id": rows[example.idx]["id"], "label": label,
                                "p_yes": float(score), "task": rows[example.idx].get("task"),
                                "source": rows[example.idx].get("source")})

    predicted = [int(score >= 0.5) for score in scores]
    positives = sum(labels)
    negatives = len(labels) - positives
    false_reject = sum(label == 1 and pred == 0 for label, pred in zip(labels, predicted)) / max(positives, 1)
    false_accept = sum(label == 0 and pred == 1 for label, pred in zip(labels, predicted)) / max(negatives, 1)
    receipt = {
        "status": "YESTRACK_NATIVE_SLOT_EVAL_COMPLETE",
        "rows": str(args.rows), "rows_sha256": sha256(args.rows), "row_count": len(rows),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "frozen_base",
        "checkpoint_sha256": sha256(args.checkpoint / "adapter_model.safetensors") if args.checkpoint and (args.checkpoint / "adapter_model.safetensors").is_file() else None,
        "metrics": {
            "auroc": auc(labels, scores), "pr_auc_average_precision": average_precision(labels, scores),
            "accuracy_at_0.5": sum(a == b for a, b in zip(labels, predicted)) / len(labels),
            "brier": sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(labels),
            "ece_10_bins": ece(labels, scores), "false_rejection_rate_at_0.5": false_reject,
            "false_acceptance_rate_at_0.5": false_accept,
        },
        "class_counts": {"yes": positives, "no": negatives},
        "gt_used_for_prompt": False, "official_test_labels_read": False,
        "future_evidence": False, "media_root": str(args.media_root),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "row_count", "checkpoint", "metrics", "class_counts")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
