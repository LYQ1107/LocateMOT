#!/usr/bin/env python3
"""Run real OneJev multimodal Yes/No probabilities on a held-out corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
ONEJEV = Path(os.environ.get("ONEJEV_ROOT", "/data2/user/LocateMOT_GroundJEV/external/OneJev"))
sys.path[:0] = [str(ROOT), str(ONEJEV)]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_digest(model: Path) -> str | None:
    for name in ("model.safetensors", "pytorch_model.bin"):
        value = sha256(model / name)
        if value:
            return value
    return None


def select_rows(path: Path, split: str, max_per_cell: int) -> list[dict]:
    cells: dict[tuple[str, int], list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != split:
                continue
            target = row.get("target") or {}
            yes = int("yes" in target)
            key = (str(row.get("task", "unknown")), yes)
            cells.setdefault(key, [])
            if len(cells[key]) < max_per_cell:
                cells[key].append(row)
    rows = [row for key in sorted(cells) for row in cells[key]]
    if not rows:
        raise ValueError(f"no rows for split={split} in {path}")
    return rows


def evaluate(args: argparse.Namespace) -> dict:
    import torch
    from train import common, mm

    mm.set_root(args.media_root)
    rows = select_rows(args.rows, args.split, args.max_per_cell)
    device = torch.device(args.device)
    bundle = common.load_bundle(
        str(args.model), device=str(device), dtype=args.dtype, train=False,
        multimodal=True, chat_template=None, load_model=True,
    )
    bundle.model.eval()
    examples = [common.render_example(row, bundle.template_for(row), i, seed=None, augment=False)
                for i, row in enumerate(rows)]
    lengths = [bundle.prompt_length(ex.prompt, rows[ex.idx]) for ex in examples]
    usable = [i for i, length in enumerate(lengths) if length <= args.max_len]
    dropped = [i for i in range(len(rows)) if i not in set(usable)]
    batches = common.length_bucket_batches(
        [lengths[i] for i in usable], args.batch_tokens, args.max_rows, shuffle_seed=None
    )
    results: list[dict] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            indices = [usable[j] for j in batch]
            exs = [examples[i] for i in indices]
            inputs, ends = mm.mm_inputs(
                bundle.processor, [ex.prompt for ex in exs],
                [rows[i].get("media") or [] for i in indices], device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vocab_logits = mm.mm_readout_logits(bundle.model, inputs, ends)
            selected, _ = common.slot_view(vocab_logits, bundle.slot_ids, [2] * len(exs))
            probs = torch.softmax(selected.float(), dim=-1).cpu().tolist()
            for local, index in enumerate(indices):
                row = rows[index]
                p_yes = float(probs[local][0])
                target_yes = int("yes" in (row.get("target") or {}))
                results.append({
                    "id": row["id"], "task": row.get("task"), "video": row.get("source"),
                    "source": row.get("source"), "target_yes": target_yes,
                    "p_yes": p_yes, "p_no": float(probs[local][1]),
                    "predicted_yes": int(p_yes >= 0.5),
                    "uncertain_reject": bool(max(probs[local]) < args.reject_confidence),
                    "family": (row.get("supervision") or {}).get("family"),
                    "frame": (row.get("state") or {}).get("timestamp"),
                })
    elapsed = time.perf_counter() - t0
    out = {
        "status": "REAL_ONEJEV_MULTIMODAL_YESNO_PREDICTIONS_COMPLETE",
        "scientific_result": True,
        "model": str(args.model.resolve()),
        "model_sha256": model_digest(args.model),
        "rows": str(args.rows.resolve()),
        "rows_sha256": sha256(args.rows),
        "split": args.split,
        "requested_rows": len(rows),
        "valid_predictions": len(results),
        "dropped_over_max_len": len(dropped),
        "dropped_ids": [rows[i]["id"] for i in dropped],
        "selection": {"max_per_task_label_cell": args.max_per_cell,
                       "task_label_counts": {f"{task}|{label}": count for (task, label), count in
                                               Counter((r["task"], r["target_yes"]) for r in results).items()}},
        "reject_confidence": args.reject_confidence,
        "uncertain_reject_count": sum(int(r["uncertain_reject"]) for r in results),
        "uncertain_reject_rate": sum(int(r["uncertain_reject"]) for r in results) / max(1, len(results)),
        "elapsed_seconds": elapsed,
        "fps_equivalent_rows_per_second": len(results) / max(elapsed, 1e-9),
        "cuda_device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        "peak_memory_gib": (torch.cuda.max_memory_allocated(device) / 2**30) if torch.cuda.is_available() else 0.0,
        "official_test_labels_read": False,
        "predictions": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-per-cell", type=int, default=30)
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--batch-tokens", type=int, default=8192)
    p.add_argument("--max-rows", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--media-root", type=Path, default=Path("/data2/user/LocateMOT_GroundJEV"))
    p.add_argument("--reject-confidence", type=float, default=0.60)
    args = p.parse_args()
    report = evaluate(args)
    print(json.dumps({k: report[k] for k in ("status", "model", "valid_predictions", "dropped_over_max_len", "elapsed_seconds", "uncertain_reject_rate")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
