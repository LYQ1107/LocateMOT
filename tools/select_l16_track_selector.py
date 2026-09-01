"""Select and calibrate L16 checkpoints on one fixed train-validation sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import L16TrackSelector  # noqa: E402
from tools.train_l16_track_selector import (  # noqa: E402
    BankStore, load_expressions, validate,
)


def checkpoints(directory: Path):
    rows = []
    for path in directory.glob("step*.pt"):
        match = re.fullmatch(r"step(\d+)\.pt", path.name)
        if match and int(match.group(1)) % 250 == 0:
            rows.append((int(match.group(1)), path))
    return sorted(rows)


def run_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items, _ = load_expressions()
    store = BankStore(cache_size=64)
    rows = []
    selected = [row for index, row in enumerate(checkpoints(args.directory))
                if index % args.num_shards == args.shard]
    for step, path in selected:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["cfg"]
        model = L16TrackSelector(cfg["hidden"], cfg["heads"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        result = validate(
            model, items["train_val"], store, args.seed, args.episodes,
            int(cfg["sequence_length"]), device)
        result.update({
            "step": step, "checkpoint": str(path),
            "domain_macro_loss": sum(result["domain_loss"].values()) /
                                 len(result["domain_loss"]),
        })
        rows.append(result)
        print(f"[l16-select] step={step} macro={result['domain_macro_loss']:.5f} "
              f"loss={result['loss']:.5f} f1={result['f1']:.4f} "
              f"thr={result['threshold']:.3f}", flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"shard_{args.shard:02d}.json"
    path.write_text(json.dumps({
        "seed": args.seed, "episodes": args.episodes,
        "shard": args.shard, "num_shards": args.num_shards,
        "results": rows,
    }, indent=2) + "\n")


def merge(args):
    rows = []
    for path in sorted(args.output_dir.glob("shard_*.json")):
        value = json.loads(path.read_text())
        if value["seed"] != args.seed or value["episodes"] != args.episodes:
            raise RuntimeError(f"selection protocol mismatch in {path}")
        rows.extend(value["results"])
    expected = [step for step, _ in checkpoints(args.directory)]
    if sorted(row["step"] for row in rows) != expected:
        raise RuntimeError("selection results do not cover every formal checkpoint")
    best = min(rows, key=lambda row: (row["domain_macro_loss"], row["step"]))
    checkpoint = torch.load(best["checkpoint"], map_location="cpu",
                            weights_only=False)
    checkpoint["calibration"] = {
        key: best[key] for key in (
            "loss", "domain_loss", "examples", "positive_rate", "threshold",
            "f1", "precision", "recall", "domain_macro_loss")
    }
    checkpoint["selection"] = {
        "rule": "minimum domain-equal loss on one fixed train-validation sample",
        "seed": args.seed, "episodes": args.episodes,
        "selected_step": best["step"], "all_results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    summary = {
        "selected": best, "all_results": sorted(rows, key=lambda row: row["step"]),
        "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["selected"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path,
                        default=Path("outputs/l16/checkpoints"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/l16/selection"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/l16/checkpoints/track_selector_joint_selected.pt"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
