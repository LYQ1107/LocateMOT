#!/usr/bin/env python3
"""Small L85 forward/loss/gradient/reload contract audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from locatemot.models.l85_full_rmot import L85Config, L85FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore, key_only, load_fit_units  # noqa: E402
from locatemot.rmot.l85_fullvideo_bank import EXPECTED_MANIFEST_SHA, MANIFEST, sha256_file  # noqa: E402
from locatemot.rmot.l85_losses import l85_loss  # noqa: E402
from tools.train_l85_full_rmot import history_for_stage, load_group_inputs, load_cache_manifest, row_digest  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def select_groups(rows: list[dict[str, Any]], fit_units: list[dict[str, Any]]) -> list[str]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for unit in fit_units:
        key = f"{unit['dataset']}|{unit['video']}|{int(unit['frame_id'])}"
        by_group[key].add(str(unit.get("category", "unknown")))
    selected: list[str] = []
    required = [
        ("refer_kitti_v1", "positive"), ("refer_kitti_v1", "multi_positive"),
        ("refer_kitti_v2", "inactive"), ("refer_kitti_v2", "present_uncovered"),
    ]
    for dataset, category in required:
        for row in rows:
            if str(row["dataset"]) == dataset and category in by_group.get(str(row["group_key"]), set()):
                if str(row["group_key"]) not in selected:
                    selected.append(str(row["group_key"]))
                break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("manifest SHA drift")
        cache_root = (args.cache if args.cache.is_absolute() else ROOT / args.cache).resolve()
        summary = json.loads((cache_root / "summary.json").read_text())
        if summary.get("status") != "complete" or summary.get("labels_in_cache"):
            raise AssertionError("cache is not complete label-free L85 cache")
        cache_rows = load_cache_manifest(cache_root)
        fit_units = load_fit_units()
        selected = select_groups(cache_rows, fit_units)
        categories_by_group: dict[str, set[str]] = defaultdict(set)
        for unit in fit_units:
            categories_by_group[f"{unit['dataset']}|{unit['video']}|{int(unit['frame_id'])}"].add(str(unit.get("category", "unknown")))
        selected_categories = set()
        for group_key in selected:
            selected_categories.update(categories_by_group[group_key])
        required_categories = {"positive", "multi_positive", "inactive", "present_uncovered"}
        if not required_categories.issubset(selected_categories):
            raise AssertionError(f"selected groups do not cover four categories: {selected}")
        if not selected:
            raise AssertionError("no contract groups selected")
        labels_by_key = {str(row["unit_key"]): row for row in fit_units}
        store = L80BankStore(max_history=8)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = L85FullRMOT(L85Config(hidden=256)).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        records = []
        for group_key in selected:
            entry = next(row for row in cache_rows if str(row["group_key"]) == group_key)
            item = torch.load(entry["path"], map_location="cpu", weights_only=False)
            first, label_list = load_group_inputs(item, labels_by_key, store)
            history, history_mask, history_frames = history_for_stage(first, "J")
            if bool((history_frames[history_mask] > int(first.frame_id)).any()):
                raise AssertionError(f"future history in {group_key}")
            z1 = item["z1"].float().clone().to(device)
            presence = torch.cat((item["text_global"].float(), item["frame_global"].float()), dim=-1).clone().to(device)
            current = first.observations.float().clone().to(device)
            history = history.float().clone().to(device)
            history_mask = history_mask.to(device)
            history_frames = history_frames.to(device)
            output = model(z1, presence, current, history, history_mask, history_frames,
                           int(first.frame_id), temporal_enabled=True)
            labels = [x["labels"] for x in label_list]
            masks = [x["membership_mask"] for x in label_list]
            categories = [str(x["category"]) for x in label_list]
            loss, parts = l85_loss(output, labels, masks, categories, current, history_mask, temporal_enabled=True)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss {group_key}")
            positive_gradient = negative_gradient = minimum_gradient = False
            for q, label in enumerate(labels):
                pos = torch.nonzero(label, as_tuple=False).flatten()
                neg = torch.nonzero(~label, as_tuple=False).flatten()
                for index, kind in ((pos[0] if pos.numel() else None, "positive"),
                                    (neg[0] if neg.numel() else None, "negative"),
                                    (pos[-1] if pos.numel() else None, "minimum")):
                    if index is None:
                        continue
                    gradients = torch.autograd.grad(output["membership"][q, int(index)], tuple(model.parameters()),
                                                    retain_graph=True, allow_unused=True)
                    active = any(g is not None and bool(torch.isfinite(g).all()) and bool((g.abs() > 0).any()) for g in gradients)
                    if kind == "positive": positive_gradient = positive_gradient or active
                    if kind == "negative": negative_gradient = negative_gradient or active
                    if kind == "minimum": minimum_gradient = minimum_gradient or active
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite_grad = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
            nonzero_grad = any(parameter.grad is not None and bool((parameter.grad.abs() > 0).any()) for parameter in model.parameters())
            if not finite_grad or not nonzero_grad:
                raise FloatingPointError(f"invalid aggregate gradient {group_key}")
            optimizer.step()
            records.append({"group_key": group_key, "dataset": first.dataset, "video": first.video,
                            "categories": categories, "candidate_count": first.candidate_count,
                            "candidate_key_digest": row_digest(first.row_keys), "positive_gradient": positive_gradient,
                            "negative_gradient": negative_gradient, "minimum_positive_gradient": minimum_gradient,
                            "future_history_rows": int((history_frames[history_mask] > int(first.frame_id)).sum()),
                            "loss": parts, "finite": True, "gradient_finite": finite_grad,
                            "gradient_nonzero": nonzero_grad, "candidate_deletion": False,
                            "candidate_truncation": False})
            del output, loss, item, first, label_list, z1, presence, current, history, history_mask, history_frames
        checkpoint = out / "contract_checkpoint.pt"
        torch.save({"format": "locatemot-l85-forward-contract-checkpoint-v1", "model_config": L85Config(hidden=256).__dict__,
                    "model_state_dict": model.state_dict(), "step": 1}, checkpoint)
        package = torch.load(checkpoint, map_location=device, weights_only=False)
        reloaded = L85FullRMOT(L85Config(**package["model_config"])).to(device=device, dtype=torch.float32)
        result = reloaded.load_state_dict(package["model_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise AssertionError(f"strict reload mismatch: {result}")
        contract = {"format": "locatemot-l85-forward-loss-contract-v1", "status": "complete", "command": command,
                    "cwd": str(ROOT), "device": str(device), "groups": records, "selected_groups": selected,
                    "model": model.parameter_report(), "strict_reload": True, "missing_keys": [], "unexpected_keys": [],
                    "candidate_keys_complete": True, "future_history_rows": 0, "candidate_deletion": False,
                    "candidate_truncation": False, "raw_dense_cache_written": False,
                    "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
                    "screening_gt_used": False, "official_test_labels_read": False,
                    "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                    "failure_root_cause": None, "next_action": "run L85 fit smoke"}
        write_json(out / "contract.json", contract)
        write_json(out / "provenance.json", {**contract, "inputs": {"cache": str(cache_root),
            "cache_summary_sha256": hashlib.sha256((cache_root / "summary.json").read_bytes()).hexdigest(),
            "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
            "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl")},
            "labels_attached_after_cache_construction": True, "checkpoint": str(checkpoint)})
        write_json(out / "status.json", contract)
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L85 forward/loss contract — INCOMPLETE\n\n" + traceback.format_exc() + "\n")
        write_json(out / "status.json", {"format": "locatemot-l85-forward-loss-contract-v1", "status": "incomplete",
                                          "command": command, "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "next_action": "fix only the first actionable contract error",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
