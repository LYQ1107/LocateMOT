#!/usr/bin/env python3
"""Label-free L77 representation/interface audit.

Only native L69 row pointers, frozen region vectors, and the frozen masked
L48 token cache are used.  No sidecar labels are opened in this audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l77_common import (  # noqa: E402
    L77Bank, MANIFEST, MANIFEST_SHA256, load_splits, load_text_cache,
    make_label_free_record, sha256_file, unit_tensors, write_json,
)
from locatemot.models.l77_region_cross_attention import L77RegionCrossAttention  # noqa: E402

SEED = 20260829
DEFAULT_OUT = ROOT / "outputs/l77/audit/representation_contract"


def choose_units(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    fit = [row for row in splits["fit"] if str(row.get("split")) == "fit"]
    chosen: list[dict[str, Any]] = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for category in ("positive", "multi_positive", "inactive", "present_uncovered"):
            values = sorted(
                (row for row in fit if str(row["dataset"]) == dataset and str(row.get("category")) == category),
                key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"]), str(row.get("unit_key", ""))),
            )
            if not values:
                raise AssertionError(f"missing label-free fit stratum {dataset}/{category}")
            chosen.append(values[0])
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    running: dict[str, Any] = {
        "format": "locatemot-l77-label-free-representation-contract-v1",
        "status": "running", "project_root": str(ROOT), "cwd": os.getcwd(),
        "command": " ".join(sys.argv), "seed": SEED,
        "training_run": False, "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "no_hota_or_trackeval": True, "raw_dense_feature_cache_written": False,
        "sidecar_labels_read": False, "validation_labels_read": False,
    }
    write_json(out / "status.json", running)
    started = time.perf_counter()
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA256:
            raise AssertionError("fixed manifest SHA mismatch")
        torch.manual_seed(SEED)
        if not torch.cuda.is_available():
            raise RuntimeError("L77 representation audit requires GPU0")
        device = torch.device(args.device)
        if device.type != "cuda" or device.index not in (None, 0):
            raise RuntimeError(f"L77 audit requires GPU0, got {device}")
        splits = load_splits()
        chosen = choose_units(splits)
        text_cache = load_text_cache()
        model = L77RegionCrossAttention(hidden=192, heads=4, dropout=0.0).to(device).eval()
        checks: list[dict[str, Any]] = []
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit in chosen:
            by_video[str(unit["video"])].append(unit)
        for video in sorted(by_video):
            bank = L77Bank(video)  # sidecar is not read by this class until attach_labels
            try:
                for unit in by_video[video]:
                    record = make_label_free_record(unit, bank)
                    data_cpu = unit_tensors(record, text_cache)
                    data = {key: value.to(device) for key, value in data_cpu.items()}
                    with torch.inference_mode():
                        output = model(data)
                    n = int(record["candidate_count"])
                    scores = output["match_logits"].float().cpu()
                    tokens = output["candidate_tokens"].float().cpu()
                    attention = output["cross_attention"].float().cpu()
                    finite = all(bool(torch.isfinite(value).all()) for value in output.values() if torch.is_tensor(value))
                    if scores.shape != (n,) or tokens.shape != (n, 192) or attention.ndim != 4:
                        raise AssertionError(f"L77 output shape drift for {record['unit_key']}")
                    if not finite or not torch.isfinite(scores).all():
                        raise AssertionError(f"nonfinite label-free output for {record['unit_key']}")
                    checks.append({
                        "unit_key": record["unit_key"], "dataset": record["dataset"],
                        "video": record["video"], "frame_id": record["frame_id"],
                        "declared_category": record["declared_category"],
                        "candidate_count": n, "row_keys": record["row_keys"],
                        "candidate_index_provenance": record["candidate_index"],
                        "text_shape": list(data_cpu["text"].shape),
                        "text_valid_tokens": int(data_cpu["text_mask"].sum()),
                        "region_shape": list(data_cpu["region"].shape),
                        "output_shapes": {
                            "match_logits": list(output["match_logits"].shape),
                            "candidate_tokens": list(output["candidate_tokens"].shape),
                            "cross_attention": list(output["cross_attention"].shape),
                            "absent_logit": list(output["absent_logit"].shape),
                        },
                        "score_mean": float(scores.mean()), "score_std": float(scores.std(unbiased=False)),
                        "candidate_token_std_mean": float(tokens.std(dim=0, unbiased=False).mean()),
                        "cross_attention_sum_min": float(attention.sum(dim=-1).min()),
                        "cross_attention_sum_max": float(attention.sum(dim=-1).max()),
                        "finite": finite, "ordered_keys": True,
                        "candidate_deletion": False, "candidate_truncation": False,
                        "labels_read": False, "feature_cache_written": False,
                    })
                    del data, data_cpu, output, scores, tokens, attention
            finally:
                bank.close()
        if len(checks) != 8:
            raise AssertionError(f"expected 8 strata checks, got {len(checks)}")
        if {row["dataset"] for row in checks} != {"refer_kitti_v1", "refer_kitti_v2"}:
            raise AssertionError("both domains not represented")
        if {row["declared_category"] for row in checks} != {"positive", "multi_positive", "inactive", "present_uncovered"}:
            raise AssertionError("four categories not represented")
        if any(row["candidate_count"] != len(row["row_keys"]) for row in checks):
            raise AssertionError("candidate key count drift")
        write_json(out / "config.json", {
            **running, "status": "complete", "device": str(device),
            "region_source": "frozen L69 clip[512]", "text_source": "frozen L48 masked text[64,768]",
            "model": model.parameter_summary(), "label_free_units": len(checks),
            "token_region_alignment": "UNALIGNED",
        })
        (out / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in checks))
        contract = {
            **running, "status": "complete", "elapsed_seconds": time.perf_counter() - started,
            "inputs": {
                "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                "l48_text_cache": str(ROOT / "outputs/l48/data/text_cache.pt"),
                "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
            },
            "outputs": {"records": str(out / "records.jsonl")},
            "counts": {"units": len(checks), "candidate_rows": int(sum(row["candidate_count"] for row in checks)),
                       "domains": sorted({row["dataset"] for row in checks}),
                       "declared_categories": sorted({row["declared_category"] for row in checks})},
            "checks": {
                "label_free": True, "sidecar_labels_read": False, "validation_labels_read": False,
                "finite": all(row["finite"] for row in checks), "candidate_key_drift": 0,
                "candidate_deletion": False, "candidate_truncation": False,
                "candidate_specific_output_nonconstant": all(row["score_std"] > 0.0 for row in checks),
                "word_mask_used": True, "raw_dense_feature_cache_written": False,
                "token_region_alignment": "UNALIGNED",
            },
            "next_action": "run L77 100-step fit-only smoke",
        }
        write_json(out / "contract.json", contract)
        write_json(out / "provenance.json", {
            **contract, "implementation": "new L77 region cross-attention head",
            "forbidden_inputs_excluded": ["source_id", "pool_id", "track_id", "group_id", "query_id", "old_scores"],
        })
        write_json(out / "status.json", contract)
        print(json.dumps({"status": "complete", "out": str(out), "candidate_rows": contract["counts"]["candidate_rows"]}), flush=True)
        return 0
    except Exception as exc:
        failure = {**running, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "elapsed_seconds": time.perf_counter() - started,
                   "next_action": "fix only first L77 representation-contract root cause and rerun in a new directory"}
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text("# L77 representation audit INCOMPLETE\n\n" +
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n```text\n" + traceback.format_exc() + "```\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
