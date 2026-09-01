#!/usr/bin/env python3
"""L75 label-free candidate-marker/API contract audit.

This audit deliberately stops before reading candidate labels.  It proves that
two complete L69 candidate rows can reuse one frozen visual forward while
their candidate markers produce different post-fusion expression states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l75_candidate_marked_vlm import CandidateMarkedVLMMatcher  # noqa: E402
from locatemot.rmot.l75_data import (  # noqa: E402
    L75Bank, IMAGE_ROOT, MANIFEST_PATH, MANIFEST_SHA256, load_splits,
    make_record, sha256_file, unit_key,
)
from locatemot.rmot.l75_runtime import (  # noqa: E402
    build_messages, expression_positions, language_forward,
    marked_visual_batch, model_file_manifest, prepare_visual,
)

SEED = 20260829
CONTROL_BASE = "small red cars moving toward front"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def equal_length_control(tokenizer: Any, expression: str) -> tuple[str, int, bool]:
    ids = tokenizer(expression, add_special_tokens=False)["input_ids"]
    ids = ids[0] if ids and isinstance(ids[0], list) else ids
    base = tokenizer(CONTROL_BASE, add_special_tokens=False)["input_ids"]
    base = base[0] if base and isinstance(base[0], list) else base
    if not ids or not base:
        return CONTROL_BASE, len(ids), False
    repeats = (len(ids) + len(base) - 1) // len(base)
    control_ids = (list(base) * repeats)[:len(ids)]
    text = tokenizer.decode(control_ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    check = tokenizer(text, add_special_tokens=False)["input_ids"]
    check = check[0] if check and isinstance(check[0], list) else check
    return text, len(ids), len(check) == len(ids)


def text_only_prepared(processor: Any, tokenizer: Any, image: Any,
                       expression: str, base_visual: torch.Tensor,
                       original_prepared: dict[str, Any]) -> dict[str, Any]:
    inputs, prompt = build_messages(processor, image, expression)
    ids = inputs["input_ids"].detach().cpu()
    mask = inputs.get("attention_mask")
    mask = mask.detach().cpu() if mask is not None else None
    # Use the model's actual index and explicitly check visual-token count.
    image_token_index = int(original_prepared.get("image_token_index", 151665))
    image_pos = [i for i, value in enumerate(ids.reshape(-1).tolist())
                 if int(value) == image_token_index]
    if len(image_pos) != len(original_prepared["image_positions"]):
        raise AssertionError("equal-length control changed image-token count")
    expr_pos, method = expression_positions(tokenizer, expression, ids, mask, image_pos)
    if not expr_pos:
        raise AssertionError("equal-length control has no usable text positions")
    return {
        "base_visual": base_visual,
        "input_ids": ids,
        "attention_mask": mask,
        "image_positions": image_pos,
        "expression_positions": expr_pos,
        "expression_span_method": method,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "image_token_index": image_token_index,
        "candidate_cells": original_prepared["candidate_cells"],
    }


def vector_rel_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    a, b = first.detach().float(), second.detach().float()
    return float((a - b).norm() / a.norm().clamp_min(1e-8))


def select_label_free_units(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    # Dataset is used only to ensure both domains are represented.  No
    # category/target/positive field is read for this label-free audit.
    selected = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        rows = sorted(
            [row for row in splits["fit"] if str(row["dataset"]) == dataset],
            key=unit_key,
        )[:2]
        selected.extend(rows)
    if len(selected) != 4:
        raise AssertionError("could not select two label-free fit units per domain")
    return selected


def tiny_lattice_fixture() -> dict[str, Any]:
    base = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    marker = torch.ones(5, dtype=torch.float32)
    marked, mask = marked_visual_batch(base, [[0, 2], [1]], marker)
    delta = marked - base.unsqueeze(0)
    expected = torch.zeros_like(mask, dtype=torch.float32)
    expected[0, [0, 2]] = 1.0
    expected[1, 1] = 1.0
    if not torch.equal(mask, expected.bool()):
        raise AssertionError("tiny lattice marker mask mismatch")
    if not bool((delta == expected.unsqueeze(-1)).all()):
        raise AssertionError("tiny lattice marker crossed candidate cells")
    return {"candidate_count": 2, "lattice_cells": 3, "cross_cell_delta": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs/l75/audit/candidate_marker_attempt1")
    args = parser.parse_args()
    out = args.out
    started = time.perf_counter()
    base = {
        "format": "locatemot-l75-candidate-marker-audit-v1",
        "status": "running",
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "seed": SEED,
        "inputs": {
            "l69_feature_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
            "l49_fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
            "image_root": str(IMAGE_ROOT),
            "model_dir": str(ROOT / "models/LocateAnything-3B"),
            "manifest": str(MANIFEST_PATH),
            "manifest_sha256_expected": MANIFEST_SHA256,
        },
        "outputs": {"directory": str(out)},
        "labels_read_for_feature_construction": False,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "raw_dense_cache_written": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "token_span_alignment": "UNALIGNED",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "status.json", base)
    try:
        if Path.cwd().resolve() != ROOT.resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
            raise AssertionError("fixed manifest SHA mismatch")
        torch.manual_seed(SEED)
        if not torch.cuda.is_available():
            raise RuntimeError("L75 API audit requires GPU0 CUDA")
        device = "cuda:0"
        splits = load_splits()
        units = select_label_free_units(splits)
        model, processor, tokenizer, runtime = __import__(
            "locatemot.rmot.l75_runtime", fromlist=["load_locateanything"]
        ).load_locateanything(device)
        matcher = CandidateMarkedVLMMatcher(marker_std=0.01).to(device).eval()
        model_base_params = sum(p.numel() for p in model.parameters())
        examples = []
        candidate_a_b = None
        control_example = None
        max_peak = 0
        visual_forward_count = 0
        for index, unit in enumerate(units):
            video = str(unit["video"])
            bank = L75Bank(video)
            try:
                record = make_record(unit, bank, include_labels=False)
                rows = record["row_offsets"]
                image_path = IMAGE_ROOT / video / f"{int(unit['frame_id']):06d}.png"
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                if record["image_size_declared"] and record["image_size_declared"] != [image.width, image.height]:
                    raise AssertionError(f"image size mismatch {record['unit_key']}")
                boxes = bank.tensors["box"].index_select(
                    0, torch.as_tensor(rows, dtype=torch.long)
                ).float().tolist()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                prepared = prepare_visual(model, processor, tokenizer, image,
                                          record["sentence"], boxes)
                visual_forward_count += 1
                if len(prepared["candidate_cells"]) != len(rows):
                    raise AssertionError("candidate cell count drift")
                prepared["image_token_index"] = int(runtime["image_token_index"])
                nonempty = [i for i, cells in enumerate(prepared["candidate_cells"])
                            if cells]
                if len(nonempty) < 2:
                    raise AssertionError("need two nonempty candidate mappings for marker audit")
                first = nonempty[0]
                second = next((i for i in nonempty[1:]
                               if prepared["candidate_cells"][i] != prepared["candidate_cells"][first]), None)
                if second is None:
                    second = nonempty[1]
                selected_cells = [prepared["candidate_cells"][first], prepared["candidate_cells"][second]]
                base_visual = prepared["base_visual"].to(device=device)
                marked, marked_mask = marked_visual_batch(base_visual, selected_cells,
                                                          matcher.region_marker)
                regions, region_mask = __import__(
                    "locatemot.rmot.l75_runtime", fromlist=["region_value_batch"]
                ).region_value_batch(marked, selected_cells)
                hidden = language_forward(model, prepared, marked, inference=True)
                output = matcher(
                    hidden, prepared["expression_positions"], regions, region_mask,
                    return_audit=True,
                )
                hidden_delta = vector_rel_l2(hidden[0], hidden[1])
                region_delta = vector_rel_l2(output["region_state"][0], output["region_state"][1])
                matcher_delta = vector_rel_l2(output["query_state"][0], output["query_state"][1])
                if not bool(torch.isfinite(hidden.float()).all()):
                    raise AssertionError("marker hidden nonfinite")
                if not bool(torch.isfinite(output["match_logit"].float()).all()):
                    raise AssertionError("marker matcher output nonfinite")
                if candidate_a_b is None:
                    candidate_a_b = {
                        "unit_key": record["unit_key"],
                        "candidate_row_indices": [first, second],
                        "candidate_cells": selected_cells,
                        "candidate_cell_mask_shape": list(marked_mask.shape),
                        "final_expression_hidden_shape": list(hidden[:, prepared["expression_positions"], :].shape),
                        "final_expression_hidden_relative_l2": hidden_delta,
                        "region_state_relative_l2": region_delta,
                        "matcher_query_state_relative_l2": matcher_delta,
                        "marker_l2": float(matcher.region_marker.detach().norm()),
                        "one_visual_forward_reused": True,
                    }
                control, target_count, control_length_ok = equal_length_control(
                    tokenizer, record["sentence"]
                )
                control_prepared = text_only_prepared(
                    processor, tokenizer, image, control, prepared["base_visual"], prepared
                )
                control_marked, _ = marked_visual_batch(
                    prepared["base_visual"].to(device=device), [selected_cells[0]],
                    matcher.region_marker
                )
                control_hidden = language_forward(model, control_prepared, control_marked, inference=True)
                original_expr = hidden[0, prepared["expression_positions"], :]
                control_expr = control_hidden[0, control_prepared["expression_positions"], :]
                # Tokenization can place an equal-length control in a
                # different prompt span (the fallback may therefore have a
                # different count).  Compare the registered masked means,
                # not incompatible per-token matrices.
                control_rel = vector_rel_l2(original_expr.mean(dim=0), control_expr.mean(dim=0))
                if control_example is None:
                    control_example = {
                        "unit_key": record["unit_key"],
                        "original_expression_sha256": hashlib.sha256(record["sentence"].encode()).hexdigest(),
                        "control_expression": control,
                        "control_expression_sha256": hashlib.sha256(control.encode()).hexdigest(),
                        "equal_token_count": bool(control_length_ok),
                        "token_count": int(target_count),
                        "expression_hidden_relative_l2": control_rel,
                        "candidate_match_logits_original": [float(v) for v in output["match_logit"].detach().cpu()],
                    }
                elapsed = time.perf_counter() - t0
                max_peak = max(max_peak, int(torch.cuda.max_memory_allocated()))
                examples.append({
                    "unit_key": record["unit_key"],
                    "dataset": record["dataset"], "video": record["video"],
                    "frame_id": record["frame_id"],
                    "candidate_count": len(rows),
                    "row_key_count": len(record["row_keys"]),
                    "row_keys_ordered": record["row_keys"] == sorted(record["row_keys"], key=lambda k: k[-1]),
                    "duplicate_candidate_index_count": len(record["duplicate_candidate_index"]),
                    "nonempty_mapping_count": sum(bool(x) for x in prepared["candidate_cells"]),
                    "image_token_count": len(prepared["image_positions"]),
                    "expression_token_count": len(prepared["expression_positions"]),
                    "expression_span_method": prepared["expression_span_method"],
                    "projected_visual_shape": prepared["projected_visual_shape"],
                    "projected_visual_finite": prepared["projected_visual_finite"],
                    "marker_hidden_relative_l2": hidden_delta,
                    "marker_region_state_relative_l2": region_delta,
                    "control_expression_hidden_relative_l2": control_rel,
                    "match_logit_finite": bool(torch.isfinite(output["match_logit"].float()).all()),
                    "elapsed_seconds": elapsed,
                    "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                })
                del control_hidden, hidden, marked, regions, output, prepared, image
                torch.cuda.empty_cache()
            finally:
                bank.close()
        contract = {
            **base,
            "status": "complete",
            "runtime": runtime,
            "model_base_parameter_count": model_base_params,
            "units": examples,
            "unit_count": len(examples),
            "visual_forward_count": visual_forward_count,
            "candidate_marker_contract": candidate_a_b,
            "unrelated_expression_control": control_example,
            "tiny_lattice_fixture": tiny_lattice_fixture(),
            "detector_parameters_requires_grad": any(p.requires_grad for p in model.parameters()),
            "detector_gradients_expected": "none; inference-only audit",
            "peak_cuda_bytes": max_peak,
            "wall_time_seconds": time.perf_counter() - started,
            "next_action": "run independent fit-only forward/loss contract; only then authorize B0 smoke",
        }
        if not candidate_a_b or candidate_a_b["final_expression_hidden_relative_l2"] <= 0.0:
            raise AssertionError("candidate A/B marker did not change expression hidden")
        if not control_example or control_example["expression_hidden_relative_l2"] <= 0.0:
            raise AssertionError("unrelated equal-length expression did not change hidden")
        write_json(out / "contract.json", contract)
        write_json(out / "provenance.json", {
            **base, "status": "complete", "runtime": runtime,
            "model_manifest": model_file_manifest(),
            "model_file_manifest_sha256": runtime["model_manifest"]["manifest_sha256"],
            "input_unit_keys": [row["unit_key"] for row in examples],
            "labels_read_for_feature_construction": False,
            "no_persistent_feature_cache": True,
        })
        write_json(out / "status.json", contract)
        return 0
    except Exception as exc:
        failure = {
            **base, "status": "incomplete",
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
            "next_action": "preserve this attempt; fix only the first API/marker contract error and retry once in a new directory",
        }
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\n"
            f"First actionable root cause: `{failure['failure_root_cause']}`\n\n"
            "No zero-vector or remote-weight fallback was used; this directory is preserved.\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
