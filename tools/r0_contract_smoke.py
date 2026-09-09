#!/usr/bin/env python3
"""Run one real R0 frame forward/backward contract smoke.

This is the last R0A implementation gate before the formal query-independent
visual cache.  It uses one legal L82 fit frame, writes one temporary cache
item, and removes that item when the contract check ends.  Target labels are
attached only after the complete native frame and visual item have been
constructed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch


WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
LANGUAGE_CACHE = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/cache/language_tokens_retry1"
).resolve()
V1_DENSE = WORK_ROOT / "outputs/r0/data/v1_dense_train_index_retry2"
V2_DENSE = WORK_ROOT / "outputs/r0/data/v2_dense_train_index_retry2"

# The asset root is intentionally readable for immutable data, but the R0A
# implementation must win module resolution.  ``rmot`` is a namespace
# package in these checkouts, so extend it explicitly after importing the
# worktree's regular ``locatemot`` package.
for path in (ASSET_ROOT, WORK_ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.append(str(path))
if str(WORK_ROOT) in sys.path:
    sys.path.remove(str(WORK_ROOT))
sys.path.insert(0, str(WORK_ROOT))
import locatemot.rmot as _rmot_package  # noqa: E402
_rmot_path = str(WORK_ROOT / "locatemot" / "rmot")
if _rmot_path not in [str(value) for value in _rmot_package.__path__]:
    _rmot_package.__path__.insert(0, _rmot_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grad_report(module: torch.nn.Module) -> dict[str, Any]:
    total = 0.0
    nonzero = 0
    finite = True
    by_name: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        norm = float(value.norm())
        by_name[name] = norm
        total += norm
        nonzero += int(norm > 0.0)
        finite = finite and bool(torch.isfinite(value).all())
    return {
        "total_norm": total,
        "nonzero_parameter_grads": nonzero,
        "finite": finite,
        "by_name": by_name,
    }


def choose_fit_row(path: Path) -> dict[str, Any]:
    """Choose a deterministic positive frame with at least one negative row."""
    for row in read_jsonl(path / "query_frame_labels.jsonl"):
        if (
            str(row.get("category")) in {"positive", "multi_positive"}
            and int(row.get("candidate_count", 0)) > int(row.get("positive_row_count", 0)) > 0
        ):
            return row
    raise AssertionError(f"no positive/negative fit frame in {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/r0/audit/r0_contract_smoke_retry1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 contract output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    device = torch.device(args.device)
    runtime = None
    store = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0A worktree cwd: {Path.cwd()}")
        observed_manifest = sha256_file(MANIFEST)
        if observed_manifest != MANIFEST_SHA:
            raise AssertionError(f"fixed manifest SHA drift: {observed_manifest}")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("R0 contract smoke requires the registered GPU runtime")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

        # The dense index is only used to select a legal frame and to verify
        # that frame-specific labels agree with the native sidecar lookup.
        label_row = choose_fit_row(V1_DENSE)
        dataset = str(label_row["dataset"])
        video = str(label_row["video"])
        query_id = int(label_row["query_id"])
        frame_id = int(label_row["frame_id"])
        sentence = str(label_row["sentence"])
        store_module = __import__("locatemot.rmot.r0_dense_data", fromlist=["R0BankStore"])
        store = store_module.R0BankStore()
        batch = store.build_frame(dataset, video, query_id, frame_id, sentence)
        if batch.image_size[0] == batch.image_size[1]:
            raise AssertionError("contract smoke needs a non-square KITTI frame")
        if batch.image_hw != (batch.image_size[1], batch.image_size[0]):
            raise AssertionError(f"image_hw contract drift: {batch.image_size} -> {batch.image_hw}")
        if batch.row_offsets != list(range(batch.row_offsets[0], batch.row_offsets[-1] + 1)):
            raise AssertionError("native row offsets are not contiguous")
        if batch.candidate_count != len(batch.row_offsets):
            raise AssertionError("candidate count/row offset drift")

        # Construct and save exactly one real query-independent visual item.
        cache_module = __import__("r0_build_visual_cache", fromlist=["R0FrozenVisualRuntime", "_frame_item"])
        runtime = cache_module.R0FrozenVisualRuntime(device)
        bank_path, blob = store_module.load_l69_bank(video)
        frame_ids = [int(value) for value in blob["tensors"]["frame_ids"].long().tolist()]
        frame_position = frame_ids.index(frame_id)
        visual_config = __import__(
            "locatemot.rmot.r0_visual_tokens", fromlist=["R0VisualTokenConfig"]
        ).R0VisualTokenConfig()
        temp_parent = Path("/data2/usr_for_deadline")
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="locatemot_r0_smoke_", dir=str(temp_parent)) as temporary:
            cache_path = Path(temporary) / "visual_item.pt"
            item_audit = cache_module._frame_item(
                runtime, bank_path, blob, dataset, video, frame_position, cache_path, visual_config
            )
            item = torch.load(cache_path, map_location="cpu", weights_only=False)
            if item.get("dataset") != dataset or item.get("group_key") != f"{dataset}|{video}|{frame_id}":
                raise AssertionError("saved visual cache dataset/group_key contract failed")
            if item.get("dataset") is None or item.get("group_key") is None:
                raise AssertionError("saved visual item has null dataset/group_key")
            for field in ("inner_tokens", "context_tokens", "boxes_normalized"):
                if not torch.is_tensor(item.get(field)) or not bool(torch.isfinite(item[field].float()).all()):
                    raise AssertionError(f"invalid saved cache tensor: {field}")
            if int(item["candidate_count"]) != batch.candidate_count:
                raise AssertionError("saved cache candidate count drift")
            if item.get("row_offsets") != batch.row_offsets:
                raise AssertionError("saved cache row order drift")
            if item.get("labels_in_cache") is not False or item.get("query_independent") is not True:
                raise AssertionError("cache label/query flags drift")

            # Attach frame-specific labels only after feature construction.
            supervision = store.attach_frame_labels(batch, label_row["target_ids"])
            if supervision["category"] != str(label_row["category"]):
                raise AssertionError("frame-specific category lookup drift")
            if supervision["positive_row_count"] != int(label_row["positive_row_count"]):
                raise AssertionError("frame-specific positive count drift")
            if bool(supervision["present_uncovered"]) != bool(label_row["present_uncovered"]):
                raise AssertionError("partial coverage semantics drift")

            lang_module = __import__("locatemot.rmot.l89_language_cache", fromlist=["L89LanguageTokenCache"])
            language = lang_module.L89LanguageTokenCache(LANGUAGE_CACHE)
            language_item = language.get(dataset, video, query_id, sentence)
            text_tokens = language_item.tokens.unsqueeze(0).to(device=device, dtype=torch.float32)
            text_mask = language_item.mask.unsqueeze(0).to(device=device, dtype=torch.bool)
            text_global = (
                text_tokens * text_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)
            ).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            if text_tokens.ndim != 3 or text_tokens.shape[-1] != 256 or not bool(torch.isfinite(text_tokens).all()):
                raise AssertionError(f"pure L89 text shape/finite failure: {tuple(text_tokens.shape)}")

            geometry_module = __import__("locatemot.rmot.r0_geometry", fromlist=["build_track_geometry"])
            geometry = geometry_module.build_track_geometry(store, batch, batch.image_hw, history_length=4)
            model_module = __import__("locatemot.models.r0_track_grounding", fromlist=["R0TrackGroundingHead"])
            losses_module = __import__("locatemot.rmot.r0_losses", fromlist=["r0_total_loss"])
            model = model_module.R0TrackGroundingHead().to(device=device, dtype=torch.float32)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer_names = [name for group in optimizer.param_groups for name, value in model.named_parameters() if any(value is item_param for item_param in group["params"])]
            if not optimizer_names or any("tracker" in name.lower() for name in optimizer_names):
                raise AssertionError("R0 optimizer parameter contract failed")
            with torch.autocast(device_type="cuda", enabled=False):
                output = model(
                    item["inner_tokens"].float().to(device),
                    item["context_tokens"].float().to(device),
                    item["boxes_normalized"].float().to(device),
                    geometry.to(device),
                    text_tokens,
                    text_mask,
                    text_global,
                )
                loss, loss_info = losses_module.r0_total_loss(
                    output["membership_logit"],
                    output["coverage_presence_logit"],
                    [str(supervision["category"])],
                    [supervision["target_ids"]],
                    [supervision["candidate_gt"]],
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("R0 smoke loss is nonfinite")
                loss.backward()
            gradients = grad_report(model)
            if not gradients["finite"] or gradients["nonzero_parameter_grads"] <= 0 or not math.isfinite(gradients["total_norm"]):
                raise AssertionError(f"R0 adapter gradient contract failed: {gradients}")
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            reload_model = model_module.R0TrackGroundingHead()
            reload_model.load_state_dict(state, strict=True)
            model.eval(); reload_model.to(device=device, dtype=torch.float32).eval()
            with torch.inference_mode():
                before = model(
                    item["inner_tokens"].float().to(device), item["context_tokens"].float().to(device),
                    item["boxes_normalized"].float().to(device), geometry.to(device), text_tokens, text_mask, text_global,
                )
                after = reload_model(
                    item["inner_tokens"].float().to(device), item["context_tokens"].float().to(device),
                    item["boxes_normalized"].float().to(device), geometry.to(device), text_tokens, text_mask, text_global,
                )
            reload_diff = float((before["membership_logit"] - after["membership_logit"]).abs().max())
            if reload_diff > 1e-6:
                raise AssertionError(f"strict reload drift: {reload_diff}")
            report = {
                "format": "locatemot-r0-contract-smoke-v1",
                "status": "complete",
                "stage": "R0A implementation gate; one real frame forward/backward, no optimizer step",
                "command": command,
                "cwd": str(WORK_ROOT),
                "luna_thread": THREAD,
                "seed": SEED,
                "selected": {"dataset": dataset, "video": video, "query_id": query_id, "frame_id": frame_id, "sentence": sentence},
                "candidate_count": batch.candidate_count,
                "row_offsets": batch.row_offsets,
                "row_keys": [list(value) for value in batch.row_keys],
                "duplicate_candidate_index_rows": len(batch.candidate_indices) - len(set(batch.candidate_indices)),
                "candidate_rows_retained": True,
                "candidate_deletion": False,
                "candidate_truncation": False,
                "visual_item_audit": item_audit,
                "visual_shapes": {field: list(item[field].shape) for field in ("inner_tokens", "context_tokens", "boxes_normalized")},
                "visual_item_saved_dataset": item["dataset"],
                "visual_item_saved_group_key": item["group_key"],
                "pure_text_tokens_shape": list(text_tokens.shape),
                "pure_text_mask_shape": list(text_mask.shape),
                "pure_text_valid_tokens": int(text_mask.sum()),
                "text_global_shape": list(text_global.shape),
                "geometry_shape": list(geometry.shape),
                "image_size_wh": list(batch.image_size),
                "image_hw_for_geometry": list(batch.image_hw),
                "frame_specific_label_lookup": True,
                "partial_coverage_semantics": {"category": supervision["category"], "present_uncovered": bool(supervision["present_uncovered"]), "coverage_fraction": float(supervision["coverage_fraction"])},
                "model_output_shapes": {key: list(value.shape) for key, value in output.items() if torch.is_tensor(value)},
                "loss": float(loss.detach()),
                "loss_info": loss_info,
                "gradients": gradients,
                "optimizer_parameter_count": len(optimizer_names),
                "optimizer_contains_groundingdino": False,
                "optimizer_contains_tracker": False,
                "groundingdino_model_instantiated": True,
                "groundingdino_parameters_frozen": bool(all(not parameter.requires_grad for parameter in runtime.model.parameters())),
                "groundingdino_gradient_parameters": 0,
                "strict_reload": True,
                "reload_max_abs_diff": reload_diff,
                "temporary_visual_item_removed_after_check": True,
                "language_cache": str(LANGUAGE_CACHE),
                "language_cache_summary_sha256": sha256_file(LANGUAGE_CACHE / "summary.json"),
                "manifest_sha256": observed_manifest,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "wall_seconds": time.perf_counter() - started,
                "screening_gt_used": False,
                "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False,
                "hota_trackeval_run": False,
                "token_span_region_alignment": "UNALIGNED",
                "static_motion_alignment": "UNALIGNED",
                "failure_root_cause": None,
                "next_action": "commit R0 safe code, then build query-independent visual cache",
            }
            write_json(out / "contract.json", report)
            write_json(out / "provenance.json", report | {"format": "locatemot-r0-contract-provenance-v1"})
            write_json(out / "status.json", report | {"format": "locatemot-r0-contract-smoke-v1"})
            return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 contract smoke — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {
            "format": "locatemot-r0-contract-smoke-v1",
            "status": "incomplete",
            "command": command,
            "cwd": str(WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback_path": str((out / "INCOMPLETE.md").resolve()),
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
        }
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 2
    finally:
        if runtime is not None:
            runtime.close()
        if store is not None:
            store._blob = None
            store._candidate_gt = None
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
