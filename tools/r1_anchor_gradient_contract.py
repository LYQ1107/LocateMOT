#!/usr/bin/env python3
"""Small real-cache anchor/reload/gradient contract for R1.

It uses the one/two-frame targeted cache only.  Feature assembly is completed
before the fit label line and L69 candidate-GT sidecar are opened.  No
detector is constructed here; the frozen L89E anchor is the only model used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r1_aligned_track_conditioning import (  # noqa: E402
    FrozenL89EAnchor, R1AlignedTrackConditioning, R1Config, target_bag_loss,
)
from locatemot.rmot.r1_aligned_cache import R1AlignedCacheIndex, R1FeatureAssembler, write_json  # noqa: E402
from tools.r1_common import (  # noqa: E402
    CATEGORIES, FIT_ROOTS, SEED, THREAD, check_manifest, label_from_index, unit_key,
)

ANCHOR = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_input(anchor: FrozenL89EAnchor, model: torch.nn.Module, frame: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    current = frame.history_observations[:, -1]
    with torch.no_grad():
        base = anchor(frame.z1.unsqueeze(0), frame.text_tokens.unsqueeze(0), frame.text_mask.unsqueeze(0),
                      frame.text_global.unsqueeze(0), frame.frame_global.unsqueeze(0), current,
                      frame.history_observations, frame.history_mask, frame.history_frame_ids, frame.frame_id)
    geometry_mask = torch.ones(frame.geometry.shape[:2], dtype=torch.bool)
    output = model(
        frame.z0.unsqueeze(0), frame.z1.unsqueeze(0), frame.z4.unsqueeze(0),
        frame.text_tokens.unsqueeze(0), frame.text_mask.unsqueeze(0), frame.text_global.unsqueeze(0),
        frame.raw_visual_tokens.unsqueeze(0), frame.geometry, geometry_mask, frame.boxes_norm,
        base["candidate_energy"], base["presence_logit"], base["null_logit"],
    )
    return output, base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/r1/cache/contract_attempt3"))
    parser.add_argument("--request-manifest", type=Path, default=Path("outputs/r1/request_manifest_attempt1/request_manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cache_root = (args.cache if args.cache.is_absolute() else WORK_ROOT / args.cache).resolve()
    request_path = (args.request_manifest if args.request_manifest.is_absolute() else WORK_ROOT / args.request_manifest).resolve()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 anchor contract output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request_records = {str(key): dict(value) for dataset in ("refer_kitti_v1", "refer_kitti_v2")
                       for key, value in request["unique_query_records"][dataset].items()}
    aligned = R1AlignedCacheIndex(cache_root)
    entries = sorted(aligned.entries)
    selected: list[dict[str, Any]] = []
    for category in CATEGORIES:
        for dataset, video, query_id, frame_id in entries:
            key = f"{dataset}|{video}|{query_id}|{frame_id}"
            if str(request_records[key].get("category")) == category:
                selected.append({key: value for key, value in request_records[key].items() if key != "category"} | {"category": category})
                break
    if len(selected) < 2:
        raise AssertionError("targeted cache does not cover positive/inactive contract samples")
    torch.manual_seed(SEED)
    anchor = FrozenL89EAnchor(ANCHOR)
    if any(parameter.requires_grad for parameter in anchor.parameters()):
        raise AssertionError("R1 anchor is not frozen")
    model = R1AlignedTrackConditioning(R1Config())
    assembler = R1FeatureAssembler(aligned, torch.device("cpu"))
    dense = {}
    for dataset, root in FIT_ROOTS.items():
        from tools.r0_common import DenseIndex
        dense[dataset] = DenseIndex(root)
    label_maps = {dataset: {str(value["unit_key"]): value for value in item.label_records}
                  for dataset, item in dense.items()}
    samples = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    grad_audits = []
    for index, public in enumerate(selected):
        feature_record = {key: value for key, value in public.items() if key != "category"}
        frame = assembler.prepare(feature_record, attach_labels=False)
        # Explicit supervision boundary: no target data was available during
        # the feature assembly above.
        label_record = label_maps[feature_record["dataset"]][unit_key(feature_record)]
        raw_label = label_from_index(dense[feature_record["dataset"]], label_record)
        batch = assembler.store.build_frame(feature_record["dataset"], feature_record["video"],
                                             int(feature_record["query_id"]), int(feature_record["frame_id"]),
                                             str(feature_record["sentence"]))
        frame.supervision = assembler.store.attach_frame_labels(batch, raw_label["target_ids"])
        if str(frame.supervision["category"]) != str(public["category"]):
            raise AssertionError("R1 targeted category drift")
        output, base = sidecar_input(anchor, model, frame)
        output["final_energy"].retain_grad()
        loss, stats = target_bag_loss(output["final_energy"][0], frame.supervision["candidate_gt"],
                                      frame.supervision["target_ids"], frame.supervision["category"],
                                      margin=model.config.set_margin, duplicate_weight=model.config.duplicate_weight)
        reg = output["delta_energy"].square().mean() + 0.25 * output["delta_presence"].square().mean()
        total = loss + model.config.residual_reg_weight * reg
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("R1 targeted loss nonfinite")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grads = [value.grad for value in model.parameters() if value.requires_grad and value.grad is not None]
        nonzero = [value for value in grads if bool(torch.isfinite(value).all()) and float(value.abs().sum()) > 0]
        energy_grad = output["final_energy"].grad[0]
        positive = frame.supervision["labels"]
        negative = ~positive
        grad_audits.append({
            "unit_key": unit_key(feature_record), "category": frame.supervision["category"],
            "candidate_count": frame.candidate_count, "positive_count": int(positive.sum()),
            "negative_count": int(negative.sum()), "loss": float(total.detach()),
            "trainable_parameter_tensors": len(grads), "nonzero_gradient_tensors": len(nonzero),
            "positive_energy_gradient_nonzero": bool(positive.any() and (energy_grad[positive].abs() > 0).any()),
            "negative_energy_gradient_nonzero": bool(negative.any() and (energy_grad[negative].abs() > 0).any()),
            "finite": bool(torch.isfinite(total) and torch.isfinite(energy_grad).all()),
            "row_keys": [list(value) for value in frame.row_keys],
        })
        samples.append((frame, frame.supervision, base))
        optimizer.step()
        del output, total, loss, reg
    # A second pass after the zero-initialized residual heads have moved checks
    # that upstream attention/geometry/set modules receive real gradients too.
    optimizer.zero_grad(set_to_none=True)
    frame, supervision, _base = samples[0]
    output, _ = sidecar_input(anchor, model, frame)
    loss, _ = target_bag_loss(output["final_energy"][0], supervision["candidate_gt"], supervision["target_ids"],
                              supervision["category"], margin=model.config.set_margin, duplicate_weight=model.config.duplicate_weight)
    loss.backward()
    second_grads = [value.grad for value in model.parameters() if value.requires_grad and value.grad is not None]
    upstream_nonzero = sum(bool(torch.isfinite(value).all()) and float(value.abs().sum()) > 0 for value in second_grads)
    checkpoint = out / "contract_checkpoint.pt"
    torch.save({"format": "locatemot-r1-sidecar-contract-checkpoint-v1", "model_config": R1Config().__dict__,
                "model_state_dict": model.state_dict(), "anchor_sha256": sha256_file(ANCHOR)}, checkpoint)
    reloaded = R1AlignedTrackConditioning(R1Config())
    result = reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"R1 strict sidecar reload failed: {result}")
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        reference_output, _ = sidecar_input(anchor, model, frame)
        reloaded_output, _ = sidecar_input(anchor, reloaded, frame)
    max_diff = float((reloaded_output["final_energy"] - reference_output["final_energy"]).abs().max())
    if max_diff > 1e-5:
        raise AssertionError(f"R1 reload output drift {max_diff}")
    payload = {
        "format": "locatemot-r1-anchor-gradient-contract-v1", "status": "complete", "command": " ".join([str(sys.executable), *sys.argv]),
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED, "manifest_sha256": check_manifest(),
        "cache": str(cache_root), "anchor": {"path": str(ANCHOR), "sha256": sha256_file(ANCHOR), "frozen": True},
        "samples": grad_audits, "upstream_nonzero_gradient_tensors_after_update": int(upstream_nonzero),
        "strict_reload": True, "reload_max_output_diff": max_diff, "checkpoint": str(checkpoint),
        "anchor_parameters_in_optimizer": False, "detector_parameters_in_optimizer": False,
        "labels_attached_after_feature_construction": True, "persistent_raw_dense_cache": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "elapsed_seconds": float(time.perf_counter() - started), "failure_root_cause": None,
        "next_action": "run the registered <=4-GPU wiring smoke after the complete aligned cache is available",
    }
    write_json(out / "contract.json", payload)
    write_json(out / "provenance.json", payload | {"outputs": {"contract": str(out / "contract.json")},
                                                    "screening_gt_used": False, "official_test_labels_read": False,
                                                    "ordinary_mot_ovmot_touched": False})
    write_json(out / "status.json", {"format": payload["format"], "status": "complete", "command": payload["command"],
                                      "inputs": {"cache": str(cache_root), "anchor": str(ANCHOR)},
                                      "outputs": {"contract": str(out / "contract.json")}, "failure_root_cause": None,
                                      "next_action": payload["next_action"]})
    print(json.dumps({"status": "complete", "samples": len(samples), "upstream_nonzero": int(upstream_nonzero),
                      "reload_max_diff": max_diff}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
