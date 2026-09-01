"""Validate Stage L23 dense-bank provenance, alignment, finiteness and determinism."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.build_l23_candidate_bank_v3 import (  # noqa: E402
    dense_clip_map, fixed_point_set, grid_sample_points, region_points, sha256_file,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--old-bank-root", default="outputs/l19/dual_banks_features")
    ap.add_argument("--v2-bank-root", default="outputs/l22/candidate_bank_v2")
    ap.add_argument("--v3-root", default="outputs/l23/candidate_bank_v3")
    ap.add_argument("--raw-root", default="data/kitti_tracking_training/image_02")
    ap.add_argument("--out-root", default="outputs/l23/audit/candidate_bank_v3_validation")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    def path(value: str) -> Path:
        value = Path(value)
        return value if value.is_absolute() else ROOT / value
    manifest, old_root, v2_root, v3_root, raw_root, out_root = map(path, (
        args.manifest, args.old_bank_root, args.v2_bank_root, args.v3_root,
        args.raw_root, args.out_root))
    if out_root.exists():
        raise FileExistsError(out_root)
    out_root.mkdir(parents=True, exist_ok=False)
    data = json.loads(manifest.read_text())
    queries = data["queries"]
    if len(queries) != 160:
        raise AssertionError(f"manifest query count {len(queries)}")
    report = {"format": "locatemot-l23-candidate-bank-v3-validation-v1",
              "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
              "v3_root": str(v3_root), "raw_root": str(raw_root),
              "gt_used_for_feature_construction": False, "videos": {},
              "checks": {}}
    totals = {"rows": 0, "frames": 0, "dense_maps": 0}
    for video in sorted({str(q["video"]) for q in queries}):
        old_path = old_root / "kitti" / f"{video}.pt"
        v2_path = v2_root / "kitti" / f"{video}.pt"
        v3_path = v3_root / "kitti" / f"{video}.pt"
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        v2 = torch.load(v2_path, map_location="cpu", weights_only=False)
        v3 = torch.load(v3_path, map_location="cpu", weights_only=False)
        old_t, v2_t, v3_t = old["tensors"], v2["tensors"], v3["tensors"]
        old_labels = json.loads(old_path.with_suffix(".labels.json").read_text())["candidate_gt"]
        v2_labels = json.loads(v2_path.with_suffix(".labels.json").read_text())["candidate_gt"]
        v3_labels = json.loads(v3_path.with_suffix(".labels.json").read_text())["candidate_gt"]
        n, frames = len(v3_t["track_id"]), len(v3_t["frame_ids"])
        if old_labels != v2_labels or v2_labels != v3_labels:
            raise AssertionError(f"sidecar mismatch {video}")
        for key in ("frame_ptr", "frame_ids", "frame", "box", "pool_id", "track_id", "candidate_index"):
            if not torch.equal(old_t[key], v2_t[key]) or not torch.equal(v2_t[key], v3_t[key]):
                raise AssertionError(f"alignment mismatch {video}/{key}")
        if v3["row_keys"] != v2.get("row_keys"):
            raise AssertionError(f"row key mismatch {video}")
        expected_frame_index = torch.repeat_interleave(torch.arange(frames, dtype=torch.int64), torch.diff(v3_t["frame_ptr"]))
        if not torch.equal(v3_t["dense_v2_row_index"], torch.arange(n, dtype=torch.int64)):
            raise AssertionError(f"row alignment index mismatch {video}")
        if not torch.equal(v3_t["dense_map_frame_index"], expected_frame_index):
            raise AssertionError(f"dense map frame index mismatch {video}")
        finite = {}
        for key in ("dense_roi", "dense_points", "dense_context_1p5", "dense_context_3", "dense_prev_roi"):
            if key not in v3_t or len(v3_t[key]) != n:
                raise AssertionError(f"missing/short dense field {video}/{key}")
            finite[key] = bool(torch.isfinite(v3_t[key].float()).all())
            if not finite[key]:
                raise AssertionError(f"nonfinite {video}/{key}")
        map_paths = sorted((v3_root / "dense_maps").glob(f"{video}_*.pt"))
        if len(map_paths) != frames:
            raise AssertionError(f"map count {video}: {len(map_paths)} != {frames}")
        for map_path in map_paths:
            fmap = torch.load(map_path, map_location="cpu", weights_only=False)["feature_map"]
            if list(fmap.shape) != [1, 512, 7, 7] or not bool(torch.isfinite(fmap.float()).all()):
                raise AssertionError(f"invalid dense map {map_path}")
        # Recompute one fixed raw frame and one fixed candidate with the same
        # frozen model/rules. This is a determinism check, not model selection.
        device = torch.device(args.device)
        import clip
        model, _ = clip.load("ViT-B/32", device=device); model.eval()
        fi = 0; frame = int(v3_t["frame_ids"][fi]); begin = int(v3_t["frame_ptr"][fi])
        image_path = raw_root / video / f"{frame:06d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(image_path)
        h, w = image.shape[:2]
        fmap = dense_clip_map(model, image, device)
        saved_map = torch.load(v3_root / "dense_maps" / f"{video}_{frame:06d}.pt",
                               map_location="cpu", weights_only=False)["feature_map"].float()
        map_delta = float((fmap.cpu().float() - saved_map).abs().max())
        box = v3_t["box"][begin].numpy().astype(np.float32)
        roi = grid_sample_points(fmap, region_points(box, w, h, 1.0)).mean(axis=0)
        point = grid_sample_points(fmap, fixed_point_set(box, w, h))
        roi_delta = float(np.max(np.abs(roi - v3_t["dense_roi"][begin].float().numpy())))
        point_delta = float(np.max(np.abs(point - v3_t["dense_points"][begin].float().numpy())))
        if map_delta > 1e-3 or roi_delta > 2e-3 or point_delta > 2e-3:
            raise AssertionError(f"nondeterministic sample {video}: {map_delta}, {roi_delta}, {point_delta}")
        report["videos"][video] = {"rows": n, "frames": frames, "dense_maps": len(map_paths),
                                    "finite": finite, "map_delta_max": map_delta,
                                    "roi_delta_max": roi_delta, "points_delta_max": point_delta,
                                    "old_v2_v3_alignment": True, "labels_unchanged": True}
        totals["rows"] += n; totals["frames"] += frames; totals["dense_maps"] += len(map_paths)
        del model, old, v2, v3
    report["totals"] = totals
    prior = json.loads((ROOT / "outputs/l23/audit/candidate_bank_v2_readonly_corrected/audit.json").read_text())
    report["coverage_inherited_from_readonly_audit"] = prior["coverage"]
    report["coverage_check"] = {"rate": prior["coverage"]["coverage_rate"],
                                 "minimum": 0.9848662599718442 - 0.005,
                                 "passed": prior["coverage"]["coverage_rate"] >= 0.9848662599718442 - 0.005}
    report["checks"] = {"manifest_160_queries": True, "old_v2_v3_alignment": True,
                        "labels_unchanged": True, "finite_dense_features": True,
                        "dense_map_shape_and_count": True, "same_frame_bbox_deterministic": True,
                        "coverage_threshold": report["coverage_check"]["passed"]}
    if not all(report["checks"].values()):
        raise AssertionError(report["checks"])
    (out_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_root / "validation.md").write_text("# Stage L23 candidate bank v3 validation\n\n" +
        "All old/v2/frame/label alignment, finite dense features, map count/shape, coverage, and deterministic raw-frame/bbox resampling checks passed.\n\n" +
        f"- Rows: `{totals['rows']}`; frames/maps: `{totals['frames']}`/`{totals['dense_maps']}`\n" +
        f"- Inherited IoU>=0.5 coverage: `{prior['coverage']['coverage_rate']:.8f}`\n" +
        "- GT was not used for dense feature construction.\n")
    print(json.dumps({"validation": str(out_root / "validation.json"), "checks": report["checks"], "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
