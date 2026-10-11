#!/usr/bin/env python3
"""Small native-versus-packed U1 audit for a selected actual U2 artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def small_u1_retention(native_runtime, packed_runtime) -> list[dict]:
    """Deterministic legal integration scope; predictions precede label reads."""
    import torch
    from PIL import Image
    from locatemot.unified.runtime.grounding_inference import run_grounding, prediction_arrays
    from tools.unified.u1_formal_expression_gate import query_records, read_gt, iou, average_precision
    from tools.unified.u1_formal_generic_gate import read_gt as read_generic_gt, VOCABULARY
    # Small deterministic integration scope, separate from all original full
    # U1 references. Predictions run before opening each actual frame label.
    rows = [r for r in query_records() if r["row_kind"] == "positive" and r["target_ids"]]
    retention = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        pair = next((r, donor) for r in rows if r["dataset"] == dataset for donor in rows
                    if donor["dataset"] == dataset and donor["video"] == r["video"]
                    and set(donor["target_ids"]) != set(r["target_ids"]) and donor["sentence"] != r["sentence"])
        row, donor = pair
        results = []
        for prompt in (VOCABULARY, row["sentence"], donor["sentence"]):
            a = prediction_arrays(run_grounding(native_runtime, row["image"], prompt))
            b = prediction_arrays(run_grounding(packed_runtime, row["image"], prompt))
            assert all(torch.equal(x, y) for x, y in zip(a, b)), (dataset, prompt)
            order = torch.argsort(a[1], descending=True)
            results.append(a[0][order].tolist())
        with Image.open(row["image"]) as im:
            width, height = im.size
        label = ROOT / "data" / dataset / "labels_with_ids/image_02" / row["video"] / f"{row['frame']:06d}.txt"
        gt = read_gt(label, width, height)
        targets = [gt[str(t)] for t in row["target_ids"] if str(t) in gt]
        generic_gt = [g["box"] for g in read_generic_gt(label, width, height)]
        assert targets and generic_gt
        recall = lambda predictions, ground_truth, k: sum(max((iou(p, g) for p in predictions[:k]), default=0.) >= .5 for g in ground_truth) / len(ground_truth)
        metrics = {"generic_top150_recall": recall(results[0], generic_gt, 150),
                   "expression_top100_recall": recall(results[1], targets, 100),
                   "true_ap50_top300": average_precision(results[1][:300], [], targets),
                   "shuffled_ap50_top300": average_precision(results[2][:300], [], targets)}
        retention.append({"dataset": dataset, "video": row["video"], "frame": row["frame"],
                          "targets": len(targets), "generic_objects": len(generic_gt),
                          "generic_vocabulary": VOCABULARY,
                          "native_metrics": metrics, "packed_metrics": metrics.copy(),
                          "all_box_score_tensors": "BIT_EXACT"})
        print(json.dumps({"stage": "U1_SMALL_INTEGRATION_PASS", "dataset": dataset}), flush=True)
    return retention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from locatemot.unified.runtime.u2_checkpoint import load_u2_checkpoint
    from locatemot.unified.runtime.grounding_inference import load_grounding_runtime
    from tools.unified.build_u2_candidate_cache import digest, write_json
    torch.set_num_threads(2)
    packed = load_u2_checkpoint(args.checkpoint, device=args.device)
    if packed.reference_only or not packed.metadata["foundation_frozen"]:
        raise ValueError("Formal integration audit requires an actual selected frozen-foundation U2 artifact")
    source = Path(packed.metadata["ordinary_manifest"]["spec"]["checkpoint"])
    if digest(source) != packed.metadata["foundation_sha256"]:
        raise ValueError("Native integration reference differs from the selected foundation")
    native = load_grounding_runtime(checkpoint=source, device=args.device, max_per_img=900)
    if native.missing_keys or native.unexpected_keys:
        raise ValueError("Native integration reference has incomplete model tensors")
    try:
        assert all(torch.equal(value, packed.foundation.state_dict()[name]) for name,value in native.model.state_dict().items()), "Frozen packed foundation tensors changed"
        results = small_u1_retention(native, packed.runtime)
    except AssertionError as error:
        write_json(args.out, {"format": "locatemot-u2-packed-integration-retention-v1", "status": "FAIL",
                             "checkpoint_sha256": digest(args.checkpoint), "foundation_sha256": native.checkpoint_sha256,
                             "reference_only": False, "blocked_status": "LOCATEMOT_U_BLOCKED_U2_FOUNDATION_FORGETTING",
                             "error": str(error), "ready_for_u3": False, "official_test_labels_read": False})
        raise
    report = {"format": "locatemot-u2-packed-integration-retention-v1", "status": "PASS",
              "checkpoint_sha256": digest(args.checkpoint), "foundation_sha256": native.checkpoint_sha256,
              "reference_only": False, "frozen_foundation_all_tensors": "BIT_EXACT", "scope": results,
              "all_predictions": "BIT_EXACT", "original_full_u1_references_modified": False,
              "official_test_labels_read": False, "ready_for_u3": False}
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
