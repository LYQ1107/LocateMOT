#!/usr/bin/env python3
"""Execute official TETA on localization/classification/identity fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locatemot.unified.runtime.u2_teta_evaluation import convert_tao_v1, evaluate_tao_teta, write_ovmot_predictions


def main() -> int:
    vocabulary = json.loads((ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json").read_text())["categories"]
    base = next(category for category in vocabulary if category["frequency"] in {"f", "c"})
    novel = next(category for category in vocabulary if category["frequency"] == "r")
    categories = [base, novel]
    video = "val/synthetic"
    images = [{"id": frame, "video_id": 1, "file_name": f"{video}/{frame:06d}.jpg",
               "frame_index": frame, "width": 200, "height": 100} for frame in range(4)]
    annotations = [{"id": 2 * frame + index, "image_id": frame, "video_id": 1, "track_id": index + 1,
                    "category_id": category["id"], "bbox": [40. * index, 0., 20., 50.], "area": 1000., "iscrowd": 0}
                   for frame in range(3) for index, category in enumerate(categories)]
    data = {"videos": [{"id": 1, "name": video, "neg_category_ids": [], "not_exhaustive_category_ids": []}],
            "images": images, "annotations": annotations, "categories": categories,
            "tracks": [{"id": index + 1, "video_id": 1, "category_id": category["id"]} for index, category in enumerate(categories)]}
    frames = [f"{frame:06d}.jpg" for frame in range(4)]
    group = {"video": video, "frames": frames}
    predictions = [{"frame": frame, "track_ids": [0, 1], "boxes": [[0., 0., 20., 50.], [40., 0., 60., 50.]],
                    "scores": [.9, .9], "predicted_class": [base["id"] - 1, novel["id"] - 1]} for frame in frames[:3]]
    predictions.append({"frame": frames[-1], "track_ids": [], "boxes": [], "scores": [], "predicted_class": []})
    image_lookup = {image["file_name"]: image for image in images}
    results = {}
    with tempfile.TemporaryDirectory(prefix="locatemot_u2_teta_") as raw:
        directory = Path(raw)
        gt = directory / "gt.json"
        gt.write_text(json.dumps(data))
        for kind in ("perfect", "wrong_class", "identity_switch", "unlabeled_distant_object"):
            changed = copy.deepcopy(predictions)
            if kind == "wrong_class":
                for frame in changed:
                    frame["predicted_class"].reverse()
            elif kind == "identity_switch":
                changed[-2]["track_ids"].reverse()
            elif kind == "unlabeled_distant_object":
                for frame in changed:
                    frame["track_ids"].append(5)
                    frame["boxes"].append([150., 0., 170., 50.])
                    frame["scores"].append(.9)
                    frame["predicted_class"].append(novel["id"] - 1)
            path = directory / f"{kind}.json"
            contract = write_ovmot_predictions(group, changed, image_lookup, path)
            metric = evaluate_tao_teta(gt, path, directory / kind, kind, workers=1)
            assert metric["videos_evaluated"] == 1 and metric["frames_evaluated"] == 4
            results[kind] = {"metrics": metric["metrics"], "export_frames": contract["frames"]}
        for group in ("overall", "base", "novel"):
            assert all(abs(results["perfect"]["metrics"][group][key] - 100.) < 1e-8
                       for key in ("TETA", "LocA", "AssocA", "ClsA"))
        wrong = results["wrong_class"]["metrics"]["overall"]
        assert wrong["LocA"] == 100 and wrong["AssocA"] == 100 and wrong["ClsA"] == 0
        switched = results["identity_switch"]["metrics"]["overall"]
        assert switched["LocA"] == 100 and switched["ClsA"] == 100 and switched["AssocA"] < 100
        assert results["unlabeled_distant_object"]["metrics"] == results["perfect"]["metrics"], "Unlabeled distant object penalized local-cluster TETA"
    # Explicit category conversion evidence: an unmapped v0.5 unit is reported,
    # while image IDs, timelines, boxes and identities for mapped units survive.
    source = copy.deepcopy(data)
    source["categories"].append({"id": 1231, "name": "unmapped", "synset": "not_in_lvis_v1"})
    source["annotations"].append({**annotations[0], "id": 999, "track_id": 99, "category_id": 1231})
    source["tracks"].append({"id": 99, "video_id": 1, "category_id": 1231})
    view, conversion = convert_tao_v1(source, vocabulary)
    assert conversion["unmapped_gt_units"] == 1 and conversion["unmapped_annotation_ids"] == [999]
    assert len(view["images"]) == 4 and view["annotations"] == annotations
    report = {"format": "locatemot-u2-official-teta-pipeline-contract-v1", "status": "PASS",
              "input_source": "synthetic fixtures through actual pinned official TETA",
              "cases": results, "classification_localization_association_separated": True,
              "unknown_distant_prediction_not_scored_as_false_positive": True,
              "base_novel_and_overall_reported": True, "unmapped_conversion_counts_explicit": True,
              "every_frame_exported_including_empty_outputs": True,
              "benchmark_metrics_measured": False, "formal_joint_optimization_launched": False,
              "official_test_labels_read": False}
    path = ROOT / "outputs/unified/u2/eval/teta_pipeline_contract.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
