"""Official unmodified TETA and explicit original-TAO LVIS-v1 GT view."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TETA_ROOT = ROOT / "third_party/TETA"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_tao_v1(original: dict, vocabulary: list[dict]) -> tuple[dict, dict]:
    """Reproduce OVTrack's synset conversion, retaining every image/video.

    Unmapped v0.5 units cannot be represented by a v1 textual classifier.
    Their IDs/counts are explicitly reported; the raw audit remains separate.
    """
    data = copy.deepcopy(original)
    source_categories = {category["id"]: category for category in data["categories"]}
    v1 = {category["synset"]: category for category in vocabulary}
    appeared = {annotation["category_id"] for annotation in data["annotations"]}
    mapping = {category: v1[source_categories[category]["synset"]]["id"]
               for category in appeared if source_categories[category]["synset"] in v1}
    excluded_annotations = [annotation["id"] for annotation in data["annotations"] if annotation["category_id"] not in mapping]
    excluded_tracks = [track["id"] for track in data["tracks"] if track["category_id"] not in mapping]
    retained_annotations, retained_tracks = [], []
    instance_counts, image_counts = Counter(), defaultdict(set)
    for annotation in data["annotations"]:
        if annotation["category_id"] in mapping:
            annotation["category_id"] = mapping[annotation["category_id"]]
            retained_annotations.append(annotation)
            instance_counts[annotation["category_id"]] += 1
            image_counts[annotation["category_id"]].add(annotation["image_id"])
    for track in data["tracks"]:
        if track["category_id"] in mapping:
            track["category_id"] = mapping[track["category_id"]]
            retained_tracks.append(track)
    for video in data["videos"]:
        for key in ("neg_category_ids", "not_exhaustive_category_ids"):
            video[key] = [mapping[category] for category in video[key] if category in mapping]
    by_video = {video["id"]: video for video in data["videos"]}
    for image in data["images"]:
        for key in ("neg_category_ids", "not_exhaustive_category_ids"):
            image[key] = by_video[image["video_id"]][key]
    data["annotations"], data["tracks"] = retained_annotations, retained_tracks
    data["categories"] = copy.deepcopy(vocabulary)
    for category in data["categories"]:
        category["instance_count"] = instance_counts[category["id"]]
        category["image_count"] = len(image_counts[category["id"]])
    report = {"source_images": len(original["images"]), "view_images": len(data["images"]),
              "source_videos": len(original["videos"]), "view_videos": len(data["videos"]),
              "source_gt_units": len(original["annotations"]), "view_gt_units": len(retained_annotations),
              "source_tracks": len(original["tracks"]), "view_tracks": len(retained_tracks),
              "unmapped_gt_units": len(excluded_annotations), "unmapped_tracks": len(excluded_tracks),
              "unmapped_annotation_ids": excluded_annotations, "unmapped_track_ids": excluded_tracks,
              "source_to_v1_category_ids": {str(key): value for key, value in sorted(mapping.items())},
              "source_geometry_id_and_timeline_preserved": True,
              "conversion_reference": "https://github.com/SysCV/ovtrack/blob/main/tools/convert_datasets/create_tao_v1.py",
              "raw_v05_localization_audit_scope_unchanged": True}
    return data, report


def prepare_tao_validation_gt(directory: Path) -> dict:
    source = ROOT / "data/TAO-Amodal/annotations/validation.json"
    frozen = json.loads((ROOT / "outputs/unified/u2/perception_cache/supervision_source_inventory.json").read_text())["ovmot"]["tao_val"]
    if digest(source) != frozen["sha256"]:
        raise ValueError("Official validation export cannot change original frozen TAO labels")
    original = json.loads(source.read_text())
    if len(original["images"]) != frozen["source_images"] or any(not video["name"].startswith("val/") for video in original["videos"]):
        raise ValueError("TETA scope must contain exactly frozen full TAO validation")
    vocabulary_path = ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json"
    converted, report = convert_tao_v1(original, json.loads(vocabulary_path.read_text())["categories"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tao_validation_lvis_v1.json"
    path.write_text(json.dumps(converted) + "\n")
    report.update({"format": "locatemot-u2-tao-official-v1-ground-truth-view-v1", "status": "COMPLETE",
                   "source_path": str(source), "source_sha256": frozen["sha256"],
                   "view_path": str(path), "view_sha256": digest(path),
                   "vocabulary_sha256": digest(vocabulary_path), "official_test_labels_read": False,
                   "benchmark_metrics_measured": False})
    (directory / "gt_view.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_teta_source() -> dict:
    report = json.loads((ROOT / "outputs/unified/u2/protocol/teta_source.json").read_text())
    for name, sha in report["files"].items():
        if digest(TETA_ROOT / name) != sha:
            raise ValueError(f"Official TETA source differs: {name}")
    return report


def ovmot_frame_prediction(core, output: dict, candidates: dict, class_text) -> dict:
    """Classify emitted IDs from their updated persistent semantic memories."""
    import torch
    indices = output["candidate_indices"]
    by_id = {track_id: index for index, track_id in enumerate(output["state"].track_ids.tolist())}
    state_rows = [by_id[track_id] for track_id in output["track_ids"].tolist()]
    logits = core.ovmot.class_logits(output["state"].semantic_memory[state_rows], class_text)
    probabilities = torch.softmax(logits, dim=-1)
    class_scores, classes = probabilities.max(dim=-1)
    return {"track_ids": output["track_ids"].tolist(), "boxes": candidates["boxes"][indices].float().tolist(),
            "scores": candidates["scores"][indices].float().tolist(),
            "predicted_class": classes.tolist(), "class_scores": class_scores.float().tolist()}


def write_ovmot_predictions(group: dict, predictions: list[dict], images: dict[str, dict], path: Path) -> dict:
    if [frame["frame"] for frame in predictions] != group["frames"]:
        raise ValueError("OVMOT export must include every registered validation frame, including empty outputs")
    rows = []
    for frame in predictions:
        image = images[f"{group['video']}/{frame['frame']}"]
        count = len(frame["track_ids"])
        if any(len(frame[key]) != count for key in ("boxes", "scores", "predicted_class")):
            raise ValueError("OVMOT emission fields do not align")
        for index, track_id in enumerate(frame["track_ids"]):
            x0, y0, x1, y1 = frame["boxes"][index]
            category = int(frame["predicted_class"][index]) + 1
            if not 1 <= category <= 1203:
                raise ValueError("OVMOT export changed the full LVIS-v1 class convention")
            rows.append({"image_id": image["id"], "video_id": image["video_id"],
                         # Keep compact runtime IDs. Official TAO explicitly
                         # separates overlapping video namespaces itself; large
                         # bit-packed IDs trigger huge dense upstream ID maps.
                         "track_id": int(track_id), "category_id": category,
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(frame["scores"][index])})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows) + "\n")
    return {"video": group["video"], "frames": len(predictions), "detections": len(rows),
            "path": str(path), "sha256": digest(path), "predicted_class_source": "updated shared-core semantic memory",
            "score_source": "frozen candidate detector confidence", "gt_annotations_used_as_predictions": False,
            "track_id_namespace": "video-local runtime IDs; official TAO makes IDs unique across videos"}


def summarize_teta(summary: dict, categories: list[dict], fields: list[str]) -> dict:
    combined = summary["COMBINED_SEQ"]
    groups = {"overall": ["average"],
              "base": [category["name"] for category in categories if category["frequency"] in {"f", "c"}
                       and category["name"] in combined],
              "novel": [category["name"] for category in categories if category["frequency"] == "r"
                        and category["name"] in combined]}
    result = {}
    for group, names in groups.items():
        # The upstream OVMOT script averages the official per-class summary
        # vectors at cluster threshold 50; values are already percentages.
        vectors = np.array([combined[name]["TETA"][50] for name in names], dtype=float)
        result[group] = (dict(zip(fields, vectors.mean(axis=0).tolist())) if names else None)
    return {"metrics": result, "classes_evaluated": len(combined) - 1,
            "base_classes_evaluated": len(groups["base"]), "novel_classes_evaluated": len(groups["novel"]),
            "scale": "official percentage 0..100", "cluster_threshold": 50,
            "averaging": "official TETA class mean; official OVTrack base/novel class-summary mean"}


def evaluate_tao_teta(gt_path: Path, prediction_path: Path, output: Path,
                      tracker_name: str = "shared_core", workers: int = 8) -> dict:
    source = verify_teta_source()
    sys.path.insert(0, str(TETA_ROOT))
    import teta
    if Path(teta.__file__).resolve().parent != (TETA_ROOT / "teta").resolve():
        raise RuntimeError("U2 must evaluate through the pinned official TETA package")
    data = json.loads(gt_path.read_text())
    dataset = teta.datasets.TAO({"GT_FOLDER": str(gt_path.resolve()), "TRACKERS_FOLDER": str(output.resolve()),
                                "TRACKERS_TO_EVAL": [tracker_name], "TRACKER_SUB_FOLDER": str(prediction_path.resolve()),
                                "OUTPUT_FOLDER": str(output.resolve()), "PRINT_CONFIG": False,
                                "MAX_DETECTIONS": 300})
    metric = teta.metrics.TETA(exhaustive=False)
    evaluator = teta.Evaluator({"USE_PARALLEL": workers > 1, "NUM_PARALLEL_CORES": workers,
                               "BREAK_ON_ERROR": True, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                               "OUTPUT_PER_SEQ_RES": True, "OUTPUT_TEM_RAW_DATA": True,
                               # The frozen official evaluator also initializes
                               # its returned threshold in the summary branch.
                               "PRINT_RESULTS": False, "OUTPUT_SUMMARY": True, "LOG_ON_ERROR": None})
    _, messages = evaluator.evaluate([dataset], [metric])
    if messages["TAO"][tracker_name] != "Success":
        raise RuntimeError("Official TETA did not complete every validation sequence")
    summary_path = output / tracker_name / "teta_summary_results.pth"
    with summary_path.open("rb") as handle:
        summary = pickle.load(handle)
    result = summarize_teta(summary, data["categories"], metric.summary_fields)
    result.update({"status": "COMPLETE", "metric": "official unmodified TETA", "exhaustive": False,
                   "teta_source_commit": source["commit"], "gt_sha256": digest(gt_path),
                   "prediction_sha256": digest(prediction_path), "summary_path": str(summary_path),
                   "videos_evaluated": len(dataset.seq_list), "frames_evaluated": len(data["images"]),
                   "gt_units": len(data["annotations"]), "max_candidates": 300,
                   "main_gate": {"TETA": 34., "AssocA": 37.}})
    overall = result["metrics"]["overall"]
    result["ovmot_gate_pass"] = overall["TETA"] >= 34. and overall["AssocA"] >= 37.
    return result
