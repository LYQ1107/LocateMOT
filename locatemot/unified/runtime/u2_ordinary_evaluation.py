"""Full-sequence ordinary-MOT evaluation through unmodified TrackEval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TRACKEVAL = ROOT / "third_party/TrackEval"
# TrackEval's fixed upstream revision predates NumPy's alias removal.
for name, value in (("int", int), ("float", float), ("bool", bool)):
    if name not in np.__dict__:
        setattr(np, name, value)
sys.path.insert(0, str(TRACKEVAL))
import trackeval

from ..data.bdd_tracking import frame_objects
from ..data.legal_scope import assert_legal_video


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_trackeval_source() -> dict:
    manifest = json.loads((ROOT / "outputs/unified/u2/protocol/trackeval_source.json").read_text())
    if Path(trackeval.__file__).resolve().parent != (TRACKEVAL / "trackeval").resolve():
        raise RuntimeError("Ordinary MOT cannot use the expression-modified TrackEval fork")
    for name, sha in manifest["files"].items():
        if digest(TRACKEVAL / name) != sha:
            raise ValueError(f"Official TrackEval source differs: {name}")
    return manifest


def _bdd_label(obj: dict, identity: int, *, ignored: bool = False) -> dict:
    x0, y0, x1, y1 = obj["bbox"]
    return {"id": identity, "category": obj["category"],
            "box2d": {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
            "attributes": {"Crowd": ignored}}


def prepare_gt_view(dataset: str, groups: list[dict], supervision: dict, directory: Path) -> dict:
    """Preserve original identity/ignore semantics in the public file readers.

    MOT half frames are reindexed to contiguous dev time without altering IDs,
    geometry, mark or category. BDD's raw aliases and inclusive box extents
    follow the already verified toolkit contract; ignored boxes become crowd
    regions for the unchanged TrackEval BDD reader. KITTI labels stay original.
    """
    directory.mkdir(parents=True, exist_ok=True)
    seq_info, provenance = {}, []
    for group in groups:
        video, frames = group["video"], group["frames"]
        source = supervision[group["key"]]
        source_path = Path(source["source_path"])
        if dataset == "kitti_legal_development":
            assert_legal_video(video, dataset="refer_kitti_v2")
        if digest(source_path) != source["sha256"]:
            raise ValueError(f"Registered original GT changed: {source_path}")
        seq_info[video] = len(frames)
        if dataset == "bdd100k":
            source_frames = json.loads(source_path.read_text())
            by_name = {frame["name"]: frame for frame in source_frames}
            identities, normalized, positive_count = {}, [], 0
            for index, name in enumerate(frames):
                positives, ignored = frame_objects(by_name[name])
                labels = []
                for obj in positives + [obj for obj in ignored if "bbox" in obj]:
                    identity = identities.setdefault(obj["track_id"], len(identities))
                    labels.append(_bdd_label(obj, identity, ignored="reason" in obj))
                positive_count += len(positives)
                normalized.append({"name": name, "index": index, "labels": labels})
            path = directory / f"{video}.json"
            path.write_text(json.dumps(normalized) + "\n")
        elif dataset == "kitti_legal_development":
            # Legal KITTI dev uses complete original videos, zero-based frames.
            if [int(Path(frame).stem) for frame in frames] != list(range(len(frames))):
                raise ValueError("KITTI development is not the complete registered video")
            path = directory / "label_02" / f"{video}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_path.read_bytes())
            positive_count = None  # Count after the official visibility preprocessing.
        else:
            mapping = {int(Path(frame).stem): index + 1 for index, frame in enumerate(frames)}
            lines, positive_count = [], 0
            for line in source_path.read_text().splitlines():
                fields = line.split(",")
                frame = int(float(fields[0]))
                if frame in mapping:
                    fields[0] = str(mapping[frame])
                    lines.append(",".join(fields))
                    positive_count += float(fields[6]) == 1 and float(fields[7]) == 1
            path = directory / video / "gt/gt.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
        provenance.append({"video": video, "source": source, "source_frames": frames,
                           "view_path": str(path), "view_sha256": digest(path),
                           "positive_gt_units_before_official_preprocessing": positive_count})
    if dataset == "kitti_legal_development":
        (directory / "evaluate_tracking.seqmap.training").write_text(
            "".join(f"{video} empty 0 {length}\n" for video, length in sorted(seq_info.items())))
    return {"dataset": dataset, "sequence_lengths": seq_info, "sources": provenance,
            "frame_count": sum(seq_info.values()), "official_test_labels_read": False}


def write_predictions(dataset: str, group: dict, frames: list[dict], path: Path) -> None:
    if [item["frame"] for item in frames] != group["frames"]:
        raise ValueError("Prediction export does not cover the complete registered frame sequence")
    path.parent.mkdir(parents=True, exist_ok=True)
    if dataset == "bdd100k":
        classes = [name.strip() for name in group["vocabulary"].split(".") if name.strip()]
        result = []
        for index, frame in enumerate(frames):
            labels = []
            for identity, box, score, category in zip(frame["track_ids"], frame["boxes"], frame["scores"], frame["predicted_class"]):
                label = _bdd_label({"bbox": box, "category": classes[category]}, identity)
                label["score"] = score
                labels.append(label)
            result.append({"name": frame["frame"], "index": index, "labels": labels})
        path.write_text(json.dumps(result) + "\n")
    elif dataset == "kitti_legal_development":
        classes = [name.strip() for name in group["vocabulary"].split(".") if name.strip()]
        # These retain the original U1 prompt meanings. Person refers to the
        # sitting-person distractor, not a replacement pedestrian class.
        mapping = {"car": "Car", "van": "Van", "truck": "Truck", "bus": "Truck",
                   "tram": "Tram", "pedestrian": "Pedestrian", "person": "Person_sitting",
                   "cyclist": "Cyclist", "bicycle": "Cyclist", "motorcycle": "Cyclist"}
        with path.open("w") as handle:
            for frame in frames:
                index = int(Path(frame["frame"]).stem)
                for identity, box, score, category in zip(frame["track_ids"], frame["boxes"], frame["scores"], frame["predicted_class"]):
                    fields = [index, identity, mapping[classes[category]], 0, 0, -10, *box,
                              -1, -1, -1, -1000, -1000, -1000, -10, score]
                    handle.write(" ".join(map(str, fields)) + "\n")
    else:
        with path.open("w") as handle:
            for index, frame in enumerate(frames, start=1):
                for identity, box, score in zip(frame["track_ids"], frame["boxes"], frame["scores"]):
                    x0, y0, x1, y1 = box
                    fields = [index, identity, x0, y0, x1 - x0, y1 - y0, score, -1, -1, -1]
                    handle.write(",".join(map(str, fields)) + "\n")


def evaluate_ordinary(dataset: str, gt_directory: Path, prediction_root: Path, seq_info: dict, tracker: str) -> dict:
    source = verify_trackeval_source()
    config = {"GT_FOLDER": str(gt_directory), "TRACKERS_FOLDER": str(prediction_root),
              "TRACKERS_TO_EVAL": [tracker], "PRINT_CONFIG": False, "TRACKER_SUB_FOLDER": "data"}
    if dataset == "bdd100k":
        adapter = trackeval.datasets.BDD100K(config)
    elif dataset == "kitti_legal_development":
        adapter = trackeval.datasets.Kitti2DBox(config)
    else:
        config.update(BENCHMARK="MOT20" if dataset == "mot20" else "MOT17", SKIP_SPLIT_FOL=True,
                      SEQ_INFO=seq_info, DO_PREPROC=True)
        adapter = trackeval.datasets.MotChallenge2DBox(config)
    if set(adapter.seq_list) != set(seq_info):
        raise ValueError("TrackEval file reader uses a different registered sequence scope")
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
               trackeval.metrics.Identity({"PRINT_CONFIG": False})]
    evaluator = trackeval.Evaluator({"PRINT_CONFIG": False, "PRINT_RESULTS": False, "TIME_PROGRESS": False,
                                    "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                                    "BREAK_ON_ERROR": True})
    result, messages = evaluator.evaluate([adapter], metrics)
    if messages[adapter.get_name()][tracker] != "Success":
        raise ValueError("The official TrackEval run did not complete")
    combined = result[adapter.get_name()][tracker]["COMBINED_SEQ"]
    classes = adapter.class_list
    class_results = {name: combined[name] for name in classes}
    aggregate = {}
    for metric in metrics:
        aggregate[metric.get_name()] = metric.combine_classes_class_averaged(
            {name: class_results[name][metric.get_name()] for name in classes})

    def scalars(values: dict) -> dict:
        return {**{key: float(np.mean(values["HOTA"][key])) for key in ("HOTA", "DetA", "AssA")},
                "IDF1": float(values["Identity"]["IDF1"]), "IDSW": int(values["CLEAR"]["IDSW"])}

    return {"dataset": dataset, "status": "COMPLETE", "metrics": scalars(aggregate),
            "per_class": {name: scalars(values) for name, values in class_results.items()},
            "gt_detections": sum(values["Count"]["GT_Dets"] for values in class_results.values()),
            "gt_tracks": sum(values["Count"]["GT_IDs"] for values in class_results.values()),
            "sequences_evaluated": len(seq_info), "frames_evaluated": sum(seq_info.values()),
            "aggregation": "official sequence combination, then official class-averaged combination",
            "hota_alpha_thresholds": metrics[0].array_labels.tolist(),
            "metric_scale": "fraction; IDSW is an integer count", "trackeval_commit": source["commit"],
            "official_test_labels_read": False}


__all__ = ["prepare_gt_view", "write_predictions", "evaluate_ordinary", "verify_trackeval_source"]
