"""Validate the Stage L16 KITTI crop-PBD cache and summarize empty generations."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = {}
    for path in sorted(args.records.glob("*.pkl")):
        record = pickle.load(path.open("rb"))
        records[str(record["video_id"])] = {
            int(frame["frame"]): frame for frame in record["frames"]
        }

    counts = Counter()
    failed_widths = []
    failed_heights = []
    failed_areas = []
    failed_by_video = Counter()
    malformed = []
    nonfinite = []
    failure_encoding_errors = []

    for video, frames in records.items():
        for frame_id, frame in frames.items():
            base = args.cache / "kitti" / video / f"{frame_id:05d}" / "pbd_full"
            paths = {
                "complete": Path(f"{base}.complete"),
                "tensor": Path(f"{base}.safetensors"),
                "meta": Path(f"{base}.meta.json"),
            }
            counts["frames_expected"] += 1
            counts["candidates_expected"] += len(frame["boxes"])
            missing = [name for name, path in paths.items() if not path.exists()]
            if missing:
                malformed.append({"video": video, "frame": frame_id,
                                  "missing": missing})
                continue

            meta = json.loads(paths["meta"].read_text())
            tensors = load_file(str(paths["tensor"]))
            n = len(frame["boxes"])
            failures = [int(index) for index in meta.get("failed_candidates", [])]
            shapes = {name: list(value.shape) for name, value in tensors.items()}
            expected_shapes = {
                "boxes": [n, 4], "clip": [n, 512],
                "pbd_box_end_last": [n, 2048], "gen_score": [n],
            }
            if int(meta.get("candidate_count", -1)) != n or any(
                    shapes.get(name) != shape for name, shape in expected_shapes.items()):
                malformed.append({"video": video, "frame": frame_id,
                                  "candidate_count": meta.get("candidate_count"),
                                  "expected": n, "shapes": shapes})
                continue

            counts["frames_valid"] += 1
            counts["candidates_valid"] += n
            counts["failed_generations"] += len(failures)
            failed_by_video[video] += len(failures)
            for name, value in tensors.items():
                if not bool(value.isfinite().all()):
                    nonfinite.append({"video": video, "frame": frame_id,
                                      "tensor": name})

            pbd = tensors["pbd_box_end_last"].float().numpy()
            scores = tensors["gen_score"].float().numpy()
            failure_set = set(failures)
            zero_rows = set(np.flatnonzero(np.linalg.norm(pbd, axis=1) == 0).tolist())
            zero_scores = set(np.flatnonzero(scores == 0).tolist())
            if zero_rows != failure_set or not failure_set.issubset(zero_scores):
                failure_encoding_errors.append({
                    "video": video, "frame": frame_id,
                    "recorded": sorted(failure_set),
                    "zero_pbd": sorted(zero_rows),
                    "zero_score_extra": sorted(zero_scores - failure_set),
                })

            boxes = np.asarray(frame["boxes"], dtype=np.float32)
            for index in failures:
                x1, y1, x2, y2 = boxes[index]
                width = max(0.0, float(x2 - x1))
                height = max(0.0, float(y2 - y1))
                failed_widths.append(width)
                failed_heights.append(height)
                failed_areas.append(width * height)

    def stats(values: list[float]) -> dict:
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()), "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)), "max": float(array.max()),
        }

    expected = counts["candidates_expected"]
    summary = {
        "counts": dict(counts),
        "failure_rate": counts["failed_generations"] / expected if expected else None,
        "failed_by_video": dict(sorted(failed_by_video.items())),
        "failed_crop_width": stats(failed_widths),
        "failed_crop_height": stats(failed_heights),
        "failed_crop_area": stats(failed_areas),
        "malformed": malformed,
        "nonfinite": nonfinite,
        "failure_encoding_errors": failure_encoding_errors,
        "status": "pass" if not (malformed or nonfinite or failure_encoding_errors)
                  else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
