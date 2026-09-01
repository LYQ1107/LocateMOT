"""Stream-validate all Stage L16 causal track banks and aggregate audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


EXPECTED_TAILS = {
    "frame": (), "candidate_index": (), "track_id": (), "box": (4,),
    "objectness": (), "clip": (512,), "history_clip": (512,),
    "pbd": (2048,), "uidm_h": (384,), "uidm_ref_pbd": (2048,),
    "uidm_anchor_pbd": (2048,), "geometry": (7,), "motion": (8,),
    "context": (8,), "lifecycle": (8,),
}
CHECKPOINT_SHA256 = "f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("outputs/l16/track_banks"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("outputs/l16/data/protocol/split_manifest.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/l16/track_banks/integrity.json"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    expected = {
        "kitti": set(sum((protocol["kitti_v2"][key]
                          for key in ("train", "train_val", "official_eval")), [])),
        "dance_train": set(protocol["refer_dance"]["train"] +
                           protocol["refer_dance"]["train_val"]),
        "dance_eval": set(protocol["refer_dance"]["official_eval"]),
    }
    failures = []
    summaries = {}
    spec_hashes = set()
    total_bytes = 0
    for dataset, videos in expected.items():
        directory = args.root / dataset
        found = {path.stem for path in directory.glob("*.pt")}
        if found != videos:
            failures.append({"dataset": dataset,
                             "missing": sorted(videos - found),
                             "extra": sorted(found - videos)})
        rows = []
        for video in sorted(found.intersection(videos)):
            path = directory / f"{video}.pt"
            complete = path.with_suffix(".complete")
            audit_path = path.with_suffix(".audit.json")
            labels_path = path.with_suffix(".labels.json")
            if not complete.exists() or not audit_path.exists():
                failures.append({"dataset": dataset, "video": video,
                                 "missing_sidecar": True})
                continue
            bank = torch.load(path, map_location="cpu", weights_only=False)
            meta, tensors = bank["metadata"], bank["tensors"]
            n = int(meta["observations"])
            frames = int(meta["frames"])
            spec_hashes.add(meta.get("generic_spec_sha256"))
            if meta.get("shared_checkpoint_sha256") != CHECKPOINT_SHA256:
                failures.append({"dataset": dataset, "video": video,
                                 "checkpoint_hash": meta.get("shared_checkpoint_sha256")})
            for name, tail in EXPECTED_TAILS.items():
                value = tensors.get(name)
                if value is None or tuple(value.shape) != (n,) + tail:
                    failures.append({"dataset": dataset, "video": video,
                                     "tensor": name,
                                     "shape": None if value is None else list(value.shape),
                                     "expected": [n, *tail]})
                    continue
                if value.is_floating_point() and not bool(value.isfinite().all()):
                    failures.append({"dataset": dataset, "video": video,
                                     "nonfinite": name})
            ptr = tensors.get("frame_ptr")
            frame_ids = tensors.get("frame_ids")
            if ptr is None or tuple(ptr.shape) != (frames + 1,) or \
                    int(ptr[0]) != 0 or int(ptr[-1]) != n or \
                    not bool((ptr[1:] >= ptr[:-1]).all()):
                failures.append({"dataset": dataset, "video": video,
                                 "frame_ptr": "invalid"})
            elif frame_ids is None or tuple(frame_ids.shape) != (frames,):
                failures.append({"dataset": dataset, "video": video,
                                 "frame_ids": "invalid"})
            else:
                flat_frames = tensors["frame"]
                for index in range(frames):
                    start, end = int(ptr[index]), int(ptr[index + 1])
                    if end > start and not bool(
                            (flat_frames[start:end] == frame_ids[index]).all()):
                        failures.append({"dataset": dataset, "video": video,
                                         "frame_alignment": index})
                        break
            should_have_labels = meta.get("split") != "official_eval"
            if labels_path.exists() != should_have_labels:
                failures.append({"dataset": dataset, "video": video,
                                 "supervision_boundary": "unexpected"})
            elif should_have_labels:
                labels = json.loads(labels_path.read_text())["candidate_gt"]
                if len(labels) != n:
                    failures.append({"dataset": dataset, "video": video,
                                     "label_count": len(labels), "expected": n})
            audit = json.loads(audit_path.read_text())
            audit["file_bytes"] = path.stat().st_size
            rows.append(audit)
            total_bytes += path.stat().st_size
            del bank, tensors
        def weighted(key, weight="frames"):
            values = [(row.get(key), row.get(weight, 0)) for row in rows]
            values = [(value, amount) for value, amount in values
                      if value is not None and amount]
            return (sum(value * amount for value, amount in values) /
                    sum(amount for _, amount in values)) if values else None
        summaries[dataset] = {
            "videos": len(rows),
            "frames": sum(row["frames"] for row in rows),
            "observations": sum(row["observations"] for row in rows),
            "unique_tracks_sum": sum(row["unique_tracks"] for row in rows),
            "runtime_seconds_sum": sum(row["runtime_seconds"] for row in rows),
            "bytes": sum(row["file_bytes"] for row in rows),
            "candidates_per_frame": weighted("candidates_per_frame"),
            "observation_recall": weighted("observation_recall", "gt_observations"),
            "mean_trajectory_coverage": weighted(
                "mean_trajectory_coverage", "gt_trajectories"),
            "fragmentations": sum(row.get("fragmentations", 0) for row in rows),
            "id_switches": sum(row.get("id_switches", 0) for row in rows),
        }
    result = {
        "status": "pass" if not failures and len(spec_hashes) == 1 else "fail",
        "generic_spec_hashes": sorted(value for value in spec_hashes if value),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "summaries": summaries,
        "total_bytes": total_bytes,
        "failures": failures,
    }
    if len(spec_hashes) != 1:
        result["failures"].append({"generic_spec_hashes": sorted(spec_hashes)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
