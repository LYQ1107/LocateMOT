#!/usr/bin/env python3
"""Evaluate the coupled readout on an immutable native feature clip.

This entrypoint deliberately evaluates *fixed candidates*.  It never reads
GT while producing decisions, never changes the candidate set, and never
loads a second VLM.  The feature archive is the native GroundAnything cache
registered by the source result; its provenance is written verbatim because
an older cache must not be called a current G2 Clean checkpoint.

The resulting TrackEval numbers are a TRAIN-scope diagnostic when the chosen
clip comes from the GroundAnything training manifest.  They are useful for
checking the association/runtime contract, but are not an independent
generalization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Direct ``python tools/...`` execution does not otherwise add the repository
# root to sys.path.  Keep the entrypoint runnable from a clean checkout.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.g2_shared_runtime import CoupledGroundJEVRuntime
from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel


DEFAULT_CLIP = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/"
    "broader_native_MOT1056_gpu7_v3_complete_manifest/clips/MOT20-03"
)
DEFAULT_CHECKPOINT = Path(
    "/data2/user/LocateMOT/outputs/groundjev/coupled/"
    "phase_b_choice_yes_pilot_20261021/adapter-step000192.pt"
)
DEFAULT_OUTPUT = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/"
    "fixed_candidate_mot20_03_phaseb_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(clip: Path) -> list[dict[str, Any]]:
    path = clip / "frame_candidates.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("candidate archive is empty")
    frames = [int(row["frame"]) for row in rows]
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError("fixed-candidate evaluator requires contiguous physical frames")
    return rows


def load_model(checkpoint: Path, device: torch.device) -> GroundJEVModel:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model_state")
    model = GroundJEVModel(2560, hidden_dim=256).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The 192-step checkpoint predates the optional language-context
    # projection.  Context is absent in that cached replay, so the single
    # missing parameter is safe and explicit.  A newer repair checkpoint may
    # contain the projection, in which case no parameter is missing; every
    # other schema mismatch remains an error.
    if unexpected or set(missing) not in (set(), {"context_projector.weight"}):
        raise RuntimeError(f"unexpected coupled checkpoint schema: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


class CachedNativeProvider:
    """Serve one immutable [N,2560] region cache per physical frame."""

    def __init__(self, clip: Path, rows: list[dict[str, Any]], device: torch.device):
        self.clip = clip
        self.rows = rows
        self.device = device
        self.index = 0
        self.forward_count = 0

    def encode_frame(self, image: Any, boxes: list[list[float]], *, image_key: str,
                     task_query: str | None = None) -> dict[str, Any]:
        del image, task_query
        if self.index >= len(self.rows):
            raise RuntimeError("cached provider received more frames than registered")
        row = self.rows[self.index]
        row_boxes = [candidate["box"] for candidate in row.get("candidates", [])]
        if len(boxes) != len(row_boxes) or not np.allclose(np.asarray(boxes), np.asarray(row_boxes), atol=1e-4):
            raise ValueError(f"candidate geometry changed at frame {row['frame']}")
        feature_ref = row.get("native_region_features_npz")
        if not feature_ref:
            raise ValueError(f"frame {row['frame']} has no native feature archive")
        feature_path = self.clip / feature_ref
        with np.load(feature_path) as archive:
            regions = np.asarray(archive["regions"], dtype=np.float32).copy()
        if regions.ndim != 2 or regions.shape != (len(boxes), 2560):
            raise ValueError(f"feature shape mismatch at frame {row['frame']}: {regions.shape}")
        if not np.isfinite(regions).all():
            raise ValueError(f"non-finite cached feature at frame {row['frame']}")
        self.index += 1
        self.forward_count += 1
        return {
            "region_tokens": torch.from_numpy(regions).to(self.device),
            "local_tokens": None,
            "contextual_tokens": None,
            "metadata": {
                "feature_source": "registered_official_DLM_native_region_cache",
                "cache_frame": int(row["frame"]),
                "cache_image": str(image_key),
                "single_backbone": True,
                "cached_replay": True,
            },
        }


def load_gt(path: Path, frames: list[int]) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    wanted = set(frames)
    result = {frame: [] for frame in frames}
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 7:
            continue
        frame = int(float(fields[0]))
        if frame not in wanted or int(float(fields[6])) != 1:
            continue
        track_id = int(float(fields[1]))
        x, y, w, h = (float(value) for value in fields[2:6])
        result[frame].append((track_id, (x, y, x + w, y + h)))
    return result


def write_trackeval_inputs(output: Path, clip_name: str, rows: list[dict[str, Any]],
                           gt: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
                           predictions: list[dict[str, Any]]) -> tuple[Path, Path]:
    eval_root = output / "trackeval"
    sequence_dir = eval_root / "gt" / clip_name
    gt_dir = sequence_dir / "gt"
    tracker_dir = eval_root / "trackers" / "groundjev" / "data"
    gt_dir.mkdir(parents=True, exist_ok=False)
    tracker_dir.mkdir(parents=True, exist_ok=True)
    frame_map = {int(row["frame"]): index for index, row in enumerate(rows, 1)}
    gt_lines: list[str] = []
    for frame, entries in gt.items():
        if frame not in frame_map:
            continue
        for track_id, (x1, y1, x2, y2) in entries:
            gt_lines.append(
                f"{frame_map[frame]},{track_id},{x1:.6f},{y1:.6f},"
                f"{x2 - x1:.6f},{y2 - y1:.6f},1,1,1,1\n"
            )
    (gt_dir / "gt.txt").write_text("".join(gt_lines))
    first_image = Path(rows[0]["image_path"])
    from PIL import Image
    with Image.open(first_image) as image:
        width, height = image.size
    (sequence_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={clip_name}\nimgDir=img1\nframeRate=25\n"
        f"seqLength={len(rows)}\nimWidth={width}\nimHeight={height}\nimExt=.jpg\n"
    )
    pred_lines = []
    for row in predictions:
        frame = frame_map[int(row["frame"])]
        x1, y1, x2, y2 = (float(value) for value in row["box"])
        pred_lines.append(
            f"{frame},{int(row['track_id'])},{x1:.6f},{y1:.6f},"
            f"{x2 - x1:.6f},{y2 - y1:.6f},1,-1,-1,-1\n"
        )
    pred_path = tracker_dir / f"{clip_name}.txt"
    pred_path.write_text("".join(pred_lines))
    return eval_root, pred_path


def run_trackeval(eval_root: Path, clip_name: str, sequence_length: int) -> dict[str, Any]:
    # The pinned TrackEval checkout predates NumPy 1.24.  Keep the
    # compatibility shim local to this evaluator instead of modifying the
    # third-party source tree or changing the metric implementation.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    sys.path.insert(0, str(ROOT / "third_party" / "TrackEval"))
    import trackeval  # type: ignore

    config = {
        "GT_FOLDER": str(eval_root / "gt"),
        "TRACKERS_FOLDER": str(eval_root / "trackers"),
        "TRACKERS_TO_EVAL": ["groundjev"],
        "TRACKER_SUB_FOLDER": "data",
        "PRINT_CONFIG": False,
        "SKIP_SPLIT_FOL": True,
        "BENCHMARK": "MOT20",
        "SEQ_INFO": {clip_name: int(sequence_length)},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "DO_PREPROC": True,
    }
    adapter = trackeval.datasets.MotChallenge2DBox(config)
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]
    evaluator = trackeval.Evaluator({
        "PRINT_CONFIG": False, "PRINT_RESULTS": False, "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False, "BREAK_ON_ERROR": True,
    })
    raw, messages = evaluator.evaluate([adapter], metrics)
    if messages[adapter.get_name()]["groundjev"] != "Success":
        raise RuntimeError(f"TrackEval failed: {messages}")
    values = raw[adapter.get_name()]["groundjev"]["COMBINED_SEQ"]["pedestrian"]
    return {
        "HOTA": float(np.mean(values["HOTA"]["HOTA"])),
        "DetA": float(np.mean(values["HOTA"]["DetA"])),
        "AssA": float(np.mean(values["HOTA"]["AssA"])),
        "IDF1": float(values["Identity"]["IDF1"]),
        "IDSW": int(values["CLEAR"]["IDSW"]),
        "FP": int(values["CLEAR"]["CLR_FP"]),
        "FN": int(values["CLEAR"]["CLR_FN"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--admit-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    clip = args.clip.resolve()
    checkpoint = args.checkpoint.resolve()
    rows = load_rows(clip)
    gt_path = (args.gt or (Path(rows[0]["image_path"]).parents[1] / "gt" / "gt.txt")).resolve()
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_model(checkpoint, device)
    provider = CachedNativeProvider(clip, rows, device)
    bridge = G2FeatureBridge(provider, model_sha256="3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2", device=str(device))
    runtime = CoupledGroundJEVRuntime(bridge, model, admit_threshold=args.admit_threshold)
    predictions: list[dict[str, Any]] = []
    decision_counts: dict[str, int] = {}
    rejected = 0
    started = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            boxes = [candidate["box"] for candidate in row.get("candidates", [])]
            labels = [str(candidate.get("label", "person")) for candidate in row.get("candidates", [])]
            result = runtime.step(
                None, boxes, labels, frame_id=int(row["frame"]),
                image_key=str(row["image_path"]), task_query="person",
            )
            for decision in result["decisions"]:
                token = str(decision["identity"])
                decision_counts[token] = decision_counts.get(token, 0) + 1
            if not result["decisions"]:
                rejected += 1
            for candidate_index, track_id in result["assignments"].items():
                predictions.append({
                    "frame": int(row["frame"]),
                    "candidate_index": int(candidate_index),
                    "track_id": int(track_id),
                    "box": [float(value) for value in boxes[int(candidate_index)]],
                })
    elapsed = time.perf_counter() - started
    gt = load_gt(gt_path, [int(row["frame"]) for row in rows])
    eval_root, prediction_path = write_trackeval_inputs(
        args.output, clip.name, rows, gt, predictions,
    )
    metrics = run_trackeval(eval_root, clip.name, len(rows))
    receipt = {
        "status": "FIXED_CANDIDATE_GROUNDJEV_VIDEO_EVAL_COMPLETE",
        "task": "MOT",
        "dataset": "MOT20",
        "sequence": clip.name,
        "scope": "registered_train_only_native_source_not_benchmark_evaluation",
        "frames": len(rows),
        "frame_start": int(rows[0]["frame"]),
        "frame_end": int(rows[-1]["frame"]),
        "candidate_count": int(sum(len(row.get("candidates", [])) for row in rows)),
        "candidate_source": str(clip / "frame_candidates.jsonl"),
        "candidate_manifest_sha256": sha256_file(clip / "frame_candidates.jsonl"),
        "feature_source": "registered_official_DLM_native_region_cache",
        "feature_archive_provenance": str(clip / "result.json"),
        "backbone_model_sha256": "3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2",
        "coupled_readout_checkpoint": str(checkpoint),
        "coupled_readout_checkpoint_sha256": sha256_file(checkpoint),
        "admit_threshold": float(args.admit_threshold),
        "predictions": len(predictions),
        "decision_counts": decision_counts,
        "empty_decision_frames": rejected,
        "frames_with_predictions": len({int(row["frame"]) for row in predictions}),
        "visual_forward_calls": int(provider.forward_count),
        "second_vlm_loaded": False,
        "single_backbone": True,
        "association_replay_fps": float(len(rows) / max(elapsed, 1e-9)),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "metrics_source": "third_party/TrackEval local MotChallenge2DBox HOTA/CLEAR/Identity",
        "prediction_path": str(prediction_path),
        "gt_path_for_evaluation_only": str(gt_path),
        "official_test_labels_read": False,
        "train_scope_overlap": True,
        "independent_generalization_claim": False,
        "scientific_claim": "fixed-candidate association/runtime diagnostic only; not independent G2 Clean or end-to-end grounding evidence",
        "note": "The cached feature archive predates the current G2 Clean terminal checkpoint and is not relabeled as G2 Clean step 12245.",
    }
    (args.output / "official_metrics.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
