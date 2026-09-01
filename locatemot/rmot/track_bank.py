"""Causal, query-reusable UIDM track-bank construction for Stage L16."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from locatemot.tracking.online_tracker import OnlineTracker


STATUS = {"TENTATIVE": 0, "ACTIVE": 1, "LOST": 2, "TERMINATED": 3}


def _geometry(boxes: np.ndarray, image_size) -> np.ndarray:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    width, height = [max(1.0, float(value)) for value in image_size]
    if not len(boxes):
        return np.zeros((0, 7), np.float32)
    x1, y1, x2, y2 = boxes.T
    bw = np.maximum(0.0, x2 - x1)
    bh = np.maximum(0.0, y2 - y1)
    nw, nh = bw / width, bh / height
    return np.stack(((x1 + x2) * 0.5 / width,
                     (y1 + y2) * 0.5 / height,
                     nw, nh, nw * nh,
                     np.clip(bw / np.maximum(bh, 1.0), 0.0, 20.0) / 20.0,
                     y2 / height), axis=1).astype(np.float32)


def _pair_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-9, aa + bb - inter)


def _context(boxes: np.ndarray, image_size) -> np.ndarray:
    """Eight causal same-frame set features for every observation."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    n = len(boxes)
    if not n:
        return np.zeros((0, 8), np.float32)
    geom = _geometry(boxes, image_size)
    centers = geom[:, :2]
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1) if n > 1 else np.zeros(n, np.int64)
    nearest_distance = distance[np.arange(n), nearest] if n > 1 else np.ones(n)
    nearest_delta = centers[nearest] - centers if n > 1 else np.zeros((n, 2))
    nearest_iou = np.asarray([
        _pair_iou(boxes[i], boxes[nearest[i]]) if n > 1 else 0.0
        for i in range(n)], np.float32)
    finite_distance = np.where(np.isfinite(distance), distance, 0.0)
    mean_distance = finite_distance.sum(axis=1) / max(1, n - 1)
    rank_x = np.argsort(np.argsort(centers[:, 0], kind="stable"),
                        kind="stable") / max(1, n - 1)
    rank_y = np.argsort(np.argsort(centers[:, 1], kind="stable"),
                        kind="stable") / max(1, n - 1)
    return np.column_stack((
        np.full(n, np.log1p(n) / 6.0, np.float32),
        nearest_delta[:, 0], nearest_delta[:, 1], nearest_distance,
        nearest_iou, rank_x, rank_y, mean_distance,
    )).astype(np.float32)


def _motion(track, image_size) -> np.ndarray:
    width, height = [max(1.0, float(value)) for value in image_size]
    current = np.asarray(track.last_box, np.float32)
    previous = np.asarray(track.prev_box if track.prev_box is not None
                          else current, np.float32)
    cc = np.asarray([(current[0] + current[2]) * 0.5,
                     (current[1] + current[3]) * 0.5])
    pc = np.asarray([(previous[0] + previous[2]) * 0.5,
                     (previous[1] + previous[3]) * 0.5])
    delta = (cc - pc) / np.asarray([width, height])
    cw, ch = max(1.0, current[2] - current[0]), max(1.0, current[3] - current[1])
    pw, ph = max(1.0, previous[2] - previous[0]), max(1.0, previous[3] - previous[1])
    return np.asarray([
        delta[0], delta[1], np.log(cw / pw), np.log(ch / ph),
        float(np.linalg.norm(delta)), min(track.age, 300) / 300.0,
        track.hits / max(1, track.age), min(track.lost_age, 30) / 30.0,
    ], np.float32)


def _as_feature(frame: dict, name: str, n: int, dim: int) -> np.ndarray:
    value = np.asarray(frame.get(name, np.zeros((n, dim))), np.float32)
    if value.shape != (n, dim):
        value = np.zeros((n, dim), np.float32)
    return np.nan_to_num(value, copy=False)


def _diagnostics(frame_ids, track_ids, candidate_gt, gt_boxes_by_frame,
                 frame_ptr) -> dict:
    visible = 0
    covered = 0
    per_identity_visible = defaultdict(int)
    per_identity_tracks = defaultdict(list)
    switches = 0
    last_track = {}
    for fi, frame in enumerate(frame_ids):
        start, end = int(frame_ptr[fi]), int(frame_ptr[fi + 1])
        gt_ids = set(str(value) for value in gt_boxes_by_frame.get(int(frame), {}))
        visible += len(gt_ids)
        for gt_id in gt_ids:
            per_identity_visible[gt_id] += 1
        best = {}
        for index in range(start, end):
            gt_id = candidate_gt[index]
            if gt_id is None:
                continue
            gt_id = str(gt_id)
            # One bank identity per GT/frame is sufficient for coverage.  The
            # first candidate is deterministic because proposal order is fixed.
            best.setdefault(gt_id, int(track_ids[index]))
        covered += len(gt_ids.intersection(best))
        for gt_id, bank_id in best.items():
            per_identity_tracks[gt_id].append(bank_id)
            if gt_id in last_track and last_track[gt_id] != bank_id:
                switches += 1
            last_track[gt_id] = bank_id
    fragments = sum(max(0, len(set(values)) - 1)
                    for values in per_identity_tracks.values())
    trajectory_coverage = [
        len(values) / max(1, per_identity_visible[identity])
        for identity, values in per_identity_tracks.items()
    ]
    return {
        "gt_observations": int(visible),
        "covered_gt_observations": int(covered),
        "observation_recall": covered / visible if visible else None,
        "gt_trajectories": len(per_identity_visible),
        "covered_gt_trajectories": len(per_identity_tracks),
        "trajectory_recall": (len(per_identity_tracks) /
                              max(1, len(per_identity_visible))),
        "mean_trajectory_coverage": (float(np.mean(trajectory_coverage))
                                     if trajectory_coverage else None),
        "fragmentations": int(fragments),
        "id_switches": int(switches),
    }


def build_track_bank(model, frames: Iterable[dict], image_size, generic_spec,
                     dataset: str, video_id: str, split: str,
                     checkpoint_sha256: str, query_count: int = 0) -> tuple[dict, dict, list]:
    """Run one fixed-specification UIDM pass and return bank, audit, labels."""
    frames = list(frames)
    device = next(model.parameters()).device
    tracker = OnlineTracker(
        variant="UIDM", uidm=model.uidm, uidm_adapter=model.adapter,
        uidm_spec=np.asarray(generic_spec, np.float32), device=str(device),
        output_all_candidates=True)
    tracker.uidm_sem_in_core = model.sem_in_core
    tracker.uidm_new_margin = 0.0
    tracker.l1d_weights = (0.4, 0.2, 0.4)
    tracker.l1d_threshold = 0.25
    tracker.image_size = image_size

    arrays = defaultdict(list)
    frame_ptr = [0]
    candidate_gt = []
    gt_boxes_by_frame = {}
    appearance_ema = {}
    start_time = time.time()
    for frame in frames:
        frame_id = int(frame["frame"])
        boxes = np.asarray(frame.get("boxes", []), np.float32).reshape(-1, 4)
        n = len(boxes)
        pbd = _as_feature(frame, "pbd", n, 2048)
        clip = _as_feature(frame, "clip", n, 512)
        gen = np.asarray(frame.get("gen", np.zeros(n)), np.float32).reshape(-1)
        if len(gen) != n:
            gen = np.zeros(n, np.float32)
        gen = np.nan_to_num(gen)
        candidates = [{
            "box": boxes[index],
            "features": {
                "pbd": pbd[index], "pbd_be": pbd[index],
                "clip": clip[index], "region": np.zeros(4608, np.float32),
                "geom": np.zeros(5, np.float32), "gen": float(gen[index]),
            },
            "index": index,
        } for index in range(n)]
        outputs = tracker.process_frame(frame_id, candidates) if n else []
        if len(outputs) != n:
            raise RuntimeError(
                f"{dataset}/{video_id}/{frame_id}: {len(outputs)} outputs for {n} candidates")
        tracks = {track.track_id: track for track in tracker.tracks}
        context = _context(boxes, image_size)
        geometry = _geometry(boxes, image_size)
        frame_track_ids = []
        h, ref, anchor, lifecycle, motion, history_clip = [], [], [], [], [], []
        for index, output in enumerate(outputs):
            track_id = int(output["track_id"])
            track = tracks[track_id]
            state = track.uidm_state or {}
            hidden = np.asarray(state.get("h", np.zeros(model.d_model)), np.float32)
            reference = np.asarray(state.get("ref_pbd", pbd[index]), np.float32)
            anchored = np.asarray(state.get("anchor_pbd", reference), np.float32)
            old_ema = appearance_ema.get(track_id, clip[index])
            new_ema = 0.8 * old_ema + 0.2 * clip[index]
            appearance_ema[track_id] = new_ema
            frame_track_ids.append(track_id)
            h.append(hidden)
            ref.append(reference)
            anchor.append(anchored)
            lifecycle.append([
                track.hits, track.age, track.lost_age,
                float(state.get("alive", 0.0)), STATUS.get(track.status, -1),
                track.confidence, frame_id - track.birth_frame + 1,
                len(track.history),
            ])
            motion.append(_motion(track, image_size))
            history_clip.append(new_ema)

        arrays["frame"].append(np.full(n, frame_id, np.int32))
        source_index = np.asarray(
            frame.get("source_index", np.arange(n)), dtype=np.int32)
        if source_index.shape != (n,):
            raise RuntimeError(f"{dataset}/{video_id}/{frame_id}: bad source index")
        arrays["candidate_index"].append(source_index)
        arrays["track_id"].append(np.asarray(frame_track_ids, np.int32))
        arrays["box"].append(boxes.astype(np.float32))
        arrays["objectness"].append(gen.astype(np.float32))
        arrays["clip"].append(clip.astype(np.float16))
        arrays["history_clip"].append(np.asarray(history_clip, np.float16).reshape(n, 512))
        arrays["pbd"].append(pbd.astype(np.float16))
        arrays["uidm_h"].append(np.asarray(h, np.float16).reshape(n, model.d_model))
        arrays["uidm_ref_pbd"].append(np.asarray(ref, np.float16).reshape(n, 2048))
        arrays["uidm_anchor_pbd"].append(np.asarray(anchor, np.float16).reshape(n, 2048))
        arrays["geometry"].append(geometry.astype(np.float32))
        arrays["motion"].append(np.asarray(motion, np.float32).reshape(n, 8))
        arrays["context"].append(context.astype(np.float32))
        arrays["lifecycle"].append(np.asarray(lifecycle, np.float32).reshape(n, 8))
        raw_gt = list(frame.get("cand_gt", [None] * n))
        candidate_gt.extend([None if value is None else str(value)
                             for value in raw_gt[:n]] +
                            [None] * max(0, n - len(raw_gt)))
        gt_boxes_by_frame[frame_id] = frame.get("gt_boxes", {})
        frame_ptr.append(frame_ptr[-1] + n)

    tensors = {}
    widths = {
        "frame": (), "candidate_index": (), "track_id": (), "box": (4,),
        "objectness": (), "clip": (512,), "history_clip": (512,),
        "pbd": (2048,), "uidm_h": (model.d_model,),
        "uidm_ref_pbd": (2048,), "uidm_anchor_pbd": (2048,),
        "geometry": (7,), "motion": (8,), "context": (8,),
        "lifecycle": (8,),
    }
    for name, tail in widths.items():
        if arrays[name]:
            value = np.concatenate(arrays[name], axis=0)
        else:
            dtype = np.int32 if name in {"frame", "candidate_index", "track_id"} else np.float32
            value = np.zeros((0,) + tail, dtype=dtype)
        tensors[name] = torch.from_numpy(value)
    tensors["frame_ptr"] = torch.as_tensor(frame_ptr, dtype=torch.int64)
    tensors["frame_ids"] = torch.as_tensor(
        [int(frame["frame"]) for frame in frames], dtype=torch.int32)

    track_ids = tensors["track_id"].numpy()
    frame_ids = tensors["frame_ids"].numpy()
    integrity = _diagnostics(frame_ids, track_ids, candidate_gt,
                             gt_boxes_by_frame, frame_ptr)
    reuse_hash = hashlib.sha256()
    for name in ("frame", "candidate_index", "track_id", "box"):
        reuse_hash.update(tensors[name].numpy().tobytes())
    elapsed = time.time() - start_time
    source_observations = sum(int(frame.get("_source_count", len(frame.get("boxes", []))))
                              for frame in frames)
    metadata = {
        "format": "locatemot-l16-track-bank-v1",
        "dataset": dataset, "video_id": video_id, "split": split,
        "image_size": [int(value) for value in image_size],
        "generic_specification": "all objects in the scene",
        "generic_spec_sha256": hashlib.sha256(
            np.asarray(generic_spec, np.float32).tobytes()).hexdigest(),
        "shared_checkpoint_sha256": checkpoint_sha256,
        "frames": len(frames), "observations": int(len(track_ids)),
        "source_observations": int(source_observations),
        "exact_duplicates_removed": int(source_observations - len(track_ids)),
        "unique_tracks": int(len(set(track_ids.tolist()))),
        "candidates_per_frame": (len(track_ids) / max(1, len(frames))),
        "queries_reusing_bank": int(query_count),
        "runtime_seconds": elapsed,
        "reuse_equivalence_sha256": reuse_hash.hexdigest(),
        "causal": True, "query_independent_identity_pass": True,
    }
    bank = {"metadata": metadata, "tensors": tensors}
    audit = dict(metadata)
    audit.update(integrity)
    labels = candidate_gt
    return bank, audit, labels


def save_track_bank(bank: dict, audit: dict, labels: list, output: Path,
                    save_supervision: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(bank, temporary)
    os.replace(temporary, output)
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if save_supervision:
        label_path = output.with_suffix(".labels.json")
        label_path.write_text(json.dumps({"candidate_gt": labels}) + "\n")
    output.with_suffix(".complete").write_text("ok\n")


def load_track_bank(path: str | Path, map_location="cpu") -> dict:
    return torch.load(Path(path), map_location=map_location,
                      weights_only=False)
