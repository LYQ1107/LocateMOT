"""Long-lived reserve identity and cross-pool observation grouping for L19.

The reserve detector supplies boxes and crop CLIP vectors, not UIDM/PBD
states.  This module therefore builds a causal identity memory from those
vectors and motion, and exposes the provenance explicitly.  It is an
RMOT-only bank transform; the frozen L16 bank is never edited.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def safe_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, 1e-6)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


def box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) * 0.5,
                       (box[1] + box[3]) * 0.5], np.float32)


def shift_box(box: np.ndarray, velocity: np.ndarray, gap: int) -> np.ndarray:
    result = np.asarray(box, np.float32).copy()
    result[[0, 2]] += float(velocity[0]) * gap
    result[[1, 3]] += float(velocity[1]) * gap
    return result


@dataclass
class ReserveMemory:
    track_id: int
    box: np.ndarray
    velocity: np.ndarray
    appearance: np.ndarray
    anchor: np.ndarray
    last_frame: int
    first_frame: int
    hits: int
    age: int


def _match_score(box: np.ndarray, memory: ReserveMemory, clip: np.ndarray,
                 gap: int, image_size: tuple[int, int]) -> tuple[float, bool]:
    predicted = shift_box(memory.box, memory.velocity, gap)
    overlap = box_iou(box, predicted)
    appearance = float(np.dot(safe_normalize(clip[None])[0],
                              safe_normalize(memory.appearance[None])[0]))
    center_delta = np.linalg.norm(box_center(box) - box_center(predicted))
    width = max(1.0, float(predicted[2] - predicted[0]))
    height = max(1.0, float(predicted[3] - predicted[1]))
    scale = max(1.0, min(width, height))
    center_distance = center_delta / scale
    # These are detector-agnostic gates selected from train-only motion and
    # CLIP distributions, not from test GT.  A track can survive an occlusion
    # gap up to 12 frames when appearance and predicted motion agree.
    # Gap-aware gating is the second and final repair round.  Recent frames
    # retain the L18 linker operating region; a reactivation across a longer
    # gap must satisfy stronger appearance and motion agreement.  A single
    # static threshold was either over-merging (L19 v1) or fragmenting almost
    # every observation (L19 v2).
    if gap <= 2:
        allowed = ((appearance >= 0.78 and center_distance <= 3.0) or
                   (overlap >= 0.12 and center_distance <= 4.0))
    else:
        allowed = ((appearance >= 0.86 and center_distance <= 2.0) or
                   (overlap >= 0.20 and center_distance <= 2.5) or
                   (appearance >= 0.92 and center_distance <= 3.5))
    score = (0.50 * appearance + 0.35 * overlap +
             0.15 * max(0.0, 1.0 - center_distance / 5.0))
    return score, allowed


def long_reserve_track_ids(boxes_by_frame: list[np.ndarray],
                           clips_by_frame: list[np.ndarray],
                           frame_ids: list[int],
                           image_size: tuple[int, int],
                           max_gap: int = 12) -> list[np.ndarray]:
    """Causally associate reserve boxes with a persistent memory.

    Matching is one-to-one per frame.  Unmatched memories remain available
    through ``max_gap`` frames, while their velocity and appearance are only
    updated on an observed assignment.
    """
    active: dict[int, ReserveMemory] = {}
    next_id = 1
    result = []
    for boxes, clips, frame in zip(boxes_by_frame, clips_by_frame, frame_ids):
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        clips = np.asarray(clips, np.float32).reshape(-1, 512)
        memories = [memory for memory in active.values()
                    if int(frame) - memory.last_frame <= max_gap]
        assignments = np.full(len(boxes), -1, np.int64)
        if len(boxes) and memories:
            costs = np.full((len(boxes), len(memories)), 1e6, np.float32)
            for i, box in enumerate(boxes):
                for j, memory in enumerate(memories):
                    gap = max(1, int(frame) - memory.last_frame)
                    score, allowed = _match_score(
                        box, memory, clips[i], gap, image_size)
                    if allowed:
                        costs[i, j] = 1.0 - score
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows.tolist(), cols.tolist()):
                if costs[i, j] >= 1e5:
                    continue
                assignments[i] = memories[j].track_id
                memory = memories[j]
                current_center = box_center(box)
                previous_center = box_center(memory.box)
                gap = max(1, int(frame) - memory.last_frame)
                velocity = (current_center - previous_center) / gap
                memory.velocity = 0.70 * memory.velocity + 0.30 * velocity
                memory.box = box.copy()
                memory.appearance = (0.80 * memory.appearance +
                                     0.20 * clips[i])
                memory.last_frame = int(frame)
                memory.hits += 1
                memory.age = int(frame) - memory.first_frame + 1
        for i, box in enumerate(boxes):
            if assignments[i] >= 0:
                continue
            track_id = next_id
            next_id += 1
            assignments[i] = track_id
            active[track_id] = ReserveMemory(
                track_id=track_id, box=box.copy(),
                velocity=np.zeros(2, np.float32),
                appearance=clips[i].copy(), anchor=clips[i].copy(),
                last_frame=int(frame), first_frame=int(frame), hits=1, age=1)
        # Keep updated memories and unmatched recent memories.  Explicitly
        # remove stale entries so state size is bounded on long videos.
        active = {key: value for key, value in active.items()
                  if int(frame) - value.last_frame <= max_gap}
        result.append(assignments)
    return result


def reserve_identity_features(boxes_by_frame: list[np.ndarray],
                              clips_by_frame: list[np.ndarray],
                              ids_by_frame: list[np.ndarray],
                              frame_ids: list[int],
                              image_size: tuple[int, int]) -> dict[str, list[np.ndarray]]:
    """Create non-zero causal identity views from clip/history/motion memory."""
    memories: dict[int, ReserveMemory] = {}
    out = {key: [] for key in (
        "history_clip", "pbd", "uidm_h", "uidm_ref_pbd",
        "uidm_anchor_pbd", "motion", "lifecycle")}
    width, height = max(1.0, float(image_size[0])), max(1.0, float(image_size[1]))
    for boxes, clips, ids, frame in zip(
            boxes_by_frame, clips_by_frame, ids_by_frame, frame_ids):
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        clips = np.asarray(clips, np.float32).reshape(-1, 512)
        histories = np.zeros_like(clips, np.float32)
        pbd_values = np.zeros((len(boxes), 2048), np.float32)
        uidm_values = np.zeros((len(boxes), 384), np.float32)
        ref_values = np.zeros((len(boxes), 2048), np.float32)
        anchor_values = np.zeros((len(boxes), 2048), np.float32)
        motion_values = np.zeros((len(boxes), 8), np.float32)
        lifecycle_values = np.zeros((len(boxes), 8), np.float32)
        for i, track_id in enumerate(np.asarray(ids).tolist()):
            clip = clips[i]
            old = memories.get(int(track_id))
            if old is None:
                old_box = boxes[i].copy()
                history = clip.copy()
                velocity = np.zeros(2, np.float32)
                hits, age = 1, 1
                anchor = clip.copy()
            else:
                old_box = old.box.copy()
                history = 0.80 * old.appearance + 0.20 * clip
                current_center = box_center(boxes[i])
                old_center = box_center(old.box)
                gap = max(1, int(frame) - old.last_frame)
                velocity = (current_center - old_center) / gap
                hits, age, anchor = old.hits + 1, old.age + 1, old.anchor
            histories[i] = safe_normalize(history[None])[0]
            delta = clip - histories[i]
            # Four 512-D views retain current appearance, causal history,
            # change, and disagreement.  This replaces the L18 all-zero PBD
            # and UIDM placeholders without pretending to be official UIDM.
            pbd_values[i] = np.concatenate((clip, histories[i], delta,
                                            np.abs(delta)), axis=0)
            ref_values[i] = np.concatenate((histories[i], clip,
                                             np.abs(delta), delta), axis=0)
            anchor_delta = anchor - histories[i]
            anchor_values[i] = np.concatenate((anchor, histories[i],
                                               anchor_delta,
                                               np.abs(anchor_delta)), axis=0)
            uidm_values[i] = np.concatenate((clip[:256], histories[i][:128]),
                                              axis=0)
            current = boxes[i]
            old_w = max(1.0, float(old_box[2] - old_box[0]))
            old_h = max(1.0, float(old_box[3] - old_box[1]))
            current_w = max(1.0, float(current[2] - current[0]))
            current_h = max(1.0, float(current[3] - current[1]))
            norm_velocity = velocity / np.asarray([width, height], np.float32)
            motion_values[i] = np.asarray([
                norm_velocity[0], norm_velocity[1],
                np.log(current_w / old_w), np.log(current_h / old_h),
                np.linalg.norm(norm_velocity), min(age, 300) / 300.0,
                hits / max(1, age), 0.0,
            ], np.float32)
            lifecycle_values[i] = np.asarray([
                hits, age, 0.0, 1.0, 1.0, 1.0,
                age, hits,
            ], np.float32)
            memories[int(track_id)] = ReserveMemory(
                track_id=int(track_id), box=current.copy(),
                velocity=0.70 * (old.velocity if old is not None else
                                  np.zeros(2, np.float32)) + 0.30 * velocity,
                appearance=history.copy(), anchor=anchor.copy(),
                last_frame=int(frame),
                first_frame=(old.first_frame if old is not None else int(frame)),
                hits=hits, age=age)
        out["history_clip"].append(histories.astype(np.float16))
        out["pbd"].append(pbd_values.astype(np.float16))
        out["uidm_h"].append(uidm_values.astype(np.float16))
        out["uidm_ref_pbd"].append(ref_values.astype(np.float16))
        out["uidm_anchor_pbd"].append(anchor_values.astype(np.float16))
        out["motion"].append(motion_values)
        out["lifecycle"].append(lifecycle_values)
    return out


def observation_groups(main_boxes_by_frame: list[np.ndarray],
                        main_clips_by_frame: list[np.ndarray],
                        reserve_boxes_by_frame: list[np.ndarray],
                        reserve_clips_by_frame: list[np.ndarray],
                        frame_ids: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Group same-frame main/reserve views without deleting either view."""
    group_values, duplicate_values = [], []
    for main_boxes, main_clips, reserve_boxes, reserve_clips, frame in zip(
            main_boxes_by_frame, main_clips_by_frame, reserve_boxes_by_frame,
            reserve_clips_by_frame, frame_ids):
        main_boxes = np.asarray(main_boxes, np.float32).reshape(-1, 4)
        reserve_boxes = np.asarray(reserve_boxes, np.float32).reshape(-1, 4)
        main_clips = np.asarray(main_clips, np.float32).reshape(-1, 512)
        reserve_clips = np.asarray(reserve_clips, np.float32).reshape(-1, 512)
        base = (int(frame) + 1) * 1_000_000
        main_groups = np.arange(len(main_boxes), dtype=np.int64) + base + 1
        reserve_groups = np.arange(len(reserve_boxes), dtype=np.int64) + \
            base + 1 + len(main_boxes)
        reserve_duplicate = np.zeros(len(reserve_boxes), np.uint8)
        for ri, box in enumerate(reserve_boxes):
            best = (-1.0, -1)
            for mi, main_box in enumerate(main_boxes):
                overlap = box_iou(box, main_box)
                appearance = float(np.dot(
                    safe_normalize(reserve_clips[ri][None])[0],
                    safe_normalize(main_clips[mi][None])[0]))
                # The high-IoU condition is sufficient; the lower-IoU path
                # requires appearance agreement to avoid over-grouping.
                if overlap >= 0.50 or (overlap >= 0.30 and appearance >= 0.82):
                    score = overlap + 0.10 * max(0.0, appearance)
                    if score > best[0]:
                        best = (score, mi)
            if best[1] >= 0:
                reserve_groups[ri] = main_groups[best[1]]
                reserve_duplicate[ri] = 1
        groups = np.concatenate((main_groups, reserve_groups), axis=0)
        duplicates = np.concatenate((np.zeros(len(main_boxes), np.uint8),
                                     reserve_duplicate), axis=0)
        group_values.append(groups)
        duplicate_values.append(duplicates)
    return group_values, duplicate_values
