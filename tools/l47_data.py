"""Shared read-only data helpers for the L47 output-contract probe."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.audit_l28_identity_bank import load_labels  # noqa: E402
from tools.train_l26_crossmodal_adapter import FAST, SPLIT, V5, load_expressions  # noqa: E402
from tools.train_l28_track_set_decoder import state_at  # noqa: E402

L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
FROZEN_L29_THRESHOLD = -1.1392689042308812
ALL_TRAIN_VIDEOS = (
    "0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009",
    "0010", "0012", "0014", "0015", "0016", "0017", "0020",
)
FIT_VIDEOS = ALL_TRAIN_VIDEOS[:12]
VAL_VIDEOS = ALL_TRAIN_VIDEOS[12:]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_bank(video: str):
    path = L19 / f"{video}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tensors = blob["tensors"]
    labels, label_path = load_labels(path, int(tensors["track_id"].numel()), tensors=tensors)
    return {
        "path": path,
        "box": tensors["box"].float(),
        "frame": tensors["frame"].long(),
        "candidate_index": tensors["candidate_index"].long(),
        "track": tensors["track_id"].long(),
        "pool": tensors["pool_id"].long(),
        "objectness": tensors["objectness"].float(),
        "clip": tensors["clip"].float(),
        "history_clip": tensors["history_clip"].float(),
        "uidm_h": tensors["uidm_h"].float(),
        "geometry": tensors["geometry"].float(),
        "motion": tensors["motion"].float(),
        "context": tensors["context"].float(),
        "lifecycle": tensors["lifecycle"].float(),
        "frame_ids": tensors["frame_ids"].long(),
        "ptr": tensors["frame_ptr"].long(),
        "labels": labels,
        "label_path": str(label_path),
    }


def load_queries():
    train = {str(x) for x in json.loads(SPLIT.read_text())["kitti_v2"]["train"]}
    text_manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
                  for x in text_manifest}
    result = []
    for row in load_expressions():
        video = str(row["video"])
        expression = str(row["expression"])
        key = (video, expression)
        if video not in train or key not in text_index:
            continue
        result.append({
            "video": video,
            "expression": expression,
            "sentence": str(row.get("sentence", expression)),
            "query_index": int(text_index[key]),
            "target": {int(k): {str(x) for x in values}
                       for k, values in row.get("label", {}).items()},
        })
    if len(result) != 7757:
        raise AssertionError(f"expected 7757 train expressions, found {len(result)}")
    return result


def load_text():
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    # Keep the frozen 9,778 x 64 x 768 cache in its stored half precision;
    # both L29 and L47 cast only the selected expression to float for compute.
    return text["token_hidden"].to(dtype=torch.float16), text["attention_mask"].bool()


def load_l29(device):
    from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
    model = L29FrameMembershipSetDecoder().to(device)
    state = torch.load(L29, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def build_l19_cache(bank):
    """Build the existing 1432-D L28-compatible cache in memory only."""
    from collections import defaultdict
    track_ids = bank["track"].numpy().astype(np.int64)
    frames = bank["frame"].numpy().astype(np.int32)
    by_track = defaultdict(list)
    for row, track in enumerate(track_ids.tolist()):
        by_track[int(track)].append(row)
    ordered = []
    ptr = [0]
    for track in sorted(by_track):
        rows = sorted(by_track[track], key=lambda row: (int(frames[row]), row))
        ordered.extend(rows)
        ptr.append(ptr[-1] + len(rows))
    order = torch.as_tensor(np.asarray(ordered, dtype=np.int64))
    count = len(track_ids)
    feature = torch.cat([
        bank[name].float().reshape(count, -1)
        for name in ("clip", "history_clip", "uidm_h", "geometry",
                     "motion", "lifecycle", "objectness")
    ], dim=1).half()
    return {
        "track_ids": torch.as_tensor(np.asarray(sorted(by_track), dtype=np.int64)),
        "track_ptr": torch.as_tensor(np.asarray(ptr, dtype=np.int64)),
        "obs_features": feature[order].contiguous(),
        "obs_frame": torch.as_tensor(frames[order.numpy()], dtype=torch.int32),
        "obs_gt_ids": [None] * len(ordered),
    }


def valid_track_indices(cache, cutoff: int):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def unit_labels(bank, query, frame_index):
    begin, end = int(bank["ptr"][frame_index]), int(bank["ptr"][frame_index + 1])
    frame = int(bank["frame_ids"][frame_index])
    target = query["target"].get(frame, set())
    labels = np.asarray([
        bank["labels"][row] is not None and str(bank["labels"][row]) in target
        for row in range(begin, end)
    ], dtype=bool)
    return frame, begin, end, target, labels


def numeric_for(bank, begin: int, end: int):
    rows = slice(begin, end)
    # 7 geometry + 8 motion + 8 lifecycle + 8 context + 1 objectness = 32.
    return torch.cat((
        bank["geometry"][rows], bank["motion"][rows],
        bank["lifecycle"][rows], bank["context"][rows],
        bank["objectness"][rows, None],
    ), dim=1).float()


def teacher_scores(teacher, cache, bank, query, frame: int, begin: int, end,
                   text_hidden, text_mask):
    obs, obs_mask, obs_time, _, _ = state_at(cache, frame, history=8)
    teacher_device = next(teacher.parameters()).device
    with torch.inference_mode():
        encoded = teacher.encode_observations(
            obs.to(teacher_device), obs_mask.to(teacher_device),
            obs_time.to(teacher_device),
        )
        out = teacher.forward_encoded(
            encoded, encoded[1], text_hidden[query["query_index"]].to(teacher_device),
            text_mask[query["query_index"]].to(teacher_device),
        )
    valid = valid_track_indices(cache, frame)
    values = {
        int(track): float(score)
        for track, score in zip(cache["track_ids"][valid].tolist(),
                                out["current_membership_logits"].float().tolist())
    }
    result = torch.tensor([
        values.get(int(track), float("nan"))
        for track in bank["track"][begin:end].tolist()
    ], dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError(f"L29 row mapping failed for {query['video']}/{frame}")
    return result


def hard_indices(labels, objectness, teacher_score, prelimit=96, hard_limit=24):
    negative = np.flatnonzero(~np.asarray(labels, dtype=bool))
    if not len(negative):
        return np.empty(0, dtype=np.int64)
    pre = negative[np.argsort(-np.asarray(objectness)[negative], kind="stable")[:prelimit]]
    return pre[np.argsort(-np.asarray(teacher_score)[pre], kind="stable")[:hard_limit]]


def pair_contract(labels, teacher_score, hard):
    positive = np.flatnonzero(np.asarray(labels, dtype=bool))
    hard = np.asarray(hard, dtype=np.int64)
    if not len(positive) or not len(hard):
        return (torch.empty((0, 2), dtype=torch.long),
                torch.empty((0, 2), dtype=torch.long))
    pairs = np.asarray([(int(p), int(n)) for p in positive for n in hard], dtype=np.int64)
    correct = np.asarray(teacher_score)[pairs[:, 0]] > np.asarray(teacher_score)[pairs[:, 1]]
    return torch.as_tensor(pairs[correct], dtype=torch.long), torch.as_tensor(pairs[~correct], dtype=torch.long)


def build_unit(query, frame_index, bank, cache, teacher, text_hidden, text_mask):
    frame, begin, end, target, labels = unit_labels(bank, query, frame_index)
    teacher_score = teacher_scores(teacher, cache, bank, query, frame, begin, end,
                                   text_hidden, text_mask)
    hard = hard_indices(labels, bank["objectness"][begin:end].numpy(), teacher_score.numpy())
    correct, error = pair_contract(labels, teacher_score.numpy(), hard)
    return {
        "query": query,
        "frame": frame,
        "begin": begin,
        "end": end,
        "clip": bank["clip"][begin:end].float().contiguous(),
        "history_clip": bank["history_clip"][begin:end].float().contiguous(),
        "numeric": numeric_for(bank, begin, end).contiguous(),
        "teacher": teacher_score.contiguous(),
        "labels": torch.as_tensor(labels, dtype=torch.bool),
        "objectness": bank["objectness"][begin:end].float().contiguous(),
        "hard_indices": torch.as_tensor(hard, dtype=torch.long),
        "teacher_correct_pairs": correct,
        "teacher_error_pairs": error,
        "target_ids": sorted(str(x) for x in target),
        "video": query["video"],
        "expression": query["expression"],
        "query_index": int(query["query_index"]),
    }


def category(labels, target):
    return (
        "multi_positive" if int(np.asarray(labels).sum()) > 1 else
        "positive" if bool(np.asarray(labels).any()) else
        "inactive" if not target else "other"
    )


def smoke_refs(queries, banks, videos, limit=32):
    """Deterministic category-diverse refs spanning at least eight videos."""
    by_video = defaultdict(list)
    for query in queries:
        if query["video"] in videos:
            by_video[query["video"]].append(query)
    selected = []
    seen = set()
    # First guarantee two units per video (usually one inactive and one
    # positive/multi-positive) so a smoke cannot repeat L46's single-video fit.
    for video in videos:
        local = {name: [] for name in ("multi_positive", "positive", "inactive", "other")}
        for query in by_video[video]:
            bank = banks[video]
            for fi in range(len(bank["frame_ids"])):
                frame, _, _, target, labels = unit_labels(bank, query, fi)
                key = (video, int(query["query_index"]), frame)
                if key in seen:
                    continue
                name = category(labels, target)
                if len(local[name]) < 2:
                    local[name].append((query, fi, labels))
                if all(len(local[name]) >= 2 for name in local):
                    break
            if all(len(local[name]) >= 2 for name in local):
                break
        # Keep one target-bearing and one no-target/other unit per video when
        # possible, then fill the global cap deterministically.  This retains
        # the required multi-positive and inactive regression coverage.
        local_items = []
        target_items = local["multi_positive"] + local["positive"]
        no_target_items = local["inactive"] + local["other"]
        if target_items:
            local_items.append(target_items[0])
        if no_target_items:
            local_items.append(no_target_items[0])
        for item in (target_items[1:] + no_target_items[1:]):
            if len(local_items) >= 2:
                break
            local_items.append(item)
        if len(local_items) < 2:
            for name in ("multi_positive", "positive", "inactive", "other"):
                local_items.extend(local[name])
                if len(local_items) >= 2:
                    break
        for item in local_items[:2]:
            key = (video, int(item[0]["query_index"]),
                   int(banks[video]["frame_ids"][item[1]]))
            if key not in seen:
                selected.append(item)
                seen.add(key)
        if (len(selected) >= limit and
                len({item[0]["video"] for item in selected}) >= 8):
            break
    # Fill category quotas from the deterministic global order.
    if len(selected) < limit:
        for video in videos:
            bank = banks[video]
            for query in by_video[video]:
                for fi in range(len(bank["frame_ids"])):
                    frame, _, _, target, labels = unit_labels(bank, query, fi)
                    key = (video, int(query["query_index"]), frame)
                    if key in seen:
                        continue
                    selected.append((query, fi, labels)); seen.add(key)
                    if len(selected) >= limit:
                        return selected
    return selected[:limit]


def evenly_spaced_refs(queries, banks, videos, limit):
    """Select refs by structural order only, without inspecting labels."""
    qv = defaultdict(list)
    for query in queries:
        if query["video"] in videos:
            qv[query["video"]].append(query)
    total = sum(len(qv[v]) * len(banks[v]["frame_ids"]) for v in videos)
    if total <= 0:
        return []
    count = min(int(limit), total)
    wanted = set(np.linspace(0, total - 1, count, dtype=np.int64).tolist())
    result = []
    cursor = 0
    for video in videos:
        bank = banks[video]
        for query in qv[video]:
            for fi in range(len(bank["frame_ids"])):
                if cursor in wanted:
                    result.append((query, fi))
                cursor += 1
    if len(result) != count:
        raise AssertionError(f"structural sampling mismatch {len(result)} != {count}")
    return result


def fast_entries():
    """Read manifest membership only; GT labels remain outside this helper."""
    manifest = json.loads(FAST.read_text())
    expressions = {(str(x["video"]), str(x["expression"])): x
                   for x in load_expressions()}
    result = []
    for row in sorted(manifest["queries"], key=lambda x: int(x["query_index"])):
        key = (str(row["video"]), str(row["expression"]))
        source = expressions[key]
        result.append({
            "video": key[0], "expression": key[1],
            "query_index": int(row["query_index"]),
            "split": str(row["split"]),
            "target": {int(k): {str(x) for x in values}
                       for k, values in source.get("label", {}).items()},
        })
    if len([x for x in result if x["split"] == "calibration"]) != 64:
        raise AssertionError("fast calibration query count is not 64")
    if len([x for x in result if x["split"] == "screening"]) != 96:
        raise AssertionError("fast screening query count is not 96")
    return result


def fast_refs(entries, banks, split="screening", cap=100):
    refs = []
    for entry in entries:
        if entry["split"] != split:
            continue
        bank = banks[entry["video"]]
        refs.extend((entry, fi) for fi in range(len(bank["frame_ids"])))
    # Match the immutable L27/L46 fixed-slice contract exactly: selection is
    # made after lexicographic (video, expression, frame) ordering, not by
    # manifest query_index ordering.
    refs.sort(key=lambda item: (
        str(item[0]["video"]), str(item[0]["expression"]),
        int(banks[item[0]["video"]]["frame_ids"][item[1]]),
    ))
    count = min(int(cap), len(refs))
    wanted = set(np.linspace(0, len(refs) - 1, count, dtype=np.int64).tolist())
    return [ref for index, ref in enumerate(refs) if index in wanted]
