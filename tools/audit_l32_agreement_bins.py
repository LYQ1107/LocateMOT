#!/usr/bin/env python3
"""Stage L32 read-only agreement/continuity audit.

This audit replays the frozen L29 current-membership head and the frozen L30
fragment probe.  It does not fit a fusion weight or threshold and does not
write to any historical output.  The emitted score cache is deliberately
row-oriented so the later train-only gate can consume exactly the same
candidate/frame contract without another backbone pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.eval_l31_bounded_identity_fusion import (
    ASSOC_CHECKPOINT,
    BANK_ROOT,
    MEMBERSHIP_CHECKPOINT,
    SCORE_ROOT,
    build_bank,
    build_seq,
    feature_np,
    valid_tracks,
)
from tools.train_l28_track_set_decoder import state_at

MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
OUT_ROOT = ROOT / "outputs/l32/audit/agreement_bins"


def _bucket(value: float, cuts: tuple[float, ...], names: tuple[str, ...]) -> str:
    return names[int(np.searchsorted(np.asarray(cuts), value, side="right"))]


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.float32)
    rank[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return rank


def _add_bin(counter: Counter, name: str, labels: np.ndarray) -> None:
    for value in labels.tolist():
        counter[f"{name}={value}"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    parser.add_argument("--screen-cap", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {out}")
    out.mkdir(parents=True)
    started = time.time()

    entries = make_entries()
    if len([x for x in entries if x["split"] == "calibration"]) != 64 or len(
        [x for x in entries if x["split"] == "screening"]
    ) != 96:
        raise AssertionError("fixed manifest must contain 64 calibration and 96 screening queries")
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden = text["token_hidden"]
    mask = text["attention_mask"].bool()
    del text
    caches = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    assoc_state = torch.load(ASSOC_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    assoc_w = assoc_state["linear.weight"].numpy().reshape(-1)
    assoc_b = float(assoc_state["linear.bias"].item())
    device = torch.device(args.device)
    model = L29FrameMembershipSetDecoder().to(device)
    model.load_state_dict(torch.load(MEMBERSHIP_CHECKPOINT, map_location=device, weights_only=False)["model"])
    model.eval()

    entries_by_video: dict[str, list[dict]] = defaultdict(list)
    grouped: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    screen_units: list[tuple[str, str, int]] = []
    for entry in entries:
        video, expression = str(entry["video"]), str(entry["expression"])
        data = caches[(video, expression)]
        groups = {int(frame): idx for frame, idx in frame_groups(data)}
        grouped[(video, expression)] = groups
        entries_by_video[video].append(entry)
        if entry["split"] == "screening":
            screen_units.extend((video, expression, int(frame)) for frame in groups)
    screen_units.sort()
    selected = {
        screen_units[int(i)]
        for i in np.linspace(0, len(screen_units) - 1, min(args.screen_cap, len(screen_units)), dtype=int)
    }

    # Lists are compact row provenance.  GT labels are retained only for the
    # audit/report; they never affect score construction or gate fitting.
    columns: dict[str, list] = defaultdict(list)
    bins = Counter()
    split_units = Counter()
    split_rows = Counter()
    source_rows = Counter()
    audit_keys: set[tuple] = set()
    duplicate_keys = 0
    missing_assoc = 0
    nonfinite = 0
    representative: list[dict] = []
    unit_id = 0

    for video, video_entries in sorted(entries_by_video.items()):
        bank_meta = build_bank(video, assoc_w, assoc_b)
        seq = build_seq(video)
        frames_to_process: set[int] = set()
        for entry in video_entries:
            key = (video, str(entry["expression"]))
            groups = grouped[key]
            if entry["split"] == "calibration":
                frames_to_process.update(groups)
            else:
                frames_to_process.update(frame for frame in groups if (video, key[1], frame) in selected)

        for frame in sorted(frames_to_process):
            obs, obs_mask, obs_time, _selected_gt, selected_frames = state_at(seq, frame)
            with torch.inference_mode():
                encoded = model.encode_observations(obs.to(device), obs_mask.to(device), obs_time.to(device))
            active = valid_tracks(seq, frame)
            latest_by_track: dict[int, tuple[int, float, float, float]] = {}
            for i, track_index in enumerate(active):
                valid_history = torch.nonzero(obs_mask[i], as_tuple=False).flatten()
                if not len(valid_history):
                    continue
                j = int(valid_history[-1])
                latest_frame = int(selected_frames[i][j])
                feat = obs[i, j].float().numpy()
                latest_by_track[int(seq["track_ids"][track_index])] = (
                    latest_frame,
                    float(np.linalg.norm(feat[1415:1423])),
                    float(np.linalg.norm(feat[1423:1431])),
                    float(frame - latest_frame),
                )
            with torch.inference_mode():
                for entry in video_entries:
                    expression = str(entry["expression"])
                    groups = grouped[(video, expression)]
                    if frame not in groups:
                        continue
                    if entry["split"] == "screening" and (video, expression, frame) not in selected:
                        continue
                    qh, qm = hidden[int(entry["query_index"])].to(device), mask[int(entry["query_index"])].to(device)
                    output = model.forward_encoded(encoded, encoded[1], qh, qm)
                    current = output["current_membership_logits"].float().cpu().numpy()
                    membership_by_track = {
                        int(seq["track_ids"][track_index]): float(current[i])
                        for i, track_index in enumerate(active)
                    }
                    data = caches[(video, expression)]
                    rows = groups[frame]
                    tracks = data["track_id"][rows].astype(np.int64)
                    sources = data["source"][rows].astype(np.int8)
                    labels = data["label"][rows].astype(np.uint8)
                    assoc = np.asarray(
                        [bank_meta["lookup"].get((int(frame), int(track), int(source)), np.nan)
                         for track, source in zip(tracks, sources)], dtype=np.float32
                    )
                    missing_assoc += int(np.isnan(assoc).sum())
                    assoc = np.nan_to_num(assoc, nan=0.0)
                    raw = np.asarray([membership_by_track.get(int(track), -20.0) for track in tracks], dtype=np.float32)
                    recency = np.asarray([latest_by_track.get(int(track), (int(frame), 0.0, 0.0, 0.0))[3] for track in tracks], dtype=np.float32)
                    motion_norm = np.asarray([latest_by_track.get(int(track), (int(frame), 0.0, 0.0, 0.0))[1] for track in tracks], dtype=np.float32)
                    lifecycle_norm = np.asarray([latest_by_track.get(int(track), (int(frame), 0.0, 0.0, 0.0))[2] for track in tracks], dtype=np.float32)
                    raw_rank, assoc_rank = _rank_fraction(raw), _rank_fraction(assoc)
                    rank_gap = np.abs(raw_rank - assoc_rank)
                    agreement = np.where(rank_gap <= 0.25, "agree", np.where(rank_gap >= 0.50, "disagree", "mixed"))
                    recency_bucket = np.asarray([_bucket(x, (0.5, 1.5, 3.5), ("current", "lag1", "lag2_3", "lag4plus")) for x in recency])
                    motion_bucket = np.asarray([_bucket(x, (0.25, 0.75), ("low", "mid", "high")) for x in motion_norm])
                    lifecycle_bucket = np.asarray([_bucket(x, (0.25, 0.75), ("low", "mid", "high")) for x in lifecycle_norm])
                    _add_bin(bins, "agreement", agreement)
                    _add_bin(bins, "association_sign", np.where(assoc >= 0.0, "nonnegative", "negative"))
                    _add_bin(bins, "recency", recency_bucket)
                    _add_bin(bins, "motion_norm", motion_bucket)
                    _add_bin(bins, "lifecycle_norm", lifecycle_bucket)
                    _add_bin(bins, "source", np.where(sources == 0, "main", "reserve"))
                    split_units[entry["split"]] += 1
                    split_rows[entry["split"]] += len(rows)
                    source_rows[f"{entry['split']}_{'main'}"] += int(np.count_nonzero(sources == 0))
                    source_rows[f"{entry['split']}_{'reserve'}"] += int(np.count_nonzero(sources == 1))
                    if not (np.isfinite(raw).all() and np.isfinite(assoc).all() and np.isfinite(recency).all()):
                        nonfinite += 1
                    for local, (track, source) in enumerate(zip(tracks.tolist(), sources.tolist())):
                        key = (video, int(entry["query_index"]), int(frame), int(track), int(local))
                        duplicate_keys += int(key in audit_keys)
                        audit_keys.add(key)
                        columns["video"].append(int(video))
                        columns["query_index"].append(int(entry["query_index"]))
                        columns["frame"].append(int(frame))
                        columns["track_id"].append(int(track))
                        columns["source"].append(int(source))
                        columns["observation"].append(int(local))
                        columns["split"].append(0 if entry["split"] == "calibration" else 1)
                        columns["unit_id"].append(unit_id)
                        columns["membership"].append(float(raw[local]))
                        columns["association"].append(float(assoc[local]))
                        columns["recency"].append(float(recency[local]))
                        columns["motion_norm"].append(float(motion_norm[local]))
                        columns["lifecycle_norm"].append(float(lifecycle_norm[local]))
                        columns["membership_rank"].append(float(raw_rank[local]))
                        columns["association_rank"].append(float(assoc_rank[local]))
                        columns["agreement_code"].append(0 if agreement[local] == "agree" else 1 if agreement[local] == "mixed" else 2)
                        columns["label"].append(int(labels[local]))
                    if labels.any() and len(labels) > 1 and len(representative) < 80:
                        negative = np.flatnonzero(~labels.astype(bool))
                        if len(negative):
                            hard = int(negative[np.argmax(raw[negative])])
                            positive = int(np.flatnonzero(labels)[np.argmin(raw[labels.astype(bool)])])
                            representative.append({
                                "split": entry["split"], "video": video,
                                "query_index": int(entry["query_index"]), "frame": int(frame),
                                "positive_track": int(tracks[positive]), "hard_track": int(tracks[hard]),
                                "positive_membership": float(raw[positive]), "hard_membership": float(raw[hard]),
                                "positive_association": float(assoc[positive]), "hard_association": float(assoc[hard]),
                                "positive_recency": float(recency[positive]), "hard_recency": float(recency[hard]),
                                "positive_motion_norm": float(motion_norm[positive]), "hard_motion_norm": float(motion_norm[hard]),
                                "agreement": str(agreement[hard]),
                            })
                    unit_id += 1
        del bank_meta, seq

    arrays = {key: np.asarray(value) for key, value in columns.items()}
    np.savez_compressed(out / "score_cache.npz", **arrays)
    manifest_sha = __import__("hashlib").sha256(MANIFEST.read_bytes()).hexdigest()
    metadata = {
        "format": "locatemot-l32-agreement-audit-v1",
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": manifest_sha,
        "calibration_queries": 64, "screening_queries": 96,
        "screening_units_requested": int(args.screen_cap),
        "calibration_frame_units": int(split_units["calibration"]),
        "screening_frame_units": int(split_units["screening"]),
        "rows": int(len(arrays["label"])), "duplicate_keys": int(duplicate_keys),
        "nonfinite_units": int(nonfinite), "missing_association_rows": int(missing_assoc),
        "row_key": ["video", "query_index", "frame", "track_id", "observation"],
        "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
        "screening_gt_used_for_model_or_selection": False,
        "membership_checkpoint": str(MEMBERSHIP_CHECKPOINT.resolve()),
        "association_checkpoint": str(ASSOC_CHECKPOINT.resolve()),
        "bank_root": str(BANK_ROOT.resolve()), "cache_root": str(SCORE_ROOT.resolve()),
        "agreement_definition": "rank_fraction absolute gap <=0.25 agree, >=0.50 disagree; no source/pool condition",
        "continuity_definition": "frame minus latest available observation frame; motion/lifecycle are norms of frozen feature slices",
        "bins": dict(sorted(bins.items())), "source_rows": dict(sorted(source_rows.items())),
        "representative_cases": representative,
        "elapsed_sec": time.time() - started,
    }
    (out / "agreement_bins.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "README.md").write_text(
        "# L32 agreement audit\n\n"
        "Read-only replay of frozen L29 current membership and frozen L30 association. "
        "The `score_cache.npz` rows preserve query/video/frame isolation. Screening labels "
        "are retained only for post-hoc audit reporting; no screening label selected a model, "
        "gate, weight, or threshold. Semantic score construction excludes pool/source/group/state.\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
