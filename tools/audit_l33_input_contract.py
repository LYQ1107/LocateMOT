#!/usr/bin/env python3
"""Stage L33 train-only input/label contract audit; no model or GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.train_l28_track_set_decoder import CACHE_ROOT, build_queries

MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gt_set(value):
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(x) for x in value}
    return {str(value)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out); out = out if out.is_absolute() else ROOT / out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    queries = build_queries()
    if len(queries) != 7757 or len({(q["video"], q["expression"]) for q in queries}) != len(queries):
        raise AssertionError("train query uniqueness/count contract failed")
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden, attention = text["token_hidden"], text["attention_mask"].bool()
    query_indices = [int(q["text_index"]) for q in queries]
    text_finite = bool(torch.isfinite(hidden[query_indices]).all())
    mask_nonempty = int(attention[query_indices].any(1).sum())
    del text

    by_video = defaultdict(list)
    for query in queries: by_video[query["video"]].append(query)
    counters = Counter(); pair_counts = Counter(); duplicate = 0; finite_failures = 0
    observed_queries = set(); key_seen = set(); query_examples = []
    class_fields = set(); max_candidates = 0
    for video, video_queries in sorted(by_video.items()):
        cache_path = CACHE_ROOT / f"{video}.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        frames = cache["obs_frame"].numpy().astype(np.int64)
        tracks = cache["track_ids"].numpy().astype(np.int64)
        ptr = cache["track_ptr"].numpy().astype(np.int64)
        obs_gt = cache["obs_gt_ids"]
        obs_source = cache["obs_source"].numpy().astype(np.int64)
        # The full pair count is audited for every unit, while appearance
        # similarity is only materialized for a bounded representative set.
        # Reusing this array avoids re-converting the complete feature bank for
        # every query/frame/positive.
        features = cache["obs_features"].float().numpy()
        if len(obs_gt) != len(frames) or not torch.isfinite(cache["obs_features"].float()).all():
            finite_failures += 1
        frame_rows = defaultdict(list)
        for row, frame in enumerate(frames.tolist()): frame_rows[int(frame)].append(row)
        for query in video_queries:
            qkey = (video, query["expression"]); observed_queries.add(qkey)
            targets = {int(frame): {str(x) for x in ids} for frame, ids in query["target"].items()}
            for frame, rows in sorted(frame_rows.items()):
                labels = np.asarray([bool(gt_set(obs_gt[row]) & targets.get(frame, set())) for row in rows], bool)
                pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
                unit_key = (video, int(query["text_index"]), int(frame))
                duplicate += int(unit_key in key_seen); key_seen.add(unit_key)
                counters["frame_units"] += 1; counters["candidate_rows"] += len(rows)
                counters["positive_rows"] += int(labels.sum())
                counters["null_units"] += int(not labels.any())
                counters["inactive_units"] += int(not labels.any() and all(obs_gt[row] is None for row in rows))
                counters["multi_positive_units"] += int(len(pos) > 1)
                if len(pos) > 1:
                    pair_counts["multi_positive_pairs"] += len(pos) * (len(pos) - 1) // 2
                if len(pos) and len(neg):
                    pair_counts["same_frame_negative_pairs"] += len(pos) * len(neg)
                    # Frozen CLIP is used only to define bounded, auditable
                    # hard-case examples. It is not a training score or model
                    # choice. Pair counts above still include every pair.
                    pair_counts["appearance_hard_negative_examples"] += len(pos)
                    for positive in pos.tolist():
                        if len(query_examples) >= 12:
                            break
                        p_row = rows[positive]; n_rows = np.asarray([rows[x] for x in neg], np.int64)
                        p = features[p_row, :512]; n = features[n_rows, :512]
                        sim = (n @ p) / (np.linalg.norm(n, axis=1) * max(1e-6, np.linalg.norm(p)))
                        hard = int(neg[int(np.argmax(sim))])
                        query_examples.append({"video": video, "query_index": int(query["text_index"]), "frame": int(frame),
                                               "positive_track": int(tracks[positive]), "hard_track": int(tracks[hard]),
                                               "appearance_cosine": float(np.max(sim)), "positive_gt": sorted(gt_set(obs_gt[rows[positive]])),
                                               "hard_gt": sorted(gt_set(obs_gt[rows[hard]])), "same_class_label": None})
                max_candidates = max(max_candidates, len(rows))
        del cache

    payload = {
        "format": "locatemot-l33-input-contract-v1", "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha(MANIFEST),
        "train_only": True, "train_videos": len(by_video), "train_queries": len(queries),
        "unique_query_keys": len(observed_queries), "text_hidden_shape": list(hidden.shape),
        "text_attention_shape": list(attention.shape), "text_finite": text_finite,
        "nonempty_attention_queries": mask_nonempty, "cache_root": str(CACHE_ROOT.resolve()),
        "cache_manifest_sha256": sha(CACHE_ROOT / "manifest.json"), "counts": dict(counters),
        "pair_counts": dict(pair_counts), "max_candidates_per_frame": max_candidates,
        "duplicate_video_query_frame_keys": duplicate, "finite_failures": finite_failures,
        "row_key": ["video", "query", "frame", "track_or_fragment", "observation"],
        "comparison_scope": "pairs are constructed only within one video/query/frame; no cross-query or cross-frame comparison",
        "same_class_label_available": False,
        "same_class_note": "train cache exposes GT IDs/boxes but no verified class field; same-frame negatives and frozen-appearance hard examples are reported without claiming class labels",
        "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
        "screening_labels_read": False, "gt_used_for": "train-side supervision/audit only",
        "representative_pairs": query_examples,
    }
    (out / "input_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "README.md").write_text(
        "# L33 input contract audit\n\n"
        "This is a CPU-only, train-only audit over the 15-video L28 sequence cache. "
        "Every pair is confined to one `(video, query, frame)` unit. The cache has no "
        "verified object-class field, so same-frame negatives are not mislabeled as "
        "same-class examples. Screening labels were not read.\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
