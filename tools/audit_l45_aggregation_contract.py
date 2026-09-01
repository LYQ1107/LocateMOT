#!/usr/bin/env python3
"""Recompute and audit the frozen L43 pair residual on the L45 replay slice.

This is an audit/replay only.  It does not train, choose a model, run
TrackEval, or write a bank.  The pair matrices are retained in a compact
replay cache so the subsequent CPU aggregation probe does not repeat the
streaming image encoder or L29/L43 forwards.
"""
from __future__ import annotations

import argparse
import hashlib
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
from locatemot.models.l43_teacher_anchored_pairwise import L43TeacherAnchoredPairwiseResidual
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import StreamingCropPatchEncoder, numeric_for
from tools.train_l44_integrated_query_region_track_decoder import history_for
from tools.l40_raw_data import WEIGHTS

L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L43 = ROOT / "outputs/l43/train/teacher_anchored_smoke100/checkpoint_l43_teacher_anchored_step100.pt"
SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L43_AUDIT = ROOT / "outputs/l43/audit/pairwise_residual_contract.json"
V1 = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def valid_track_indices(cache, cutoff: int):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_bank(video: str):
    d = torch.load(L19 / f"{video}.pt", map_location="cpu", weights_only=False)
    t = d["tensors"]
    return {
        "box": t["box"].float(), "frame": t["frame"].long(),
        "track": t["track_id"].long(), "objectness": t["objectness"].float(),
        "geometry": t["geometry"].float(), "motion": t["motion"].float(),
        "context": t["context"].float(), "lifecycle": t["lifecycle"].float(),
        "history_clip": t["history_clip"].float(),
        "frame_ids": t["frame_ids"].long(), "ptr": t["frame_ptr"].long(),
    }


def load_text(entries):
    manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
             for x in manifest}
    needed = sorted({index[(str(e["video"]), str(e["expression"]))]
                     for e in entries})
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden = text["token_hidden"][needed].float()
    mask = text["attention_mask"][needed].bool()
    remap = {old: i for i, old in enumerate(needed)}
    del text
    return index, hidden, mask, remap


def selected_units(entries, arrays, cap=100):
    available = []
    for entry in entries:
        if entry["split"] != "screening":
            continue
        data = arrays[(entry["video"], entry["expression"])]
        available.extend((str(entry["video"]), str(entry["expression"]), int(frame))
                        for frame, _ in frame_groups(data))
    available.sort()
    if not available:
        raise RuntimeError("fixed L27 cache contains no screening frame units")
    chosen = {available[i] for i in np.linspace(0, len(available) - 1,
                                                min(cap, len(available)), dtype=int)}
    if len(chosen) != min(cap, len(available)):
        raise AssertionError("fixed screening unit selection is not unique")
    needed = []
    for entry in entries:
        data = arrays[(entry["video"], entry["expression"])]
        for frame, _ in frame_groups(data):
            unit = (str(entry["video"]), str(entry["expression"]), int(frame))
            if entry["split"] == "calibration" or unit in chosen:
                needed.append((unit, entry))
    return available, chosen, needed


def teacher_scores(model, cache, frame, qh, qm, bank, rows, device):
    obs, obs_mask, obs_time, _, _ = state_at(cache, int(frame), history=8)
    valid = valid_track_indices(cache, int(frame))
    valid_ids = cache["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
    with torch.inference_mode():
        encoded = model.encode_observations(obs.to(device), obs_mask.to(device), obs_time.to(device))
        output = model.forward_encoded(encoded, encoded[1], qh, qm)
    by_track = {int(track): float(score) for track, score in
                zip(valid_ids, output["current_membership_logits"].float().cpu().tolist())}
    values = np.asarray([by_track.get(int(bank["track"][row]), -20.0) for row in rows], np.float32)
    return values, encoded, by_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/l45/audit/aggregation_contract.json")
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    l43_audit = json.loads(L43_AUDIT.read_text())
    if l43_audit["train_video_count"] != 15 or l43_audit["expression_count"] != 7757:
        raise AssertionError("L43 train-side audit is not the expected 15-video/7757-query contract")
    if sha256(FAST) != V1:
        raise AssertionError("fixed manifest SHA changed")
    entries = make_entries()
    arrays = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    text_index, text_hidden, text_mask, text_remap = load_text(entries)
    available, chosen_screen, needed = selected_units(entries, arrays, args.cap)

    by_frame = defaultdict(list)
    for unit, entry in needed:
        by_frame[(unit[0], unit[2])].append((unit, entry))
    videos = sorted({str(entry["video"]) for _, entry in needed})
    banks = {video: load_bank(video) for video in videos}
    caches = {}
    cache_sources = {}
    for video in videos:
        cache_path = L28 / f"{video}.pt"
        if cache_path.exists():
            caches[video] = torch.load(cache_path, map_location="cpu", weights_only=False)
            cache_sources[video] = {"path": str(cache_path.resolve()), "kind": "L28_persistent_cache"}
        else:
            caches[video] = build_l19_sequence_cache(video)
            cache_sources[video] = {"path": str((L19 / f"{video}.pt").resolve()),
                                    "kind": "L19_read_only_in_memory_sequence_replay"}

    device = torch.device(args.device)
    teacher_model = L29FrameMembershipSetDecoder().to(device)
    teacher_model.load_state_dict(torch.load(L29, map_location=device,
                                              weights_only=False)["model"], strict=True)
    teacher_model.eval()
    l43_state = torch.load(L43, map_location=device, weights_only=False)
    l43_cfg = l43_state.get("config", {}).get("model_config",
                                               {"image_dim": 768, "text_dim": 768,
                                                "numeric_dim": 36, "hidden": 128,
                                                "heads": 4, "layers": 1})
    pair_model = L43TeacherAnchoredPairwiseResidual(**l43_cfg).to(device)
    pair_model.load_state_dict(l43_state["model"], strict=True)
    pair_model.eval()
    del l43_state
    encoder = StreamingCropPatchEncoder(device, batch_size=32)

    metadata = []
    candidate_chunks = []
    pair_chunks = []
    label_chunks = []
    objectness_chunks = []
    source_chunks = []
    track_chunks = []
    candidate_offset = 0
    pair_offset = 0
    stats = Counter()
    degree_values = []
    cancellation_values = []
    residual_sign_mixed = 0
    residual_sign_total = 0
    teacher_error_margins = []
    teacher_correct_margins = []
    hard_counts = Counter()
    pair_symmetry_max = 0.0
    residual_max = 0.0
    physical_frames = 0
    raw_crops = 0

    for (video, frame), frame_entries in sorted(by_frame.items()):
        bank = banks[video]
        frame_to_index = {int(value): index for index, value in
                          enumerate(bank["frame_ids"].tolist())}
        if int(frame) not in frame_to_index:
            raise RuntimeError(f"frame missing from L19 bank: {video}/{frame}")
        fi = frame_to_index[int(frame)]
        begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
        rows = list(range(begin, end))
        tracks = bank["track"][rows].numpy().astype(np.int64)
        if len(np.unique(tracks)) != len(tracks):
            raise RuntimeError(f"duplicate current-frame track IDs: {video}/{frame}")
        patches = encoder.encode(video, bank, rows)
        numeric = numeric_for(bank, rows)
        history, history_mask, history_time, _, _ = history_for(
            caches[video], bank, rows, int(frame), {"target": {}}, history_len=8)
        # L29 is encoded once per physical frame and queried once per expression.
        obs, obs_mask, obs_time, _, _ = state_at(caches[video], int(frame), history=8)
        valid = valid_track_indices(caches[video], int(frame))
        valid_ids = caches[video]["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
        with torch.inference_mode():
            encoded = teacher_model.encode_observations(obs.to(device), obs_mask.to(device), obs_time.to(device))
        physical_frames += 1
        raw_crops += len(rows)
        degree = max(0, len(rows) - 1)
        degree_values.append(degree)

        for unit, entry in frame_entries:
            data = arrays[(entry["video"], entry["expression"])]
            idx = np.flatnonzero(data["frame"] == int(frame))
            cache_tracks = data["track_id"][idx].astype(np.int64)
            cache_pos = {int(track): int(i) for i, track in enumerate(cache_tracks)}
            if set(cache_pos) != set(tracks):
                raise RuntimeError(f"pair cache/bank candidate mismatch: {video}/{entry['expression']}/{frame}")
            aligned = np.asarray([cache_pos[int(track)] for track in tracks], dtype=np.int64)
            labels = data["label"][idx][aligned].astype(bool)
            sources = data["source"][idx][aligned].astype(np.int8)
            old_text_index = text_index[(str(entry["video"]), str(entry["expression"]))]
            row = text_remap[old_text_index]
            qh = text_hidden[row].to(device)
            qm = text_mask[row].to(device)
            with torch.inference_mode():
                teacher_out = teacher_model.forward_encoded(encoded, encoded[1], qh, qm)
                teacher_map = {int(track): float(score) for track, score in
                               zip(valid_ids, teacher_out["current_membership_logits"].float().cpu().tolist())}
                teacher = np.asarray([teacher_map.get(int(track), -20.0) for track in tracks], np.float32)
                pair_out = pair_model(patches.to(device).float(), qh,
                                      numeric.to(device).float(),
                                      torch.from_numpy(teacher).to(device),
                                      torch.ones(len(rows), dtype=torch.bool, device=device), qm)
                residual = pair_out["residual"].float().cpu().numpy()
            if not np.isfinite(teacher).all() or not np.isfinite(residual).all():
                raise FloatingPointError(f"nonfinite replay values: {video}/{frame}")
            if residual.shape != (len(rows), len(rows)):
                raise RuntimeError("L43 residual matrix is not candidate-square")
            pair_symmetry_max = max(pair_symmetry_max,
                                    float(np.abs(residual + residual.T).max(initial=0.0)))
            residual_max = max(residual_max, float(np.abs(residual).max(initial=0.0)))
            mask = ~np.eye(len(rows), dtype=bool)
            pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
            stats["units"] += 1
            stats["candidate_rows"] += len(rows)
            stats["positive_rows"] += int(labels.sum())
            stats["unordered_pairs"] += len(rows) * max(0, len(rows) - 1) // 2
            stats["positive_negative_pairs"] += len(pos) * len(neg)
            if len(pos) > 1:
                stats["multi_positive_units"] += 1
                stats["multi_positive_pairs"] += len(pos) * (len(pos) - 1) // 2
            if not len(pos):
                stats["inactive_units"] += 1
            if len(pos) and len(neg):
                margins = teacher[pos, None] - teacher[neg][None, :]
                teacher_correct_margins.extend(margins[margins > 0].tolist())
                teacher_error_margins.extend(margins[margins <= 0].tolist())
                hard = np.flatnonzero(~labels)
                hard = hard[np.argsort(-teacher[hard], kind="stable")[:min(24, len(hard))]]
                hard_counts["pairs"] += len(pos) * len(hard)
                hard_counts["teacher_correct"] += int((teacher[pos, None] - teacher[hard][None, :] > 0).sum())
                hard_counts["teacher_error"] += int((teacher[pos, None] - teacher[hard][None, :] <= 0).sum())
            for i in range(len(rows)):
                values = residual[i, mask[i]]
                if not len(values):
                    continue
                residual_sign_total += 1
                if np.any(values > 0) and np.any(values < 0):
                    residual_sign_mixed += 1
                denom = float(np.abs(values).sum())
                if denom > 1e-12:
                    cancellation_values.append(float(1.0 - abs(float(values.sum())) / denom))

            cs, ce = candidate_offset, candidate_offset + len(rows)
            ps, pe = pair_offset, pair_offset + residual.size
            metadata.append({
                "video": video, "expression": str(entry["expression"]),
                "query_index": int(entry["query_index"]), "frame": int(frame),
                "split": str(entry["split"]), "candidate_count": len(rows),
                "candidate_start": cs, "candidate_end": ce,
                "pair_start": ps, "pair_end": pe,
            })
            candidate_chunks.append(teacher.astype(np.float32, copy=False))
            label_chunks.append(labels.astype(np.uint8, copy=False))
            objectness_chunks.append(bank["objectness"][rows].numpy().astype(np.float32))
            source_chunks.append(sources)
            track_chunks.append(tracks)
            pair_chunks.append(residual.reshape(-1).astype(np.float32, copy=False))
            candidate_offset = ce
            pair_offset = pe

        del patches, numeric, history, history_mask, history_time, encoded

    del encoder, pair_model, teacher_model, banks, caches
    replay_path = out.with_name("replay_cache.npz")
    metadata_path = out.with_name("replay_metadata.json")
    np.savez(
        replay_path,
        teacher=np.concatenate(candidate_chunks),
        labels=np.concatenate(label_chunks),
        objectness=np.concatenate(objectness_chunks),
        source=np.concatenate(source_chunks),
        track_id=np.concatenate(track_chunks),
        residual=np.concatenate(pair_chunks),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    def dist(values):
        if not values:
            return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None, "max": None}
        x = np.asarray(values, dtype=np.float64)
        return {"count": int(len(x)), "mean": float(x.mean()),
                "median": float(np.median(x)), "q10": float(np.quantile(x, .1)),
                "q90": float(np.quantile(x, .9)), "max": float(x.max())}

    replay_stats = {
        "screening_units_available": len(available),
        "screening_units_selected": len(chosen_screen),
        "physical_frame_units": physical_frames,
        "expression_frame_units": stats["units"],
        "candidate_rows": stats["candidate_rows"],
        "positive_rows": stats["positive_rows"],
        "unordered_pairs": stats["unordered_pairs"],
        "positive_negative_pairs": stats["positive_negative_pairs"],
        "multi_positive_units": stats["multi_positive_units"],
        "multi_positive_pairs": stats["multi_positive_pairs"],
        "inactive_units": stats["inactive_units"],
        "candidate_degree": dist(degree_values),
        "teacher_correct_margin": dist(teacher_correct_margins),
        "teacher_error_margin": dist(teacher_error_margins),
        "teacher_hard_subset": dict(hard_counts),
        "residual_bound_max_abs": residual_max,
        "residual_bound_satisfied": residual_max <= .05 + 1e-5,
        "pair_antisymmetry_max_abs": pair_symmetry_max,
        "pair_antisymmetry_satisfied": pair_symmetry_max <= 2e-5,
        "residual_sign_mixed_fraction": residual_sign_mixed / max(1, residual_sign_total),
        "residual_cancellation": dist(cancellation_values),
        "train_reference": {
            "source": str(L43_AUDIT.resolve()),
            "population": l43_audit["population_counts"],
            "smoke_pair_subset": l43_audit["sample_pair_audit"],
        },
    }
    payload = {
        "format": "locatemot-l45-degree-normalized-pair-aggregation-contract-v1",
        "stage": "L45-A-read-only-aggregation-audit",
        "project_root": str(ROOT), "started_at": started, "completed_at": time.time(),
        "manifest": str(FAST.resolve()), "manifest_sha256": sha256(FAST),
        "score_cache": str(SCORE_ROOT.resolve()),
        "score_cache_model": "A_C1_S2000 immutable candidate labels for calibration/final reporting",
        "teacher_checkpoint": str(L29.resolve()), "teacher_checkpoint_sha256": sha256(L29),
        "pair_checkpoint": str(L43.resolve()), "pair_checkpoint_sha256": sha256(L43),
        "image_encoder": {"weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS),
                          "pixel_storage": "transient RAM only"},
        "train_videos": l43_audit["train_videos"], "train_video_count": 15,
        "train_expression_count": 7757,
        "cache_sources": cache_sources,
        "replay_cache": str(replay_path.resolve()),
        "replay_metadata": str(metadata_path.resolve()),
        "replay_cache_sha256": sha256(replay_path),
        "replay_stats": replay_stats,
        "aggregation_contract": {
            "control_0": "s_i=m_i",
            "control_1": "s_i=m_i+mean_j(r_ij)",
            "probe_2": "w_ij=exp(-abs(m_i-m_j)/0.5)/sum_j; s_i=m_i+sum_j(w_ij*r_ij)",
            "probe_3": "center all off-diagonal r per frame, weighted aggregate with tau=0.5, subtract candidate-delta mean",
            "probe_4": "GT-privileged sign-corrected teacher-error diagnostic only",
            "threshold": "not selected in Stage A; the subsequent probe uses one L29 calibration-only threshold",
            "forbidden": ["top-k output filtering", "threshold re-search per probe", "NULL post-processing", "source/pool/group/state semantic input"],
        },
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"],
        "screening_gt_used_for_training": False,
        "screening_gt_used_for_structure_selection": False,
        "token_span_region_alignment": "UNALIGNED; not claimed",
        "motion_language_decomposition": "not claimed; no verified mask",
        "decision": "enter_untrained_aggregation_probe",
        "elapsed_sec": time.time() - started,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    report = ROOT / "reports" / "l45_aggregation_audit.md"
    report.write_text(
        "# Stage L45 — Aggregation contract audit\n\n"
        f"Decision: **{payload['decision']}**.\n\n"
        f"The replay covers {replay_stats['expression_frame_units']:,} expression-frame units, "
        f"{replay_stats['physical_frame_units']:,} physical frames, and "
        f"{replay_stats['candidate_rows']:,} candidate rows. It uses the fixed 100 screening "
        f"units plus all calibration units; screening labels are retained only for frozen "
        f"held-out reporting.\n\n"
        f"- L43 residual max abs: `{residual_max:.8f}`; bound satisfied: `{replay_stats['residual_bound_satisfied']}`\n"
        f"- antisymmetry max abs: `{pair_symmetry_max:.8g}`; satisfied: `{replay_stats['pair_antisymmetry_satisfied']}`\n"
        f"- candidate degree: `{replay_stats['candidate_degree']}`\n"
        f"- mixed-sign contributor fraction: `{replay_stats['residual_sign_mixed_fraction']:.6f}`\n"
        f"- cancellation: `{replay_stats['residual_cancellation']}`\n"
        f"- teacher-error margin: `{replay_stats['teacher_error_margin']}`\n"
        f"- teacher-hard subset: `{replay_stats['teacher_hard_subset']}`\n\n"
        "The train-side population and L43 smoke pair subset are read from the frozen L43 "
        "audit; no train or screening labels are used to fit a new parameter here. The "
        "replay matrices are stored in `outputs/l45/audit/replay_cache.npz` with unit/key "
        "metadata in `replay_metadata.json`. The following probe is the only place that "
        "will evaluate the registered aggregation formulas.\n"
    )
    print(json.dumps({"out": str(out), "replay_cache": str(replay_path),
                      "decision": payload["decision"], "replay_stats": replay_stats},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
