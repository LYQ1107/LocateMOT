#!/usr/bin/env python3
"""Run the frozen L49 checkpoint once on official Refer-KITTI V1 and V2.

The validation selection and calibration artifact is read before any official
GT is opened.  Predictions are generated from the frozen step-250 semantic
warm-up checkpoint and its calibration-only thresholds, then each official
seqmap is evaluated once by the unchanged RMOT TrackEval runner.  No test
result is fed back into model, threshold, checkpoint, or branch selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
REF = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import (  # noqa: E402
    L29FrameMembershipSetDecoder,
)
from locatemot.models.l49_kitti_rmot import L49KittiRMOT  # noqa: E402
from locatemot.rmot.l49_data import (  # noqa: E402
    L29_CHECKPOINT,
    KITTI_BANK,
    history_sequence,
    relation_features,
    sha256_file,
)
from tools.eval_l49_validation import source_masks  # noqa: E402
from tools.train_l28_track_set_decoder import state_at  # noqa: E402
from tools.train_l49_kitti_rmot import build_teacher_cache  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
SELECTED = ROOT / "outputs/l49/val/selected_checkpoint.json"
L48_TEXT = ROOT / "outputs/l48/data/text_cache.pt"
ROBERTA = Path("/home/lwr/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b")
EVAL_RUN = ROOT / "references/l8/TrackEval_rmot/scripts/run_mot_challenge.py"
IMG_ROOT = ROOT / "data/kitti_tracking_training/image_02"
DATASETS = {
    "refer_kitti_v1": {
        "short": "v1",
        "meta": ROOT / "outputs/l13/data/refer_kitti_v1/expressions.json",
        "seqmap": REF / "rmot_official/datasets/data_path/seqmap.txt",
        "gt_root": ROOT / "outputs/l13/data/refer_kitti_v1/gt_template",
    },
    "refer_kitti_v2": {
        "short": "v2",
        "meta": ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
        "seqmap": REF / "temp_rmot/datasets/data_path/seqmap.txt",
        "gt_root": ROOT / "outputs/l10/data/rmot_kitti/gt_template",
    },
}


def sha256(path: Path) -> str:
    return sha256_file(path)


def load_queries(dataset: str):
    cfg = DATASETS[dataset]
    meta = json.loads(cfg["meta"].read_text())
    queries = []
    for query_id, line in enumerate(x.strip() for x in cfg["seqmap"].read_text().splitlines()):
        if not line:
            continue
        video, expression = line.split("+", 1)
        entry = next((item for item in meta.get(video, [])
                      if str(item.get("expression")) == expression), None)
        if entry is None:
            raise KeyError(f"{dataset}: seqmap expression is absent from metadata: {line}")
        queries.append({
            "dataset": dataset, "query_id": int(query_id), "video": str(video),
            "expression": expression, "sentence": str(entry.get("sentence", expression)),
            "seqmap_line": line,
        })
    return queries


def load_unlabeled_bank(video: str) -> dict:
    path = KITTI_BANK / f"{video}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tensors = blob["tensors"]
    track_rows = defaultdict(list)
    for row, track in enumerate(tensors["track_id"].long().tolist()):
        track_rows[int(track)].append(int(row))
    return {
        "path": path, "metadata": blob.get("metadata", {}), "tensors": tensors,
        "track_rows": dict(track_rows),
    }


def build_test_text_cache(queries, out_path: Path):
    """Reuse frozen word states and encode only unseen test sentences locally."""
    frozen = torch.load(L48_TEXT, map_location="cpu", weights_only=False)
    frozen_by_sentence = {
        str(sentence): int(index)
        for sentence, index in frozen["sentence_to_index"].items()
    }
    sentences = sorted({query["sentence"] for query in queries})
    hidden_rows = []
    mask_rows = []
    reused = []
    missing = []
    for sentence in sentences:
        index = frozen_by_sentence.get(sentence)
        if index is None:
            missing.append(sentence)
            hidden_rows.append(None); mask_rows.append(None)
        else:
            reused.append(sentence)
            hidden_rows.append(frozen["token_hidden"][index].cpu().half())
            mask_rows.append(frozen["attention_mask"][index].cpu().bool())
    if missing:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA), local_files_only=True)
        text_model = AutoModel.from_pretrained(str(ROBERTA), local_files_only=True).eval()
        with torch.inference_mode():
            for start in range(0, len(missing), 64):
                batch_text = missing[start:start + 64]
                enc = tokenizer(batch_text, padding="max_length", truncation=True,
                                 max_length=64, return_tensors="pt")
                values = text_model(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"]).last_hidden_state
                for offset, sentence in enumerate(batch_text):
                    slot = sentences.index(sentence)
                    hidden_rows[slot] = values[offset].cpu().half()
                    mask_rows[slot] = enc["attention_mask"][offset].cpu().bool()
        del text_model
    hidden = torch.stack(hidden_rows).contiguous()
    masks = torch.stack(mask_rows).contiguous()
    if tuple(hidden.shape[1:]) != (64, 768) or tuple(masks.shape[1:]) != (64,):
        raise AssertionError(f"test text cache shape={tuple(hidden.shape)} {tuple(masks.shape)}")
    if not bool(torch.isfinite(hidden.float()).all()):
        raise FloatingPointError("test text cache contains nonfinite values")
    payload = {"sentences": sentences, "token_hidden": hidden,
               "attention_mask": masks,
               "sentence_to_index": {sentence: i for i, sentence in enumerate(sentences)}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path) + ".tmp")
    os.replace(str(out_path) + ".tmp", out_path)
    return {
        "path": str(out_path.resolve()), "sha256": sha256(out_path),
        "sentence_count": len(sentences), "shape": list(hidden.shape),
        "reused_l48_sentences": len(reused), "encoded_local_roberta": len(missing),
        "source_l48_text_cache": str(L48_TEXT.resolve()),
        "source_l48_text_cache_sha256": sha256(L48_TEXT),
        "roberta_snapshot": str(ROBERTA),
        "labels_read": False, "labels_used_for_prediction": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_language_mask": "UNALIGNED/not claimed",
    }


def frame_inputs(bank: dict, frame_index: int):
    tensors = bank["tensors"]
    begin = int(tensors["frame_ptr"][frame_index])
    end = int(tensors["frame_ptr"][frame_index + 1])
    image_size = bank["metadata"].get("image_size", [])
    sl = slice(begin, end)
    return {
        "begin": begin, "end": end,
        "frame_id": int(tensors["frame_ids"][frame_index]),
        "track_id": tensors["track_id"][sl].long(),
        "box": tensors["box"][sl].float(),
        "clip": tensors["clip"][sl].float(),
        "history_clip": tensors["history_clip"][sl].float(),
        "geometry": tensors["geometry"][sl].float(),
        "motion": tensors["motion"][sl].float(),
        "context": tensors["context"][sl].float(),
        "lifecycle": tensors["lifecycle"][sl].float(),
        "objectness": tensors["objectness"][sl].float().reshape(-1),
        "relation": relation_features(tensors["box"][sl], image_size),
    }


@torch.inference_mode()
def semantic_batch(semantic, values: dict[str, torch.Tensor], text_tokens, text_mask):
    """Vectorized equivalent of L48SemanticMatcher.forward for B queries."""
    clip = values["clip"]
    geometry = values["geometry"]
    motion = values["motion"]
    context = values["context"]
    lifecycle = values["lifecycle"]
    objectness = values["objectness"].reshape(-1)
    relation = values["relation"]
    appearance = semantic.appearance(torch.nan_to_num(clip.float()))
    geo = semantic.geometry(torch.nan_to_num(torch.cat((geometry.float(), relation.float()), -1)))
    motion_input = torch.cat((values["history_clip"].float(), motion.float(),
                              lifecycle.float(), context.float(), objectness[:, None]), -1)
    motion_stream = semantic.motion_identity(torch.nan_to_num(motion_input))
    candidate_base = (appearance + geo + motion_stream) / 3.0
    query = semantic.text(torch.nan_to_num(text_tokens.float()))
    batch = int(query.shape[0])
    candidate_query = candidate_base.unsqueeze(0).expand(batch, -1, -1)
    cross, _ = semantic.query_to_candidate(
        candidate_query, query, query,
        key_padding_mask=~text_mask.bool(), need_weights=False)
    cross = semantic.cross_norm(cross)
    fused = semantic.fusion(torch.cat((
        appearance.unsqueeze(0).expand(batch, -1, -1),
        geo.unsqueeze(0).expand(batch, -1, -1),
        motion_stream.unsqueeze(0).expand(batch, -1, -1), cross), -1))
    set_features = semantic.candidate_set(fused)
    return semantic.semantic_head(set_features).squeeze(-1)


@torch.inference_mode()
def teacher_batch(teacher, encoded, text_tokens, text_mask):
    """Vectorized L29 current-membership logit for one frame and B texts."""
    obs, obs_mask, _track_base = encoded
    qtok = teacher.base.text_proj(torch.nan_to_num(text_tokens.float()))
    q = teacher.base._masked_mean(qtok, text_mask.bool())
    batch, tracks, length = int(q.shape[0]), int(obs.shape[0]), int(obs.shape[1])
    obs_b = obs.unsqueeze(0).expand(batch, -1, -1, -1)
    q_b = q[:, None, None, :].expand(batch, tracks, length, -1)
    memberships = teacher.base.membership_head(
        torch.cat((obs_b, q_b), -1)).squeeze(-1)
    latest = obs_mask.long().sum(-1).clamp_min(1) - 1
    index = latest.view(1, tracks, 1).expand(batch, tracks, 1)
    return memberships.gather(2, index).squeeze(-1)


def gt_ids_by_frame(path: Path):
    result = defaultdict(set)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.replace(",", " ").split()
        if len(fields) >= 2:
            result[int(float(fields[0]))].add(str(int(float(fields[1]))))
    return {int(frame): set(values) for frame, values in result.items()}


def history_bucket(bank: dict, begin: int, end: int, frame: int) -> str:
    tensors = bank["tensors"]
    frames = tensors["frame"].long().tolist()
    lengths = []
    for row in range(begin, end):
        track = int(tensors["track_id"][row])
        eligible = [candidate for candidate in bank["track_rows"].get(track, [row])
                    if int(frames[candidate]) <= frame]
        lengths.append(min(8, len(eligible)))
    mean = float(np.mean(lengths)) if lengths else 0.0
    if mean <= 1:
        return "0-1"
    if mean <= 4:
        return "2-4"
    return "5-8"


def record_payload(query, frame_value, scores, teacher_scores, labels, sources, hist_bucket):
    return {
        "split": "test", "checkpoint_step": 250,
        "dataset": query["dataset"], "video": query["video"],
        "query_id": int(query["query_id"]), "expression": query["expression"],
        "sentence": query["sentence"], "frame_id": int(frame_value["frame_id"]),
        "category": ("multi_positive" if int(labels.sum()) > 1 else
                      "positive" if bool(labels.any()) else
                      "present_uncovered" if frame_value["target_ids"] else "inactive"),
        "candidate_count": int(len(labels)), "positive_count": int(labels.sum()),
        "score": np.asarray(scores, dtype=np.float32).tolist(),
        "semantic_score": np.asarray(scores, dtype=np.float32).tolist(),
        "identity_score": np.zeros(len(labels), dtype=np.float32).tolist(),
        "continuation_score": np.zeros(len(labels), dtype=np.float32).tolist(),
        "null_logit": 0.0, "label": np.asarray(labels, dtype=bool).tolist(),
        "sources": {key: np.asarray(value, dtype=bool).tolist()
                    for key, value in sources.items()},
        "target_ids": sorted(frame_value["target_ids"]),
        "history_length_bucket": hist_bucket,
        "unit_key": f"{query['dataset']}|{query['video']}|{query['query_id']}|{frame_value['frame_id']}",
        "teacher_score": np.asarray(teacher_scores, dtype=np.float32).tolist(),
    }


def run_dataset(dataset: str, queries, model, teacher, text, thresholds, device, out_root):
    cfg = DATASETS[dataset]
    ds_root = out_root / cfg["short"]
    res_root = ds_root / "uidm49"
    res_root.mkdir(parents=True, exist_ok=True)
    score_path = ds_root / "test_scores.jsonl"
    baseline_path = ds_root / "test_baseline_scores.jsonl"
    score_handle = score_path.open("w")
    baseline_handle = baseline_path.open("w")
    query_by_video = defaultdict(list)
    for query in queries:
        query_by_video[query["video"]].append(query)
    count = 0; positive_rows = 0; accepted = 0; start = time.time()
    for video in sorted(query_by_video):
        bank = load_unlabeled_bank(video)
        tensors = bank["tensors"]
        teacher_cache = build_teacher_cache(bank)
        gt_maps = {}
        handles = {}
        for query in query_by_video[video]:
            gt_path = cfg["gt_root"] / video / query["expression"] / "gt.txt"
            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)
            gt_maps[int(query["query_id"])] = gt_ids_by_frame(gt_path)
            exp_dir = res_root / video / query["expression"]
            exp_dir.mkdir(parents=True, exist_ok=True)
            gt_dst = exp_dir / "gt.txt"
            if not gt_dst.exists():
                gt_dst.symlink_to(gt_path.resolve())
            handles[int(query["query_id"])] = (query, (exp_dir / "predict.txt").open("w"))
        qvalues = query_by_video[video]
        for frame_index in range(len(tensors["frame_ids"])):
            fv = frame_inputs(bank, frame_index)
            n = int(fv["end"] - fv["begin"])
            if n == 0:
                continue
            move = {key: value.to(device) for key, value in fv.items()
                    if key in ("clip", "history_clip", "geometry", "motion", "context",
                               "lifecycle", "objectness", "relation")}
            # Encode the same frame once for the complete query batch.  This
            # is semantically the L49 step-250 branch, not a new model.
            for start_q in range(0, len(qvalues), 64):
                batch_queries = qvalues[start_q:start_q + 64]
                indices = [text["sentence_to_index"][q["sentence"]] for q in batch_queries]
                token = text["token_hidden"][indices].to(device)
                mask = text["attention_mask"][indices].bool().to(device)
                with torch.inference_mode():
                    scores_b = semantic_batch(model.semantic, move, token, mask).float().cpu().numpy()
                    obs, obs_mask, obs_time, _gt, _frames = state_at(teacher_cache, int(fv["frame_id"]), history=8)
                    encoded = teacher.encode_observations(obs.to(device), obs_mask.to(device), obs_time.to(device))
                    teacher_b = teacher_batch(teacher, encoded, token, mask).float().cpu().numpy()
                track_ids = fv["track_id"].numpy().astype(np.int64)
                source = source_masks(bank, fv["begin"], fv["end"])
                hist = history_bucket(bank, fv["begin"], fv["end"], int(fv["frame_id"]))
                for offset, query in enumerate(batch_queries):
                    targets = gt_maps[int(query["query_id"])].get(int(fv["frame_id"]) + 1, set())
                    labels = np.asarray([str(track) in targets for track in track_ids], dtype=bool)
                    frame_meta = {"frame_id": int(fv["frame_id"]), "target_ids": targets}
                    score = scores_b[offset]
                    teacher_score = np.asarray([-20.0] * len(track_ids), dtype=np.float32)
                    teacher_values = {int(track): float(value)
                                      for track, value in zip(teacher_cache["track_ids"].tolist(), teacher_b[offset])}
                    teacher_score = np.asarray([teacher_values.get(int(track), -20.0)
                                                for track in track_ids], dtype=np.float32)
                    row = record_payload(query, frame_meta, score, teacher_score, labels, source, hist)
                    score_handle.write(json.dumps(row, allow_nan=False) + "\n")
                    base = dict(row)
                    base["checkpoint_step"] = "L29"
                    base["score"] = teacher_score.tolist()
                    base["semantic_score"] = teacher_score.tolist()
                    base["teacher_score"] = teacher_score.tolist()
                    baseline_handle.write(json.dumps(base, allow_nan=False) + "\n")
                    threshold = float(thresholds[dataset])
                    keep = np.flatnonzero(score >= threshold)
                    pred_handle = handles[int(query["query_id"])][1]
                    for local in keep.tolist():
                        x1, y1, x2, y2 = [float(value) for value in fv["box"][local].tolist()]
                        pred_handle.write(
                            f"{int(fv['frame_id']) + 1},{int(track_ids[local])},"
                            f"{x1:.3f},{y1:.3f},{x2 - x1:.3f},{y2 - y1:.3f},"
                            f"{float(score[local]):.6f},-1,-1,-1\n")
                    count += 1; positive_rows += int(labels.sum()); accepted += int(len(keep))
        for _query, handle in handles.values():
            handle.close()
        del bank, teacher_cache
        gc.collect()
        print(json.dumps({"dataset": dataset, "video": video,
                          "queries": len(qvalues), "units_so_far": count,
                          "elapsed_sec": time.time() - start}, ensure_ascii=False), flush=True)
    score_handle.close(); baseline_handle.close()
    seqmap_out = ds_root / "seqmap_official.txt"
    seqmap_out.write_text(cfg["seqmap"].read_text())
    env = dict(os.environ); env["RMOT_IMG_ROOT"] = str(IMG_ROOT.resolve())
    cmd = [PY, str(EVAL_RUN), "--METRICS", "HOTA", "CLEAR", "Identity",
           "--SEQMAP_FILE", str(seqmap_out.resolve()), "--SKIP_SPLIT_FOL", "True",
           "--GT_FOLDER", str(res_root.resolve()), "--TRACKERS_FOLDER", str(res_root.resolve()),
           "--TRACKERS_TO_EVAL", str(res_root.resolve()),
           "--GT_LOC_FORMAT", "{gt_folder}{video_id}/{expression_id}/gt.txt",
           "--USE_PARALLEL", "False", "--PRINT_ONLY_COMBINED", "False",
           "--PLOT_CURVES", "False"]
    log_path = ds_root / "trackeval_official.log"
    with log_path.open("w") as log:
        subprocess.run(cmd, cwd=str(EVAL_RUN.parent), env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    summary = {
        "dataset": dataset, "queries": len(queries), "frame_units": count,
        "positive_rows": positive_rows, "accepted_candidates": accepted,
        "predictions_per_positive": accepted / max(1, positive_rows),
        "checkpoint_step": 250, "threshold": float(thresholds[dataset]),
        "scores": str(score_path.resolve()), "baseline_scores": str(baseline_path.resolve()),
        "trackeval_log": str(log_path.resolve()), "trackeval_completed": True,
        "official_test_labels_read_after_freeze": True,
        "test_labels_used_for_selection": False,
        "screening_or_test_labels_used_for_model_selection": False,
    }
    (ds_root / "test_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/l49/test/official_frozen_step250")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    if Path(args.out_root).is_absolute():
        out_root = Path(args.out_root)
    else:
        out_root = ROOT / args.out_root
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite official test output: {out_root}")
    out_root.mkdir(parents=True)
    started = time.time()
    selection = json.loads(SELECTED.read_text())
    if int(selection["selected_step"]) != 250:
        raise RuntimeError("this frozen test runner is intentionally bound to selected step 250")
    checkpoint = Path(selection["selected_checkpoint"])
    if sha256(checkpoint) != selection["selected_checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint SHA changed after validation selection")
    thresholds = {dataset: float(value["threshold"])
                  for dataset, value in selection["selected_thresholds"].items()}
    queries = {dataset: load_queries(dataset) for dataset in DATASETS}
    all_queries = [query for values in queries.values() for query in values]
    text_path = out_root / "text_cache.pt"
    text_meta = build_test_text_cache(all_queries, text_path)
    text = torch.load(text_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device if args.device != "cpu" or not torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("model_config", {})
    model = L49KittiRMOT(hidden=int(cfg.get("hidden", 256)), heads=int(cfg.get("heads", 4)),
                         history_length=int(cfg.get("history_length", 8))).to(device)
    model.load_state_dict(payload["model"], strict=True); model.eval()
    teacher = L29FrameMembershipSetDecoder().to(device)
    teacher.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device,
                                       weights_only=False)["model"], strict=True); teacher.eval()
    summaries = {}
    try:
        # The two official evaluation splits are intentionally sequential.
        summaries["refer_kitti_v1"] = run_dataset(
            "refer_kitti_v1", queries["refer_kitti_v1"], model, teacher, text,
            thresholds, device, out_root)
        summaries["refer_kitti_v2"] = run_dataset(
            "refer_kitti_v2", queries["refer_kitti_v2"], model, teacher, text,
            thresholds, device, out_root)
    except Exception as exc:
        (out_root / "INCOMPLETE.md").write_text(
            "# L49 official test incomplete\n\n"
            f"First actionable error: `{type(exc).__name__}: {exc}`\n")
        raise
    provenance = {
        "format": "locatemot-l49-official-test-provenance-v1",
        "started_at_unix": started, "completed_at_unix": time.time(),
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "selection_artifact": str(SELECTED.resolve()),
        "selection_artifact_sha256": sha256(SELECTED),
        "selection_rule": selection["selection_rule"],
        "validation_gate_before_test": selection["validation_gate"],
        "thresholds_frozen_from_calibration": thresholds,
        "calibration_labels_only_for_thresholds": True,
        "test_labels_read_after_selection": True,
        "test_labels_used_for_selection": False,
        "test_labels_used_for_model_or_threshold_tuning": False,
        "official_test_query_counts": {key: len(value) for key, value in queries.items()},
        "text_cache": text_meta,
        "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
        "l29_checkpoint_sha256": sha256(L29_CHECKPOINT),
        "fixed_manifest": str((ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json").resolve()),
        "fixed_manifest_sha256": sha256(ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"),
        "ordinary_mot_ovmot_touched": False,
        "summaries": summaries,
        "device": str(device),
    }
    (out_root / "test_run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"out_root": str(out_root.resolve()), "summaries": summaries,
                      "elapsed_sec": time.time() - started,
                      "official_test_labels_read_after_freeze": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
