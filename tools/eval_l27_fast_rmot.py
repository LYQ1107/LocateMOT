#!/usr/bin/env python3
"""L27 fast RMOT validation for frozen L26 adapter checkpoints.

Scores are produced once for the fixed 64/96 manifest. Thresholds and NULL
parameters are fitted from calibration labels only; screening labels are used
only for frozen reporting and official fast TrackEval evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l26_crossmodal_adapter import L26BoundedResidual, L26CrossModalAdapter
from tools.eval_l18_carr import run_trackeval, trainval_queries
from tools.train_l26_crossmodal_adapter import EXP, REC1, REC2, frame_positive, load_expressions

MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
V5 = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
GT_ROOT = ROOT / "outputs/l18/data/trainval_gt/kitti"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_gt(video: str):
    path = REC1 / f"{video}.pkl"
    if not path.exists():
        path = REC2 / f"{video}.pkl"
    rec = pickle.loads(path.read_bytes())
    return {int(x["frame"]): {str(k): np.asarray(v, np.float32) for k, v in x.get("gt_boxes", {}).items()} for x in rec["frames"]}


def load_bank(video: str):
    d = torch.load(V5 / "kitti" / f"{video}.pt", map_location="cpu", weights_only=False)["tensors"]
    return {"roi": d["dino_roi_tokens_v5"].float(), "coords": d["roi_coords_v5"].float(), "box": d["box"].float(), "objectness": d["objectness"].float(), "pool": d["pool_id"].long(), "track": d["track_id"].long(), "frame_ids": d["frame_ids"].long(), "ptr": d["frame_ptr"].long()}


def load_text():
    manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    index = {(str(x["video"]), str(x["expression"])): int(x["query_index"]) for x in manifest}
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    needed = sorted(set(index.values()))
    hidden = text["token_hidden"][needed].float()
    mask = text["attention_mask"][needed].bool()
    remap = {old: i for i, old in enumerate(needed)}
    del text
    return index, hidden, mask, remap


def make_entries():
    expressions = load_expressions()
    by_key = {(x["video"], str(x["expression"])): x for x in expressions}
    manifest = json.loads(MANIFEST.read_text())
    entries = []
    for row in sorted(manifest["queries"], key=lambda x: int(x["query_index"])):
        key = (str(row["video"]), str(row["expression"]))
        if key not in by_key:
            raise KeyError(key)
        # Manifest split is authoritative for calibration/screening isolation;
        # expression metadata may carry the video's broader data split.
        entries.append({**by_key[key], **row, "video": key[0], "expression": key[1]})
    if len([x for x in entries if x["split"] == "calibration"]) != 64 or len([x for x in entries if x["split"] == "screening"]) != 96:
        raise AssertionError("fixed manifest is not 64/96")
    return entries


def score_query(model, residual, entry, bank, gt, text_index, hidden, mask, remap, device):
    ti = remap[text_index[(entry["video"], entry["expression"])]]
    qh, qm = hidden[ti].to(device), mask[ti].to(device)
    targets = {int(k): {str(x) for x in v} for k, v in entry.get("label", {}).items()}
    frames = []; tracks = []; boxes = []; scores = []; sources = []; labels = []; ious = []
    with torch.inference_mode():
        for fi, frame in enumerate(bank["frame_ids"].tolist()):
            begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
            if end <= begin:
                continue
            out = model(qh, qm, bank["roi"][begin:end].to(device), bank["coords"][begin:end].to(device))
            s = out["score"].float().cpu().numpy()
            b = bank["box"][begin:end].numpy().astype(np.float32)
            ids = targets.get(int(frame), set())
            positive, _covered = frame_positive(b, ids, gt.get(int(frame), {}))
            frame_gt = [gt[int(frame)][gid] for gid in ids if gid in gt.get(int(frame), {})]
            frame_iou = np.zeros(len(b), np.float32)
            for j, box in enumerate(b):
                best = 0.0
                for g in frame_gt:
                    l, t = max(float(box[0]), float(g[0])), max(float(box[1]), float(g[1]))
                    r, d = min(float(box[2]), float(g[2])), min(float(box[3]), float(g[3]))
                    inter = max(0.0, r - l) * max(0.0, d - t)
                    area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
                    ga = max(0.0, float(g[2] - g[0])) * max(0.0, float(g[3] - g[1]))
                    best = max(best, inter / max(1e-6, area + ga - inter))
                frame_iou[j] = best
            n = end - begin
            frames.extend([int(frame)] * n); tracks.extend(bank["track"][begin:end].tolist()); boxes.extend(b.tolist()); scores.extend(s.tolist()); sources.extend(bank["pool"][begin:end].tolist()); labels.extend(positive.astype(np.uint8).tolist()); ious.extend(frame_iou.tolist())
    return {"frame": np.asarray(frames, np.int32), "track_id": np.asarray(tracks, np.int64), "box": np.asarray(boxes, np.float32).reshape(-1, 4), "score": np.asarray(scores, np.float32), "source": np.asarray(sources, np.int8), "label": np.asarray(labels, np.uint8), "gt_iou": np.asarray(ious, np.float32)}


def frame_groups(data):
    return [(int(frame), np.flatnonzero(data["frame"] == frame)) for frame in np.unique(data["frame"])]


def pool_frame_units(items):
    """Concatenate query arrays while preserving query/video/frame isolation."""
    pooled = []
    offset = 0
    for data in items:
        copy = {key: np.asarray(value).copy() for key, value in data.items()}
        if len(copy["frame"]):
            span = int(copy["frame"].max()) + 1
            copy["frame"] = copy["frame"].astype(np.int64) + offset
            offset += span
        pooled.append(copy)
    return {key: np.concatenate([x[key] for x in pooled]) for key in pooled[0]}


def base_metrics(data):
    y = data["label"].astype(bool); s = data["score"]
    strict = []; best = []; average = []; aps = []; top1 = []; top5 = []; topk = []; pos_frames = 0; null_frames = 0
    for _frame, idx in frame_groups(data):
        order = idx[np.argsort(-s[idx], kind="stable")]; p = idx[y[idx]]; n = idx[~y[idx]]
        if not len(p):
            null_frames += 1; continue
        pos_frames += 1; top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any())); k = min(len(order), len(p)); topk.append(float(y[order[:k]].sum() / max(1, len(p))))
        if len(n):
            strict.append(float(s[p].min() - s[n].max())); best.append(float(s[p].max() - s[n].max())); average.append(float(s[p].mean() - s[n].max()))
        ordered = y[order]; positions = np.flatnonzero(ordered); aps.append(float(np.mean([ordered[:q + 1].mean() for q in positions])))
    def st(v):
        if not v: return {"count": 0, "mean": None}
        a = np.asarray(v, np.float64); return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)), "q10": float(np.quantile(a, .1)), "q90": float(np.quantile(a, .9))}
    return {"rows": int(len(y)), "positive_rows": int(y.sum()), "positive_frames": pos_frames, "null_frames": null_frames, "top1_candidate_recall": float(np.mean(top1)) if top1 else None, "top5_candidate_recall": float(np.mean(top5)) if top5 else None, "topk_set_recall_k_equals_gt_positive_count": float(np.mean(topk)) if topk else None, "frame_average_precision": float(np.mean(aps)) if aps else None, "strict_min_positive_margin": st(strict), "best_positive_margin": st(best), "average_positive_margin": st(average), "source_rows": {"main": int(np.count_nonzero(data["source"] == 0)), "reserve": int(np.count_nonzero(data["source"] == 1))}}


def selection(data, threshold, strategy, null_threshold=None, gap_threshold=None):
    selected = np.zeros(len(data["score"]), bool); s = data["score"]
    for _frame, idx in frame_groups(data):
        order = idx[np.argsort(-s[idx], kind="stable")]
        if not len(order): continue
        top1 = float(s[order[0]]); top2 = float(s[order[1]]) if len(order) > 1 else -np.inf
        if strategy == "threshold":
            selected[idx] = s[idx] >= threshold
        elif strategy == "null_max":
            if null_threshold is not None and top1 >= float(null_threshold): selected[idx] = s[idx] >= threshold
        elif strategy == "gap_top1":
            if gap_threshold is not None and top1 >= threshold and top1 - top2 >= float(gap_threshold): selected[order[0]] = True
        elif strategy == "threshold_top1":
            if top1 >= threshold: selected[order[0]] = True
        else: raise ValueError(strategy)
    return selected


def threshold_metrics(data, selected):
    y = data["label"].astype(bool); tp = selected & y; fp = selected & ~y; fn = ~selected & y
    frame_f1=[]; frame_prec=[]; frame_rec=[]; fp_per=[]; empty=[]; null_accept=[]; source={}
    for frame, idx in frame_groups(data):
        pos = bool(y[idx].any()); pred = bool(selected[idx].any()); empty.append(not pred); null_accept.append((not pos) and pred); tp_f = bool((selected[idx] & y[idx]).any())
        frame_prec.append(float(tp_f / max(1, int(selected[idx].sum())))); frame_rec.append(float(tp_f / max(1, int(y[idx].sum())))); frame_f1.append(float(2 * frame_prec[-1] * frame_rec[-1] / max(1e-12, frame_prec[-1] + frame_rec[-1])) if pos else 0.0); fp_per.append(int((selected[idx] & ~y[idx]).sum()))
    for source_id, name in ((0, "main"), (1, "reserve")):
        pool = data["source"] == source_id; source[name] = {"selected": int((selected & pool).sum()), "positive": int((y & pool).sum()), "precision": float((selected & pool & y).sum() / max(1, (selected & pool).sum())), "recall": float((selected & pool & y).sum() / max(1, (y & pool).sum()))}
    precision = float(tp.sum() / max(1, selected.sum())); recall = float(tp.sum() / max(1, y.sum())); f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"selected": int(selected.sum()), "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "precision": precision, "recall": recall, "f1": f1, "frame_f1": float(np.mean(frame_f1)), "frame_precision": float(np.mean(frame_prec)), "frame_recall": float(np.mean(frame_rec)), "false_positive_candidates_per_frame": float(np.mean(fp_per)), "empty_output_rate": float(np.mean(empty)), "null_frame_false_acceptance": float(np.mean(null_accept)), "source_precision": source, "predictions_per_gt_positive": float(selected.sum() / max(1, y.sum()))}


def calibrate(data):
    candidates = np.unique(np.quantile(data["score"], np.linspace(.01, .995, 128))).tolist()
    candidates += [float(data["score"].min()), float(data["score"].max())]
    rows = []
    for t in candidates:
        m = threshold_metrics(data, selection(data, t, "threshold")); rows.append((t, m))
    def pick(kind):
        if kind == "precision_first":
            valid = [(t,m) for t,m in rows if m["recall"] >= .45 and m["false_positive_candidates_per_frame"] <= 10]
            return max(valid or rows, key=lambda x: (x[1]["precision"], x[1]["recall"], -x[1]["false_positive_candidates_per_frame"]))
        if kind == "recall_first":
            valid = [(t,m) for t,m in rows if m["precision"] >= .10 and m["false_positive_candidates_per_frame"] <= 30]
            return max(valid or rows, key=lambda x: (x[1]["recall"], x[1]["precision"], -x[1]["false_positive_candidates_per_frame"]))
        valid = [(t,m) for t,m in rows if m["false_positive_candidates_per_frame"] <= 10]
        return max(valid or rows, key=lambda x: (x[1]["frame_f1"], x[1]["f1"], x[1]["precision"]))
    thresholds = {name: {"threshold": float(pick(name)[0]), "calibration_metrics": pick(name)[1]} for name in ("precision_first", "recall_first", "balanced")}
    null_scores = []
    gaps = []
    for _frame, idx in frame_groups(data):
        order = idx[np.argsort(-data["score"][idx], kind="stable")]
        if not data["label"][idx].any(): null_scores.append(float(data["score"][order[0]]))
        if data["label"][idx].any() and len(order) > 1: gaps.append(float(data["score"][order[0]] - data["score"][order[1]]))
    thresholds["null_max_threshold"] = float(np.quantile(null_scores, .95)) if null_scores else None
    thresholds["null_max_samples"] = len(null_scores)
    thresholds["null_max_fallback"] = "calibration_null_q95" if null_scores else "no-null: strategy emits no additional NULL suppression"
    thresholds["gap_threshold"] = float(np.quantile(gaps, .20)) if gaps else None
    thresholds["gap_samples"] = len(gaps)
    thresholds["gap_fallback"] = "calibration_positive_top1_top2_q20" if gaps else "no-gap: strategy emits no gap-gated output"
    thresholds["candidate_count"] = len(candidates)
    return thresholds


def trackeval_outputs(out_root, model_name, run_name, threshold, arrays, entries, split_name, null_threshold=None, gap_threshold=None, selection_strategy=None):
    run = out_root / model_name / run_name
    strategy = selection_strategy or run_name
    result_root = run / "uidm18"; allowed = set()
    for entry in entries:
        if entry["split"] != "screening": continue
        key = (entry["video"], entry["expression"]); allowed.add(key)
        data = arrays[model_name][key]
        pred = result_root / entry["video"] / entry["expression"] / "predict.txt"; pred.parent.mkdir(parents=True, exist_ok=True)
        gt = GT_ROOT / entry["video"] / entry["expression"] / "gt.txt"; dst = pred.parent / "gt.txt"
        if not gt.exists(): raise FileNotFoundError(f"existing trainval GT missing: {gt}")
        if not dst.exists(): dst.symlink_to(gt.resolve())
        keep = selection(data, threshold, strategy, null_threshold, gap_threshold)
        with pred.open("w") as f:
            for i in np.flatnonzero(keep):
                x1,y1,x2,y2 = [float(v) for v in data["box"][i]]
                prob = 1.0 / (1.0 + np.exp(-np.clip(float(data["score"][i]), -40, 40)))
                f.write(f"{int(data['frame'][i])+1},{int(data['track_id'][i])},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},{prob:.6f},-1,-1,-1\n")
    # The existing train_val GT is read-only; TrackEval gets a strategy-local tree.
    seqmap = GT_ROOT / "seqmap.txt"
    metrics, log = run_trackeval("trainval_kitti", run, seqmap, {"0004", "0018"}, allowed)
    return {"metrics": metrics, "log": str(log), "output": str(run)}


def main():
    out = ROOT / "outputs/l27/fast_rmot_validation"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    entries = make_entries(); text_index, hidden, mask, remap = load_text(); models={}
    a_path = ROOT / "outputs/l26/train/C1_crossmodal_adapter_S2000/checkpoint_c1_step2000.pt"
    b_path = ROOT / "outputs/l26/fallback/F4_bounded_residual_S500_retry/checkpoint_bounded_residual_step500.pt"
    for p in (a_path, b_path):
        if not p.exists(): raise FileNotFoundError(p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(1)
    a = L26CrossModalAdapter(variant="token_region").to(device); a.load_state_dict(torch.load(a_path, map_location=device, weights_only=False)["model"]); a.eval()
    b = L26BoundedResidual(L26CrossModalAdapter(variant="token_region")).to(device); b.load_state_dict(torch.load(b_path, map_location=device, weights_only=False)["model"]); b.eval()
    model_objs = {"A_C1_S2000": a, "B_F4_bounded_residual": b}; arrays={k:{} for k in model_objs}; gts={}
    banks = {video: load_bank(video) for video in sorted({x["video"] for x in entries})}
    started=time.time()
    for model_name, model in model_objs.items():
        for pos, entry in enumerate(entries, 1):
            key=(entry["video"], entry["expression"])
            if entry["video"] not in gts: gts[entry["video"]]=load_gt(entry["video"])
            arrays[model_name][key]=score_query(model, None, entry, banks[entry["video"]], gts[entry["video"]], text_index, hidden, mask, remap, device)
            if pos == 1 or pos % 16 == 0: print(f"[l27-score] {model_name} {pos}/{len(entries)}", flush=True)
    score_dir=out/"scores"; score_dir.mkdir()
    for model_name, qs in arrays.items():
        md=score_dir/model_name; md.mkdir()
        for (video, expression), data in qs.items():
            safe=hashlib.sha1(expression.encode()).hexdigest()[:12]
            np.savez_compressed(md/f"{video}_{safe}.npz", **data)
    summaries={}; trackeval={}
    for model_name, qs in arrays.items():
        cal_data=[qs[(x["video"],x["expression"])] for x in entries if x["split"]=="calibration"]
        cal={k:np.concatenate([x[k] for x in cal_data]) for k in cal_data[0]}
        calibration=calibrate(cal); summaries[model_name]={"calibration":calibration, "calibration_ranking":base_metrics(cal)}
        for strat_name in ("precision_first","recall_first","balanced"):
            t=calibration[strat_name]["threshold"]
            for strategy in ("threshold","null_max","gap_top1","threshold_top1"):
                if strategy=="null_max": nt=calibration["null_max_threshold"]; gt=None
                elif strategy=="gap_top1": nt=None; gt=calibration["gap_threshold"]
                else: nt=gt=None
                screen_items=[(x,qs[(x["video"],x["expression"])]) for x in entries if x["split"]=="screening"]
                screen={k:np.concatenate([d[k] for _e,d in screen_items]) for k in screen_items[0][1]}
                chosen=selection(screen,t,strategy,nt,gt); key=f"{strat_name}__{strategy}"
                summaries[model_name][key]={"threshold":t,"strategy":strategy,"null_threshold":nt,"gap_threshold":gt,"ranking":base_metrics(screen),"threshold_metrics":threshold_metrics(screen,chosen),"screening_gt_used_for_threshold":False}
                trackeval.setdefault(model_name,{})[key]=trackeval_outputs(out,model_name,key,t,qs,entries,"screening",nt,gt)
    payload={"format":"locatemot-l27-fast-rmot-validation-v1","manifest":str(MANIFEST),"manifest_sha256":sha(MANIFEST),"checkpoint_provenance":{"A_C1_S2000":str(a_path.resolve()),"A_sha256":sha(a_path),"B_F4_bounded_residual":str(b_path.resolve()),"B_sha256":sha(b_path)},"v5_root":str(V5),"device":str(device),"query_counts":{"all":len(entries),"calibration":64,"screening":96},"calibration_labels_used_for_threshold":True,"screening_gt_used_for_threshold":False,"elapsed_sec":time.time()-started,"candidate_metrics":summaries,"trackeval":trackeval}
    (out/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"out":str(out),"elapsed_sec":payload["elapsed_sec"],"trackeval_models":list(trackeval)},indent=2),flush=True)


if __name__=="__main__": main()
