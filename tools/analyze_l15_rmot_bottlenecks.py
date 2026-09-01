"""Stage L15 RMOT oracle decomposition.

This tool is deliberately diagnostic.  It uses evaluation annotations only to
measure the ceiling of the fixed L13 proposal pool and to separate proposal
coverage, semantic selection, and identity maintenance.  No output of this
tool is a deployable model or a training cache.

Variants written under outputs/l15/bottlenecks are:

* semantic_oracle: one fixed-pool candidate is selected for each visible
  referent using evaluation GT, then the unchanged L11 UIDM assigns IDs;
* association_oracle: the same fixed-pool boxes are assigned an oracle
  consistent ID, isolating the box/observation ceiling;
* gt_observation_uidm: exact referent GT boxes are passed through UIDM, while
  appearance features come from the nearest fixed-pool candidate (or zeros if
  there is no valid candidate).  This feature construction is recorded as an
  approximation and is never used for a main result.

The proposal statistics are computed at IoU 0.30, 0.50, and 0.70.  Official
RMOT TrackEval is invoked for the three prediction variants and for each
expression family on the semantic oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
PYTHON = "/home/lwr/anaconda3/envs/locatemot/bin/python"
sys.path.insert(0, str(ROOT))

from locatemot.models.l8_unified import (  # noqa: E402
    L8UnifiedUIDM,
    load_l8_state,
)
from tools.eval_l13_rmot import (  # noqa: E402
    DANCE_META,
    DANCE_GT,
    DANCE_SEQMAP,
    EVAL_RUN,
    KITTI_DATA,
    V1_DATA,
    V1_GT,
    V1_SEQMAP,
    V2_GT,
    V2_SEQMAP,
    build_tracker,
    load_dance_entries,
    load_queries,
    make_dance_frames,
    make_kitti_frames,
)

SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}

HOTA_KEYS = [
    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr",
    "LocA", "RHOTA", "HOTA(0)", "LocA(0)", "HOTALocA(0)",
]


def box_iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2]) - float(a[0])) * \
        max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * \
        max(0.0, float(b[3]) - float(b[1]))
    return inter / max(1e-9, aa + bb - inter)


def iou_matrix(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    out = np.zeros((len(a), len(b)), np.float32)
    for i, aa in enumerate(a):
        for j, bb in enumerate(b):
            out[i, j] = box_iou(aa, bb)
    return out


def read_expression_meta(dataset):
    if dataset == "kitti_v1":
        path = V1_DATA / "expressions.json"
    elif dataset == "kitti_v2":
        path = KITTI_DATA / "expressions.json"
    else:
        path = DANCE_META
    raw = json.loads(path.read_text())
    out = {}
    for seq, entries in raw.items():
        for entry in entries:
            name = entry.get("expression", entry.get("sentence", ""))
            out[(seq, name)] = entry
    return out


def expression_text(entry):
    return str(entry.get("sentence", entry.get("expression", ""))).lower()


def expression_family(text):
    """Fixed, predeclared lexical taxonomy for diagnostic stratification."""
    t = text.lower()
    groups = []
    if re.search(r"\b(car|cars|auto|autos|automobile|automobiles|vehicle|vehicles|truck|bus|bicycle|motor[- ]?vehicle|person|people|pedestrian|pedestrians|men|women|folks|individuals)\b", t):
        groups.append("category")
    if re.search(r"\b(black|white|red|blue|green|yellow|silver|gray|grey|light|dark|color|colou?r|bag|shirt|pants|skirt|dress|clothes|wearing|painted)\b", t):
        groups.append("appearance/color")
    if re.search(r"\b(left|right|front|ahead|behind|near|next|beside|between|side|positioned|located|in front|opposite)\b", t):
        groups.append("absolute spatial position")
    if re.search(r"\b(moving|driving|turning|parking|parked|braking|brake|slowing|stopping|walking|running|faster|direction|transit|motion)\b", t):
        groups.append("motion/action")
    if re.search(r"\b(before|after|earlier|later|then|while|during|eventually|previously|following)\b", t):
        groups.append("temporal relation")
    if re.search(r"\b(and|or|with|that are|which are|who are|next to|to the)\b", t) and len(groups) >= 2:
        groups.append("multi-clause/compositional")
    if "multi-clause/compositional" in groups:
        return "multi-clause/compositional"
    for name in ("temporal relation", "motion/action", "absolute spatial position", "appearance/color", "category"):
        if name in groups:
            return name
    return "category"


def target_ids(entry, frame):
    labels = entry.get("label", entry.get("targets", {}))
    vals = labels.get(str(int(frame)), labels.get(int(frame), []))
    return [str(x) for x in vals]


def frame_gt(fr, ids):
    boxes = fr.get("gt_boxes", {})
    out = []
    for gid in ids:
        if str(gid) in boxes:
            out.append((str(gid), np.asarray(boxes[str(gid)], np.float32)))
    return out


def frame_targets(fr, entry):
    """Return query-specific GT targets, including Dance's empty-label JSON."""
    ids = target_ids(entry, fr["frame"])
    if ids and fr.get("gt_boxes"):
        return frame_gt(fr, ids)
    by_expr = fr.get("gt_by_expression", {})
    boxes = by_expr.get(entry.get("expression", ""), {})
    if ids:
        boxes = {gid: boxes[gid] for gid in ids if gid in boxes}
    return [(str(gid), np.asarray(box, np.float32))
            for gid, box in boxes.items()]


def candidate_boxes(fr):
    return np.asarray(fr.get("boxes", []), np.float32).reshape(-1, 4)


def proposal_matches(fr, ids, threshold, targets=None):
    """One-to-one GT-to-pool matching at a declared IoU threshold."""
    if targets is None:
        targets = frame_gt(fr, ids)
    boxes = candidate_boxes(fr)
    if not targets or len(boxes) == 0:
        return [], targets
    mat = iou_matrix([x[1] for x in targets], boxes)
    rows, cols = linear_sum_assignment(-mat)
    pairs = []
    for r, c in zip(rows, cols):
        value = float(mat[r, c])
        if value >= threshold:
            pairs.append((targets[r][0], int(c), value))
    return pairs, targets


def nearest_candidate(fr, box):
    boxes = candidate_boxes(fr)
    if len(boxes) == 0:
        return None, 0.0
    vals = np.asarray([box_iou(box, x) for x in boxes], np.float32)
    idx = int(vals.argmax())
    return idx, float(vals[idx])


def feature_dict(fr, idx):
    n = len(candidate_boxes(fr))
    pbd = np.asarray(fr.get("pbd", np.zeros((n, 2048))), np.float32)
    clip = np.asarray(fr.get("clip", np.zeros((n, 512))), np.float32)
    if idx is None or idx < 0 or idx >= n:
        p = np.zeros(2048, np.float32)
        c = np.zeros(512, np.float32)
        gen = 0.0
    else:
        p = pbd[idx] if len(pbd) > idx else np.zeros(2048, np.float32)
        c = clip[idx] if len(clip) > idx else np.zeros(512, np.float32)
        gen_arr = np.asarray(fr.get("gen", np.zeros(n)), np.float32)
        gen = float(gen_arr[idx]) if len(gen_arr) > idx else 0.0
    return {
        "pbd": np.asarray(p, np.float32),
        "pbd_be": np.asarray(p, np.float32),
        "region": np.zeros(4608, np.float32),
        "geom": np.zeros(5, np.float32),
        "gen": gen,
        "clip": np.asarray(c, np.float32),
    }


def make_candidate(fr, idx):
    return {"box": candidate_boxes(fr)[idx].copy(),
            "features": feature_dict(fr, idx), "index": int(idx)}


def make_gt_candidate(fr, box, source_idx=None):
    c = make_candidate(fr, source_idx) if source_idx is not None else {
        "box": np.asarray(box, np.float32).copy(),
        "features": feature_dict(fr, None),
        "index": -1,
    }
    c["box"] = np.asarray(box, np.float32).copy()
    return c


def load_dataset(dataset):
    queries, gt_root, seqmap = load_queries(dataset)
    q_by_seq = defaultdict(list)
    for seq, expr, spec in queries:
        q_by_seq[seq].append((expr, np.asarray(spec, np.float32)))
    frames_by_seq = {}
    sizes = {}
    if dataset == "dance":
        by_video = load_dance_entries()
        for seq in sorted(q_by_seq):
            frames, size = make_dance_frames(seq, by_video)
            # Dance's compact expression metadata in this checkout has empty
            # per-frame labels.  Its official GT files are query-specific, so
            # keep that mapping on the frames instead of merging all queries.
            by_expr = {}
            for expr, _spec in q_by_seq[seq]:
                path = gt_root / seq / expr / "gt.txt"
                frame_boxes = defaultdict(dict)
                if path.exists():
                    for line in path.read_text().splitlines():
                        fields = line.strip().split(",")
                        if len(fields) < 6:
                            continue
                        frame, gid = int(float(fields[0])), str(fields[1])
                        x, y, w, h = map(float, fields[2:6])
                        frame_boxes[frame][gid] = [x, y, x + w, y + h]
                by_expr[expr] = frame_boxes
            for fr in frames:
                fr["gt_by_expression"] = {
                    expr: boxes.get(int(fr["frame"]), {})
                    for expr, boxes in by_expr.items()
                }
            frames_by_seq[seq], sizes[seq] = frames, size
    else:
        for seq in sorted(q_by_seq):
            frames, size = make_kitti_frames(seq)
            # The KITTI pickle already stores the complete frame-level GT
            # dictionary; expose it to this diagnostic without changing the
            # production evaluator.
            raw = pickle.load(open(KITTI_DATA / f"{seq}.pkl", "rb"))
            for fr, raw_fr in zip(frames, raw["frames"]):
                fr["gt_boxes"] = {
                    str(gid): np.asarray(box, np.float32)
                    for gid, box in raw_fr.get("gt_boxes", {}).items()
                }
            frames_by_seq[seq], sizes[seq] = frames, size
    return q_by_seq, frames_by_seq, sizes, gt_root, seqmap


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = L8UnifiedUIDM(
        **SIZES[cfg.get("model", "large")],
        no_interaction=cfg.get("no_interaction", False),
        use_cue_rel=cfg.get("use_cue_rel", False),
        mode=cfg.get("mode", "unified"),
        sem_in_core=cfg.get("sem_in_core", True),
        cond_gated=cfg.get("cond_gated", True),
        spec_conditioned=False,
        trajectory_memory=True,
    ).to(device)
    missing, unexpected = load_l8_state(model, ck["model"])
    if missing or unexpected:
        raise RuntimeError(
            f"L11 checkpoint mismatch: missing={missing} unexpected={unexpected}"
        )
    model.eval()
    return model


def write_prediction(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(",".join(
                f"{float(x):.3f}" if isinstance(x, (float, np.floating))
                else str(x) for x in row) + "\n")


def ensure_gt_link(dst, src):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def query_rows_uidm(model, frames, size, dataset, entry, spec, threshold,
                    mode, device):
    tracker = build_tracker(model, spec, device)
    tracker.image_size = size
    rows = []
    oracle_id = {}
    next_oracle = 1
    for fr in frames:
        ids = target_ids(entry, fr["frame"])
        targets = frame_targets(fr, entry)
        if mode == "gt_observation_uidm":
            selected = []
            for gid, box in targets:
                idx, _ = nearest_candidate(fr, box)
                selected.append((gid, make_gt_candidate(fr, box, idx)))
        else:
            selected_pairs, _ = proposal_matches(
                fr, ids, threshold, targets=targets)
            selected = [(gid, make_candidate(fr, idx))
                        for gid, idx, _ in selected_pairs]
        if not selected:
            # The historical online evaluator skips frames with no candidate;
            # retain that exact behavior so the oracle does not introduce a
            # lifecycle policy that the L11 reference did not use.
            continue
        cands = [x[1] for x in selected]
        outputs = tracker.process_frame(int(fr["frame"]), cands)
        if len(outputs) != len(cands):
            raise RuntimeError(
                f"UIDM output mismatch {dataset}/{fr['frame']}: "
                f"{len(outputs)} vs {len(cands)}"
            )
        if mode == "association_oracle":
            for (gid, _), out in zip(selected, outputs):
                if gid not in oracle_id:
                    oracle_id[gid] = next_oracle
                    next_oracle += 1
                out["track_id"] = oracle_id[gid]
        frame_no = int(fr["frame"]) + (0 if dataset == "dance" else 1)
        for out in outputs:
            x1, y1, x2, y2 = [float(x) for x in out["box"]]
            rows.append([frame_no, int(out["track_id"]), x1, y1,
                         x2 - x1, y2 - y1, 1.0, -1, -1, -1])
    return rows


def query_rows_association(frames, dataset, entry, threshold):
    rows = []
    oracle_id = {}
    next_oracle = 1
    for fr in frames:
        ids = target_ids(entry, fr["frame"])
        targets = frame_targets(fr, entry)
        pairs, _ = proposal_matches(fr, ids, threshold, targets=targets)
        frame_no = int(fr["frame"]) + (0 if dataset == "dance" else 1)
        for gid, idx, _ in pairs:
            if gid not in oracle_id:
                oracle_id[gid] = next_oracle
                next_oracle += 1
            x1, y1, x2, y2 = [float(x) for x in candidate_boxes(fr)[idx]]
            rows.append([frame_no, oracle_id[gid], x1, y1,
                         x2 - x1, y2 - y1, 1.0, -1, -1, -1])
    return rows


def parse_hota_log(path):
    lines = path.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("HOTA:"):
            continue
        for later in lines[i + 1:]:
            if not later.startswith("COMBINED"):
                continue
            vals = later.split()[1:1 + len(HOTA_KEYS)]
            try:
                nums = [float(x) for x in vals]
            except ValueError:
                continue
            if len(nums) == len(HOTA_KEYS):
                return dict(zip(HOTA_KEYS, nums))
            break
    return {}


def evaluate_variant(dataset, variant_root, gt_root, seqmap, query_lines,
                     sizes, log_name):
    res_root = (variant_root / "uidm15").resolve()
    seqmap_out = variant_root / "seqmap_l15.txt"
    seqmap_out.parent.mkdir(parents=True, exist_ok=True)
    seqmap_out.write_text("\n".join(query_lines) + "\n")
    for line in query_lines:
        seq, expr = line.split("+", 1)
        ensure_gt_link(res_root / seq / expr / "gt.txt",
                       (gt_root / seq / expr / "gt.txt").resolve())
    env = dict(os.environ)
    env["RMOT_IMG_ROOT"] = str(
        ROOT / "data" / ("refer_dance/DanceTrack/training/image_02"
                          if dataset == "dance"
                          else "kitti_tracking_training/image_02")
    )
    cmd = [
        PYTHON, str(EVAL_RUN), "--METRICS", "HOTA", "CLEAR",
        "Identity", "--SEQMAP_FILE", str(seqmap_out.resolve()),
        "--SKIP_SPLIT_FOL", "True", "--GT_FOLDER", str(res_root),
        "--TRACKERS_FOLDER", str(res_root), "--TRACKERS_TO_EVAL",
        str(res_root), "--GT_LOC_FORMAT",
        "{gt_folder}{video_id}/{expression_id}/gt.txt",
        "--USE_PARALLEL", "False", "--PRINT_ONLY_COMBINED", "False",
        "--PLOT_CURVES", "False",
    ]
    log = variant_root / log_name
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(EVAL_RUN.parent), env=env,
                              stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(f"TrackEval failed for {variant_root}:\n{tail}")
    return parse_hota_log(log), log


def proposal_stats(dataset, q_by_seq, frames_by_seq, expr_meta):
    thresholds = (0.30, 0.50, 0.70)
    out = {str(t): {"frames": 0, "frames_hit": 0, "objects": 0,
                    "objects_hit": 0, "trajectory_objects": 0,
                    "trajectory_objects_hit": 0, "queries": 0,
                    "queries_full_hit": 0} for t in thresholds}
    family = defaultdict(lambda: {str(t): {"frames": 0, "frames_hit": 0,
                                            "objects": 0, "objects_hit": 0}
                                   for t in thresholds})
    query_rows = []
    for seq, qitems in q_by_seq.items():
        frames = frames_by_seq[seq]
        by_frame = {int(fr["frame"]): fr for fr in frames}
        for expr, _spec in qitems:
            entry = expr_meta[(seq, expr)]
            fam = expression_family(expression_text(entry))
            frame_ids = sorted(set(int(fr["frame"]) for fr in frames))
            for threshold in thresholds:
                key = str(threshold)
                objects = defaultdict(lambda: False)
                for frame in frame_ids:
                    fr = by_frame[frame]
                    ids = target_ids(entry, frame)
                    targets = frame_targets(fr, entry)
                    pairs, _ = proposal_matches(
                        fr, ids, threshold, targets=targets)
                    out[key]["frames"] += int(bool(targets))
                    out[key]["frames_hit"] += int(
                        bool(targets) and len(pairs) == len(targets))
                    out[key]["objects"] += len(targets)
                    out[key]["objects_hit"] += len(pairs)
                    family[fam][key]["frames"] += int(bool(targets))
                    family[fam][key]["frames_hit"] += int(
                        bool(targets) and len(pairs) == len(targets))
                    family[fam][key]["objects"] += len(targets)
                    family[fam][key]["objects_hit"] += len(pairs)
                    for gid, _box in targets:
                        objects[gid] = objects[gid] or any(
                            x[0] == gid for x in pairs)
                out[key]["trajectory_objects"] += len(objects)
                out[key]["trajectory_objects_hit"] += sum(objects.values())
                out[key]["queries"] += 1
                out[key]["queries_full_hit"] += int(
                    bool(objects) and all(objects.values()))
            query_rows.append({"dataset": dataset, "sequence": seq,
                               "expression": expr, "family": fam})
    return out, family, query_rows


def add_rates(stats):
    for value in stats.values():
        for prefix, den in (("frame", "frames"), ("object", "objects"),
                            ("trajectory_object", "trajectory_objects"),
                            ("query", "queries")):
            if den not in value:
                continue
            if prefix == "frame":
                num = value.get("frames_hit", 0)
            elif prefix == "object":
                num = value.get("objects_hit", 0)
            elif prefix == "trajectory_object":
                num = value.get("trajectory_objects_hit", 0)
            else:
                num = value.get("queries_full_hit", 0)
            value[prefix + "_recall"] = (float(num) / float(value[den])
                                          if value[den] else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kitti_v1", "kitti_v2", "dance"],
                    action="append")
    ap.add_argument("--variant", choices=["semantic_oracle",
                                           "association_oracle",
                                           "gt_observation_uidm"],
                    action="append")
    ap.add_argument("--ckpt", default=str(
        ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/l15/bottlenecks"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="query shard used with --predict-only")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--oracle-iou", type=float, default=0.50)
    ap.add_argument("--predict-only", action="store_true",
                    help="write predictions for the selected query shard")
    ap.add_argument("--eval-only", action="store_true",
                    help="evaluate already-written predictions")
    ap.add_argument("--skip-trackeval", action="store_true")
    args = ap.parse_args()
    if args.predict_only and args.eval_only:
        raise ValueError("--predict-only and --eval-only are mutually exclusive")
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise ValueError("invalid --shard/--num-shards")
    datasets = args.dataset or ["kitti_v1", "kitti_v2", "dance"]
    variants = args.variant or ("semantic_oracle", "association_oracle",
                                "gt_observation_uidm")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    need_model = (not args.eval_only and
                  any(v != "association_oracle" for v in variants))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if need_model and device.type != "cuda":
        raise RuntimeError("L15 bottleneck tool requires CUDA for the UIDM oracle")
    model = load_model(args.ckpt, device) if need_model else None
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for dataset in datasets:
        q_by_seq, frames_by_seq, sizes, gt_root, seqmap = load_dataset(dataset)
        expr_meta = read_expression_meta(dataset)
        proposal, family_stats, query_rows = proposal_stats(
            dataset, q_by_seq, frames_by_seq, expr_meta)
        add_rates(proposal)
        for value in family_stats.values():
            add_rates(value)
        dataset_root = out_root / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        if not args.predict_only:
            (dataset_root / "proposal_stats.json").write_text(
                json.dumps({"dataset": dataset, "oracle_iou": args.oracle_iou,
                            "proposal": proposal, "families": family_stats,
                            "queries": query_rows}, indent=2))
        if args.predict_only:
            flat = [(seq, expr, spec) for seq, items in q_by_seq.items()
                    for expr, spec in items]
            selected = flat[args.shard::args.num_shards]
            q_by_seq = defaultdict(list)
            for seq, expr, spec in selected:
                q_by_seq[seq].append((expr, spec))
            query_rows = [row for row in query_rows
                          if (row["sequence"], row["expression"])
                          in {(seq, expr) for seq, expr, _ in selected}]
        query_lines = [f"{seq}+{expr}" for seq, qitems in q_by_seq.items()
                       for expr, _spec in qitems]
        metrics = {}
        if args.predict_only:
            for variant in variants:
                variant_root = dataset_root / variant
                variant_root.mkdir(parents=True, exist_ok=True)
                for seq, qitems in q_by_seq.items():
                    frames = frames_by_seq[seq]
                    for expr, spec in qitems:
                        entry = expr_meta[(seq, expr)]
                        if variant == "association_oracle":
                            rows = query_rows_association(
                                frames, dataset, entry, args.oracle_iou)
                        else:
                            rows = query_rows_uidm(
                                model, frames, sizes[seq], dataset, entry,
                                spec, args.oracle_iou, variant, device)
                        write_prediction(
                            variant_root / "uidm15" / seq / expr /
                            "predict.txt", rows)
            all_results[dataset] = {
                "query_count": len(query_rows), "metrics": {},
                "shard": args.shard, "num_shards": args.num_shards,
            }
            print(f"[l15oracle] {dataset} shard={args.shard}/"
                  f"{args.num_shards} queries={len(query_rows)}",
                  flush=True)
            continue
        if not args.skip_trackeval:
            for variant in variants:
                variant_root = dataset_root / variant
                if not args.eval_only:
                    variant_root.mkdir(parents=True, exist_ok=True)
                    for seq, qitems in q_by_seq.items():
                        frames = frames_by_seq[seq]
                        for expr, spec in qitems:
                            entry = expr_meta[(seq, expr)]
                            if variant == "association_oracle":
                                rows = query_rows_association(
                                    frames, dataset, entry, args.oracle_iou)
                            else:
                                rows = query_rows_uidm(
                                    model, frames, sizes[seq], dataset, entry,
                                    spec, args.oracle_iou, variant, device)
                            write_prediction(
                                variant_root / "uidm15" / seq / expr /
                                "predict.txt", rows)
                metric, log = evaluate_variant(
                    dataset, variant_root, gt_root, seqmap, query_lines,
                    sizes, "trackeval_l15.log")
                metrics[variant] = {"metrics": metric, "log": str(log)}
            # Per-family semantic-oracle summaries use the same predictions;
            # the family seqmaps are declared diagnostics, never calibration.
            sem_root = dataset_root / "semantic_oracle"
            family_metrics = {}
            for family in sorted(family_stats):
                lines = [f"{row['sequence']}+{row['expression']}"
                         for row in query_rows if row["family"] == family]
                if not lines:
                    continue
                metric, log = evaluate_variant(
                    dataset, sem_root, gt_root, seqmap, lines,
                    sizes, "trackeval_" + hashlib.md5(
                        family.encode()).hexdigest()[:8] + ".log")
                family_metrics[family] = {"metrics": metric, "log": str(log),
                                          "queries": len(lines)}
            metrics["semantic_oracle_by_family"] = family_metrics
        all_results[dataset] = {
            "proposal": proposal,
            "families": family_stats,
            "query_count": len(query_rows),
            "metrics": metrics,
        }
        (dataset_root / "results.json").write_text(
            json.dumps(all_results[dataset], indent=2))
        print(f"[l15oracle] {dataset} queries={len(query_rows)} "
              f"proposal50={proposal['0.5']['object_recall']:.4f} "
              f"metrics={list(metrics)}", flush=True)
    if not args.predict_only:
        (out_root / "all_results.json").write_text(
            json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
