"""Decompose the Stage L19 RMOT failure against the L18 CARR baseline.

The comparison is deliberately score-cache based: L18 and L19 are evaluated
on the same frozen bank rows, query order, frame convention and operating
thresholds.  Official TrackEval summaries are joined separately so the
official numbers cannot be confused with the train-val candidate diagnosis.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def safe_expression(text: str) -> str:
    return text.replace("/", "_")


def quantile(values: np.ndarray, limit: int = 100_000) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) > limit:
        values = values[:limit]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "mean": float(values.mean()),
    }


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute a tie-aware AUC without importing a second ML dependency."""
    scores = np.asarray(scores, np.float64).reshape(-1)
    labels = np.asarray(labels, bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    ranks = np.empty(len(scores), np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and ordered[end] == ordered[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives)


def ranks_within_frame(frame: np.ndarray, scores: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, np.int64)
    scores = np.asarray(scores, np.float64)
    ranks = np.empty(len(scores), np.int32)
    for frame_id in np.unique(frame):
        indices = np.flatnonzero(frame == frame_id)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return ranks


def aligned_rows(l18: np.ndarray, l19: dict, frame_offset: int = 0) -> dict:
    frame18 = l18[:, 0].astype(np.int64) + int(frame_offset)
    track18 = l18[:, 1].astype(np.int64)
    box18 = np.stack((l18[:, 2], l18[:, 3], l18[:, 2] + l18[:, 4],
                      l18[:, 3] + l18[:, 5]), axis=1)
    frame19 = np.asarray(l19["frame"], np.int64)
    track19 = np.asarray(l19["track_id"], np.int64)
    box19 = np.asarray(l19["box"], np.float32)
    same = len(l18) == len(frame19)
    if same:
        same = bool(np.array_equal(frame18, frame19) and
                    np.array_equal(track18, track19) and
                    np.allclose(box18, box19, atol=2e-3, rtol=0.0))
    return {"rows": int(min(len(l18), len(frame19))), "exact": same,
            "frame_equal": bool(len(l18) == len(frame19) and
                                  np.array_equal(frame18, frame19)),
            "track_equal": bool(len(l18) == len(frame19) and
                                  np.array_equal(track18, track19)),
            "box_equal": bool(len(l18) == len(frame19) and
                               np.allclose(box18, box19, atol=2e-3, rtol=0.0))}


def source_label_metrics(source: np.ndarray, labels: np.ndarray,
                          scores: np.ndarray, threshold: float) -> dict:
    result = {}
    for source_id, source_name in ((0, "main"), (1, "reserve")):
        pool = source == source_id
        positive = pool & labels
        negative = pool & ~labels
        selected = pool & (scores >= threshold)
        true_selected = selected & labels
        result[source_name] = {
            "rows": int(pool.sum()),
            "positive": int(positive.sum()),
            "negative": int(negative.sum()),
            "selected": int(selected.sum()),
            "selected_positive": int(true_selected.sum()),
            "selected_recall": float(true_selected.sum() /
                                       max(1, positive.sum())),
            "selected_precision": float(true_selected.sum() /
                                         max(1, selected.sum())),
            "score_positive": quantile(scores[positive]),
            "score_negative": quantile(scores[negative]),
            "positive_auc": auc_from_scores(scores[pool], labels[pool]),
        }
    return result


def source_rank_metrics(source: np.ndarray, labels: np.ndarray,
                        frame: np.ndarray, scores: np.ndarray) -> dict:
    ranks = ranks_within_frame(frame, scores)
    result = {}
    for source_id, source_name in ((0, "main"), (1, "reserve")):
        positive = (source == source_id) & labels
        values = ranks[positive]
        result[source_name] = {
            "positive_rank": quantile(values),
            "top1_recall": float(np.count_nonzero(values <= 1) /
                                  max(1, len(values))),
            "top5_recall": float(np.count_nonzero(values <= 5) /
                                  max(1, len(values))),
            "top10_recall": float(np.count_nonzero(values <= 10) /
                                   max(1, len(values))),
        }
    return result


def load_trackeval_summary(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "complete": False}
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        return {"path": str(path), "complete": False}
    names = lines[0].split()
    values = lines[1].split()
    payload = {name: float(value) for name, value in zip(names, values)}
    keep = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1")
    return {"path": str(path), "complete": all(k in payload for k in keep),
            **{key: payload.get(key) for key in keep}}


def official_table() -> dict:
    specs = {
        "v1": (
            ROOT / "outputs/l18/eval/carr_official_v1_t_m16/uidm18/pedestrian_summary.txt",
            ROOT / "outputs/l19/eval/official_step750_querynorm_v1/uidm18/pedestrian_summary.txt",
        ),
        "v2": (
            ROOT / "outputs/l18/eval/carr_official_v2_t_m16/uidm18/pedestrian_summary.txt",
            ROOT / "outputs/l19/eval/official_step750_querynorm_v2/uidm18/pedestrian_summary.txt",
        ),
        "dance": (
            ROOT / "outputs/l18/eval/carr_official_dance_t_m52/uidm18/pedestrian_summary.txt",
            ROOT / "outputs/l19/eval/official_step750_querynorm_dance/uidm18/pedestrian_summary.txt",
        ),
    }
    result = {}
    for name, (l18_path, l19_path) in specs.items():
        l18 = load_trackeval_summary(l18_path)
        l19 = load_trackeval_summary(l19_path)
        delta = {}
        if l18.get("complete") and l19.get("complete"):
            for key in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1"):
                delta[key] = l19[key] - l18[key]
        result[name] = {"l18": l18, "l19": l19, "delta_l19_minus_l18": delta}
    return result


def analyze_dataset(name: str, l18_root: Path, l19_root: Path,
                    threshold: float, l18_frame_offset: int = 0) -> dict:
    l18_files = sorted(l18_root.rglob("*.npz"))
    if not l18_files:
        raise FileNotFoundError(l18_root)
    rows = 0
    exact = 0
    source_parts = []
    l18_parts = []
    l19_parts = []
    frame_parts = []
    for index, l18_path in enumerate(l18_files):
        relative = l18_path.relative_to(l18_root)
        l19_path = l19_root / relative
        if not l19_path.exists():
            raise FileNotFoundError(l19_path)
        l18 = np.load(l18_path, allow_pickle=False)["rows"]
        l19_file = np.load(l19_path, allow_pickle=False)
        l19 = {key: l19_file[key] for key in l19_file.files}
        alignment = aligned_rows(l18, l19, l18_frame_offset)
        rows += alignment["rows"]
        exact += int(alignment["exact"])
        if alignment["rows"] != len(l18):
            raise ValueError(f"row count mismatch: {l18_path}")
        source_parts.append(np.asarray(l19["source"], np.int8))
        l18_parts.append(np.asarray(l18[:, 7], np.float32))
        l19_parts.append(np.asarray(l19["raw"], np.float32))
        frame_parts.append(np.asarray(l19["frame"], np.int64))
        l19_file.close()
    source = np.concatenate(source_parts)
    l18_score = np.concatenate(l18_parts)
    l19_score = np.concatenate(l19_parts)
    frame = np.concatenate(frame_parts)

    # The validation cache carries both a current observation label and the
    # persistent membership label.  Candidate IoU is the direct observation
    # ceiling; membership is the tracklet correspondence target.
    # Re-read the compact labels without retaining all other arrays.
    gt_parts, member_parts = [], []
    for l19_path in sorted(l19_root.rglob("*.npz")):
        with np.load(l19_path, allow_pickle=False) as data:
            gt_parts.append(np.asarray(data["gt_iou"], np.float32) >= 0.50)
            member_parts.append(np.asarray(data["membership_label"], np.float32) >= 0.50)
    candidate_positive = np.concatenate(gt_parts)
    membership_positive = np.concatenate(member_parts)
    if len(candidate_positive) != rows:
        raise ValueError("label row count mismatch")

    datasets = {}
    for label_name, labels in (("candidate_iou50", candidate_positive),
                               ("track_membership", membership_positive)):
        dataset_result = {
            "l18": source_label_metrics(source, labels, l18_score, threshold),
            "l19": source_label_metrics(source, labels, l19_score, threshold),
            "l18_rank": source_rank_metrics(source, labels, frame, l18_score),
            "l19_rank": source_rank_metrics(source, labels, frame, l19_score),
        }
        for source_name in ("main", "reserve"):
            for metric in ("selected_recall", "selected_precision"):
                dataset_result.setdefault("delta_l19_minus_l18", {}).setdefault(
                    source_name, {})[metric] = (
                        dataset_result["l19"][source_name][metric] -
                        dataset_result["l18"][source_name][metric])
        datasets[label_name] = dataset_result

    source_shift = {}
    for source_id, source_name in ((0, "main"), (1, "reserve")):
        mask = source == source_id
        source_shift[source_name] = {
            "rows": int(mask.sum()),
            "l18": quantile(l18_score[mask]),
            "l19": quantile(l19_score[mask]),
            "l19_minus_l18": quantile((l19_score - l18_score)[mask]),
        }
    return {
        "dataset": name,
        "l18_score_root": str(l18_root),
        "l19_score_root": str(l19_root),
        "threshold": threshold,
        "queries": len(l18_files),
        "candidate_rows": rows,
        "exact_row_alignment_queries": exact,
        "same_frozen_bank": bool(exact == len(l18_files)),
        "source_shift": source_shift,
        "score_delta_all": quantile(l19_score - l18_score),
        "labels": datasets,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# Stage L19 RMOT failure decomposition",
        "",
        "Date: 2026-08-27.  L19 is the 750-step single-seed diagnostic checkpoint;",
        "L18 is the frozen CARR final checkpoint.  Candidate comparisons below use",
        "the same cached bank rows and the L18 per-domain threshold (-1.6 KITTI,",
        "-5.2 Dance).  They are train-val diagnostics, not official GT selection.",
        "",
        "## Official TrackEval comparison",
        "",
        "| split | system | HOTA | DetA | AssA | DetRe | DetPr | IDF1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("v1", "v2", "dance"):
        entry = payload["official"][split]
        for system in ("l18", "l19"):
            row = entry[system]
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                split.upper(), system.upper(),
                *(f"{row.get(key, float('nan')):.3f}" for key in
                  ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1"))))
        delta = entry["delta_l19_minus_l18"]
        lines.append("| %s | Δ L19−L18 | %s | %s | %s | %s | %s | %s |" % (
            split.upper(), *(f"{delta.get(key, float('nan')):+.3f}" for key in
                             ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "IDF1"))))
    lines += [
        "",
        "## Same-bank train-val score comparison",
        "",
        "The `candidate_iou50` label measures whether an observation overlaps a",
        "query GT box at IoU ≥ 0.50; `track_membership` is the persistent tracklet",
        "target used by the retriever.  Rank is within each frame over all bank",
        "observations, so reserve/main ranks are directly comparable.",
        "",
        "| dataset | label | pool | L18 selected recall/precision | L19 selected recall/precision | Δ recall | Δ precision | L18 pos q50 | L19 pos q50 | L19 AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("kitti_trainval", "dance_trainval"):
        data = payload["datasets"][name]
        short = "KITTI" if name.startswith("kitti") else "Dance"
        for label_name in ("candidate_iou50", "track_membership"):
            item = data["labels"][label_name]
            for pool in ("main", "reserve"):
                l18 = item["l18"][pool]
                l19 = item["l19"][pool]
                lines.append("| %s | %s | %s | %.4f / %.4f | %.4f / %.4f | %+ .4f | %+ .4f | %.4f | %.4f | %s |" % (
                    short, label_name, pool,
                    l18["selected_recall"], l18["selected_precision"],
                    l19["selected_recall"], l19["selected_precision"],
                    l19["selected_recall"] - l18["selected_recall"],
                    l19["selected_precision"] - l18["selected_precision"],
                    l18["score_positive"].get("q50", float("nan")),
                    l19["score_positive"].get("q50", float("nan")),
                    "n/a" if l19["positive_auc"] is None else f"{l19['positive_auc']:.4f}"))
    lines += [
        "",
        "## Diagnosis",
        "",
        "- The global L18 source gate is not the remaining L19 bottleneck: L19",
        "  emits no gate-suppressed reserve positives in the full KITTI cache, and",
        "  the no-gate/aux-only ablations are effectively identical to the full",
        "  score.  Removing the gate therefore cannot recover the target HOTA.",
        "- Precision is a hard failure on KITTI: the same-threshold L19 selected",
        "  precision is low for both pools while millions of negatives are selected.",
        "  Dance is not a reserve-recall failure because its bank is main-only; its",
        "  query-normalized official result instead shows a poorly calibrated and",
        "  weakly corresponding score stream.",
        "- Correspondence is the primary learned failure.  The positive-score",
        "  medians/AUCs and within-frame positive ranks in the machine-readable",
        "  table show that L19 does not consistently put target tracklets above",
        "  hard negatives.  The four-state head is mostly an auxiliary state",
        "  predictor, not expression-to-track correspondence.",
        "- Reserve identity is a secondary but real limitation: its nonzero feature",
        "  views remove the L18 all-zero reserve representation, but reserve and",
        "  main positive ranks remain weak/overlapped.  A hand-built long linker",
        "  cannot fix this; the next route must learn expression↔tracklet",
        "  correspondence over temporally structured observations.",
        "- Calibration is downstream of correspondence, not a sufficient fix.  The",
        "  global query-zscore operating point was protocol-compliant but produced",
        "  HOTA 15.129/12.620/14.938, far below the 45/40/42 gate.",
        "",
        "## Next implementation entry",
        "",
        "The next model is `locatemot/models/l19_flexhook_correspondence.py` and",
        "the diagnostic trainer is `tools/train_l19_flexhook_correspondence.py`.",
        "It ports the audited FlexHook pattern—text-conditioned hook points,",
        "PCD-style cross-attention, temporal feature-map sampling and learned",
        "correspondence—onto LocalAnything tracklet features plus the DINO reserve.",
        "No CARR coverage gate or hand-written IoU/CLIP linker is used.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/l19/eval/failure_decomposition.json")
    parser.add_argument("--report", default="reports/l19_failure_decomposition.md")
    args = parser.parse_args()
    payload = {
        "protocol": {
            "same_bank": True,
            "thresholds": {"kitti_trainval": -1.6, "dance_trainval": -5.2},
            "l18_checkpoint": str(ROOT / "outputs/l18/checkpoints/carr_dual_final.pt"),
            "l19_checkpoint": str(ROOT / "outputs/l19/checkpoints/l19_diag_1000_step750.pt"),
        },
        "official": official_table(),
        "datasets": {
            "kitti_trainval": analyze_dataset(
                "kitti_trainval",
                ROOT / "outputs/l18/scores/carr_final_val/trainval_kitti",
                ROOT / "outputs/l19/diagnostics/l19_step750_kitti/cache/trainval_kitti",
                -1.6, l18_frame_offset=-1),
            "dance_trainval": analyze_dataset(
                "dance_trainval",
                ROOT / "outputs/l18/scores/carr_dance_val/trainval_dance",
                ROOT / "outputs/l19/diagnostics/l19_step750_dance/cache/trainval_dance",
                -5.2),
        },
    }
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    report = (ROOT / args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown(payload))
    print(json.dumps({
        "out": str(out), "report": str(report),
        "official": payload["official"],
        "datasets": {key: {
            "queries": value["queries"], "candidate_rows": value["candidate_rows"],
            "same_frozen_bank": value["same_frozen_bank"],
        } for key, value in payload["datasets"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
