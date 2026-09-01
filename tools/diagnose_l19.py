"""Export per-candidate L19 diagnostics and reusable validation score caches."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from tools.eval_l18_carr import (  # noqa: E402
    BankStore, TextStore, load_model, metadata, trainval_queries,
)
from tools.train_l18_carr import (  # noqa: E402
    frame_features, l19_frame_targets, l19_track_membership_index,
)
from locatemot.rmot.l19_reserve_identity import box_iou  # noqa: E402


CSV_FIELDS = [
    "dataset", "video", "frame", "query", "source", "track_id",
    "observation_group_id", "cross_pool_duplicate", "gt_iou", "membership_label",
    "presence_label", "coverage_state", "coverage_predicted",
    "membership_raw_logit", "presence_raw_logit", "coverage_p0",
    "coverage_p1", "coverage_p2", "coverage_p3", "coverage_contribution",
    "reserve_bias_contribution", "final_score", "no_gate_score", "raw_rank",
    "no_gate_rank", "selected", "tp_fp_fn",
]


def safe_expression(text: str) -> str:
    return text.replace("/", "_")


def load_record(video: str, dance: bool = False) -> dict:
    if dance:
        path = ROOT / "outputs/l8/data/rmot_train" / f"{video}.pkl"
    else:
        path = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
        if not path.exists():
            path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
    return __import__("pickle").load(path.open("rb"))


def group_values(bank: dict, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    tensors = bank["tensors"]
    start = int(tensors["frame_ptr"][frame_index])
    end = int(tensors["frame_ptr"][frame_index + 1])
    if "observation_group_id" in tensors:
        groups = tensors["observation_group_id"][start:end].numpy().astype(np.int64)
        duplicate = tensors.get("cross_pool_duplicate",
                                torch.zeros(end - start, dtype=torch.uint8))[start:end].numpy().astype(np.uint8)
        return groups, duplicate
    # Reconstruct the same fixed training-only grouping rule for an L18 bank.
    boxes = tensors["box"][start:end].numpy()
    clips = tensors["clip"][start:end].numpy().astype(np.float32)
    source = tensors.get("pool_id", torch.zeros(end - start, dtype=torch.long))[start:end].numpy()
    frame = int(tensors["frame_ids"][frame_index])
    base = (frame + 1) * 1_000_000
    main = np.flatnonzero(source == 0)
    reserve = np.flatnonzero(source == 1)
    groups = np.arange(end - start, dtype=np.int64) + base + 1
    duplicate = np.zeros(end - start, np.uint8)
    for ri in reserve:
        candidates = []
        for mi in main:
            overlap = box_iou(boxes[ri], boxes[mi])
            appearance = float(np.dot(
                clips[ri] / max(1e-6, np.linalg.norm(clips[ri])),
                clips[mi] / max(1e-6, np.linalg.norm(clips[mi]))))
            if overlap >= 0.50 or (overlap >= 0.30 and appearance >= 0.82):
                candidates.append((overlap + 0.10 * max(0.0, appearance), mi))
        if candidates:
            _score, mi = max(candidates)
            groups[ri] = groups[mi]
            duplicate[ri] = 1
    return groups, duplicate


def gt_boxes_for_query(video: str, dance: bool) -> dict[int, dict[str, list[float]]]:
    record = load_record(video, dance=dance)
    return {int(frame["frame"]): {str(key): value for key, value in
                                  frame.get("gt_boxes", {}).items()}
            for frame in record["frames"]}


def score_components(model, output: dict, source: torch.Tensor) -> tuple[np.ndarray, ...]:
    final = output["logits"].float()
    membership = output["membership_logits"].float()
    presence = output["presence_logits"].float()
    probabilities = output.get("state_probabilities")
    if probabilities is None:
        probabilities = final.new_zeros(4)
    probabilities = probabilities.float().reshape(-1)
    gate = output.get("coverage_contribution")
    if gate is None:
        raw_gate = output.get("coverage_gate", final.new_zeros(len(final))).float()
        scale = float(getattr(model, "coverage_scale", 1.0))
        gate = raw_gate * scale
    else:
        gate = gate.float()
    bias = output.get("reserve_bias_contribution")
    if bias is None:
        reserve_bias = float(getattr(model, "reserve_bias", 0.0))
        bias = torch.where(source == 1, final.new_tensor(reserve_bias),
                           final.new_zeros(()).float())
    else:
        bias = bias.float()
    no_gate = final - gate - bias
    return (final.detach().cpu().numpy(), membership.detach().cpu().numpy(),
            presence.detach().cpu().numpy(), probabilities.detach().cpu().numpy(),
            gate.detach().cpu().numpy(), bias.detach().cpu().numpy(),
            no_gate.detach().cpu().numpy())


def write_cache(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, np.float64)
    return {"count": int(len(array)), "q01": float(np.quantile(array, .01)),
            "q10": float(np.quantile(array, .10)), "q25": float(np.quantile(array, .25)),
            "q50": float(np.quantile(array, .50)), "q75": float(np.quantile(array, .75)),
            "q90": float(np.quantile(array, .90)), "q99": float(np.quantile(array, .99)),
            "mean": float(array.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("trainval_kitti", "trainval_dance"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--text-root", default="outputs/l18/data/text_cache")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    queries, _gt_root, _seqmap, _sequences, protocol_kind = trainval_queries(args.dataset)
    if args.max_queries:
        queries = queries[:args.max_queries]
    lookup = metadata("dance" if args.dataset == "trainval_dance" else "kitti_v2")
    store = BankStore((ROOT / args.bank_root).resolve(), cache_size=1)
    text_store = TextStore((ROOT / args.text_root).resolve())
    model, checkpoint = load_model(Path(args.checkpoint), device)
    model.eval()
    output_root = (ROOT / args.out_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "cache"
    csv_path = output_root / "candidates.csv.gz"
    handle = gzip.open(csv_path, "wt", newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    state_confusion = Counter()
    source_stats = defaultdict(Counter)
    score_values = defaultdict(list)
    reserve_suppressed = 0
    rank_changes = []
    rank_change_seen = 0
    score_seen = Counter()
    query_summaries = []
    gt_cache = {}
    total_rows = 0
    for query_index, (video, expression, spec) in enumerate(queries):
        entry = dict(lookup.get((video, expression), {
            "expression": expression, "sentence": expression,
            "spec": spec.tolist(),
        }))
        entry["spec"] = spec.tolist()
        text = str(entry.get("sentence", entry.get("expression", expression)))
        tokens, mask = text_store.get(text, device)
        query = torch.as_tensor(np.asarray(spec, np.float32), device=device)
        family = expression_family_vector(text).to(device)
        bank_dataset = "dance_train" if args.dataset == "trainval_dance" else "kitti"
        bank = store.get(bank_dataset, video)
        if "l19_track_membership" not in bank:
            bank["l19_track_membership"] = l19_track_membership_index(bank)
        gt_key = (video, args.dataset == "trainval_dance")
        if gt_key not in gt_cache:
            gt_cache[gt_key] = gt_boxes_for_query(video, args.dataset == "trainval_dance")
        gt_by_frame = gt_cache[gt_key]
        state = {}
        with torch.no_grad():
            query_context = model.query_context(tokens, query, family, mask)
        cache_rows = defaultdict(list)
        frame_rows = []
        tensors = bank["tensors"]
        for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
            features, track_ids, begin, end = frame_features(bank, frame_index, device)
            if end <= begin:
                continue
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    output = model(features, query, family, track_ids, state,
                                   query_tokens=tokens, query_mask=mask,
                                   query_context=query_context)
            state = output["state"]
            source = features.get("pool_id", torch.zeros(
                len(track_ids), dtype=torch.long, device=device)).long()
            final, membership, presence, probabilities, coverage, bias, no_gate = \
                score_components(model, output, source)
            groups, duplicate = group_values(bank, frame_index)
            target = l19_frame_targets(
                bank, begin, end, entry, int(frame_id), bank["l19_track_membership"])
            coverage_state = int(target["state"])
            predicted_state = int(np.argmax(probabilities)) if len(probabilities) else 0
            state_confusion[(coverage_state, predicted_state)] += 1
            ranks_raw = np.empty(len(final), np.int64)
            ranks_no_gate = np.empty(len(final), np.int64)
            if len(final):
                ranks_raw[np.argsort(-final, kind="stable")] = np.arange(len(final)) + 1
                ranks_no_gate[np.argsort(-no_gate, kind="stable")] = np.arange(len(final)) + 1
            boxes = tensors["box"][begin:end].numpy()
            candidate_gt = bank.get("candidate_gt", [None] * len(tensors["track_id"]))[begin:end]
            frame_gt = gt_by_frame.get(int(frame_id), {})
            target_ids = set(target["target_ids"])
            selected = final >= float(args.threshold)
            for local in range(len(final)):
                gt_iou = max((box_iou(boxes[local], frame_gt[gt_id])
                              for gt_id in target_ids if gt_id in frame_gt), default=0.0)
                current_match = float(candidate_gt[local] is not None and
                                      str(candidate_gt[local]) in target_ids)
                label = "TP" if selected[local] and current_match else \
                    "FP" if selected[local] else "FN" if current_match else "TN"
                source_name = "reserve" if int(source[local]) == 1 else "main"
                source_stats[source_name]["rows"] += 1
                source_stats[source_name]["positive"] += int(current_match)
                source_stats[source_name]["selected"] += int(selected[local])
                source_stats[source_name][label] += 1
                score_key = f"{source_name}_{'positive' if current_match else 'negative'}"
                score_seen[score_key] += 1
                if len(score_values[score_key]) < 100000:
                    score_values[score_key].append(float(final[local]))
                if source_name == "reserve" and current_match and no_gate[local] >= args.threshold > final[local]:
                    reserve_suppressed += 1
                if source_name == "reserve" and current_match:
                    rank_change_seen += 1
                    if len(rank_changes) < 100000:
                        rank_changes.append(int(ranks_raw[local] - ranks_no_gate[local]))
                frame_rows.append((int(frame_id), local, source_name, int(track_ids[local]),
                                   groups[local], int(duplicate[local]), gt_iou,
                                   float(target["membership"][local]),
                                   float(target["presence"][local]), coverage_state,
                                   predicted_state, float(membership[local]),
                                   float(presence[local]), probabilities.copy(),
                                   float(coverage[local]), float(bias[local]),
                                   float(final[local]), float(no_gate[local]),
                                   int(ranks_raw[local]), int(ranks_no_gate[local]),
                                   int(selected[local]), label,
                                   boxes[local].astype(np.float32)))
                cache_rows["frame"].append(int(frame_id))
                cache_rows["track_id"].append(int(track_ids[local]))
                cache_rows["box"].append(boxes[local].astype(np.float32))
                cache_rows["source"].append(int(source[local]))
                cache_rows["group"].append(int(groups[local]))
                cache_rows["duplicate"].append(int(duplicate[local]))
                cache_rows["gt_iou"].append(gt_iou)
                cache_rows["membership_label"].append(float(target["membership"][local]))
                cache_rows["presence_label"].append(float(target["presence"][local]))
                cache_rows["coverage_state"].append(coverage_state)
                cache_rows["coverage_predicted"].append(predicted_state)
                cache_rows["membership"].append(float(membership[local]))
                cache_rows["presence"].append(float(presence[local]))
                cache_rows["coverage"].append(float(coverage[local]))
                cache_rows["bias"].append(float(bias[local]))
                cache_rows["raw"].append(float(final[local]))
                cache_rows["no_gate"].append(float(no_gate[local]))
        cache_path = cache_root / args.dataset / video / f"{safe_expression(expression)}.npz"
        write_cache(cache_path, {
            "frame": np.asarray(cache_rows["frame"], np.int32),
            "track_id": np.asarray(cache_rows["track_id"], np.int64),
            "box": np.asarray(cache_rows["box"], np.float32).reshape(-1, 4),
            "source": np.asarray(cache_rows["source"], np.int8),
            "group": np.asarray(cache_rows["group"], np.int64),
            "duplicate": np.asarray(cache_rows["duplicate"], np.uint8),
            "gt_iou": np.asarray(cache_rows["gt_iou"], np.float32),
            "membership_label": np.asarray(cache_rows["membership_label"], np.float32),
            "presence_label": np.asarray(cache_rows["presence_label"], np.float32),
            "coverage_state": np.asarray(cache_rows["coverage_state"], np.int8),
            "coverage_predicted": np.asarray(cache_rows["coverage_predicted"], np.int8),
            "membership": np.asarray(cache_rows["membership"], np.float32),
            "presence": np.asarray(cache_rows["presence"], np.float32),
            "coverage": np.asarray(cache_rows["coverage"], np.float32),
            "bias": np.asarray(cache_rows["bias"], np.float32),
            "raw": np.asarray(cache_rows["raw"], np.float32),
            "no_gate": np.asarray(cache_rows["no_gate"], np.float32),
        })
        for row in frame_rows:
            frame_id, _local, source_name, track_id, group, duplicate, gt_iou, m_label, p_label, state_value, predicted_state, m_raw, p_raw, probs, cov, b, raw, no_g, r_raw, r_no, is_selected, label, box = row
            frame_number = frame_id if args.dataset == "trainval_dance" else frame_id + 1
            writer.writerow({
                "dataset": args.dataset, "video": video, "frame": frame_number,
                "query": expression, "source": source_name, "track_id": track_id,
                "observation_group_id": group, "cross_pool_duplicate": duplicate,
                "gt_iou": gt_iou, "membership_label": m_label,
                "presence_label": p_label, "coverage_state": state_value,
                "coverage_predicted": predicted_state, "membership_raw_logit": m_raw,
                "presence_raw_logit": p_raw, "coverage_p0": probs[0] if len(probs) > 0 else 0.0,
                "coverage_p1": probs[1] if len(probs) > 1 else 0.0,
                "coverage_p2": probs[2] if len(probs) > 2 else 0.0,
                "coverage_p3": probs[3] if len(probs) > 3 else 0.0,
                "coverage_contribution": cov, "reserve_bias_contribution": b,
                "final_score": raw, "no_gate_score": no_g, "raw_rank": r_raw,
                "no_gate_rank": r_no, "selected": is_selected, "tp_fp_fn": label,
            })
        total_rows += len(frame_rows)
        query_summaries.append({"video": video, "query": expression,
                                "rows": len(frame_rows),
                                "reserve_positive_suppressed": sum(
                                    1 for row in frame_rows if row[2] == "reserve" and
                                    row[21] == "TP" and row[17] >= args.threshold > row[16])})
        print(f"[l19-diagnose] {args.dataset} {query_index + 1}/{len(queries)} "
              f"video={video} rows={len(frame_rows)}", flush=True)
    handle.close()
    for source_name, values in source_stats.items():
        values["recall"] = values["TP"] / max(1, values["TP"] + values["FN"])
        values["precision"] = values["TP"] / max(1, values["TP"] + values["FP"])
        values["score_sample_count"] = len(score_values.get(
            f"{source_name}_positive", [])) + len(score_values.get(
                f"{source_name}_negative", []))
    summary = {
        "dataset": args.dataset, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"), "threshold": args.threshold,
        "queries": len(queries), "candidate_rows": total_rows,
        "state_confusion": {f"{key[0]}->{key[1]}": value
                             for key, value in sorted(state_confusion.items())},
        "source_stats": {key: dict(value) for key, value in source_stats.items()},
        "score_quantiles": {key: {**quantiles(value),
                                   "total_seen": score_seen[key]}
                            for key, value in score_values.items()},
        "reserve_positive_suppressed_by_gate": reserve_suppressed,
        "reserve_positive_rank_change_raw_minus_no_gate": {
            **quantiles(rank_changes), "total_seen": rank_change_seen},
        "cache_root": str(cache_root), "csv": str(csv_path),
        "query_summaries": query_summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# L19 candidate diagnostics: {args.dataset}", "",
             f"Checkpoint: `{args.checkpoint}`; threshold: `{args.threshold}`.",
             f"Candidates: **{total_rows}** across **{len(queries)}** queries.", "",
             "## Four-state confusion", "", "| true | predicted | count |",
             "|---:|---:|---:|"]
    lines.extend(f"| {key.split('->')[0]} | {key.split('->')[1]} | {value} |"
                 for key, value in summary["state_confusion"].items())
    lines.extend(["", "## Source score distributions", "",
                  "| group | count | q10 | q50 | q90 |", "|---|---:|---:|---:|---:|"])
    for key, value in summary["score_quantiles"].items():
        lines.append(f"| {key} | {value.get('count', 0)} | {value.get('q10', 0):.5f} | "
                     f"{value.get('q50', 0):.5f} | {value.get('q90', 0):.5f} |")
    lines.extend(["", f"Reserve positives suppressed by coverage gate: **{reserve_suppressed}**.",
                  "", "Machine-readable rows are in `candidates.csv.gz`; per-query numeric caches are under `cache/`."])
    (output_root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "dataset", "queries", "candidate_rows", "reserve_positive_suppressed_by_gate")}, indent=2))


if __name__ == "__main__":
    main()
