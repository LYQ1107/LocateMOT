#!/usr/bin/env python3
"""L51-B1 semantic validation for L29 control versus the step-500 residual.

Only L49 calibration/validation units are opened.  Calibration labels fit the
per-domain thresholds; validation labels are used once for held-out reporting,
never to select a checkpoint, branch, or threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder  # noqa: E402
from locatemot.models.l51_streaming_crop_adapter import L51StreamingCropAdapter  # noqa: E402
from locatemot.rmot.l49_data import (  # noqa: E402
    L29_CHECKPOINT,
    TEXT_CACHE,
    load_bank,
    sha256_file,
)
from tools.eval_l49_validation import fit_threshold, source_masks, summarize  # noqa: E402
from tools.eval_l49_validation import l29_score  # noqa: E402
from tools.train_l49_kitti_rmot import build_teacher_cache  # noqa: E402
from tools.train_l51_streaming_crop_adapter import (  # noqa: E402
    FAST_MANIFEST,
    StreamingClipPatches,
    forward_item,
    materialize_units,
)


DATA = ROOT / "outputs/l49/data"
CHECKPOINT = ROOT / "outputs/l51/train/b1_fullfit500_bound0p5/step500/checkpoint_l51_b1_step500.pt"
CONTRACT = ROOT / "outputs/l51/audit/data_contract.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


class BankStore:
    def __init__(self, limit: int = 2):
        self.limit = int(limit)
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, dataset: str, video: str) -> dict:
        key = f"{dataset}|{video}"
        if key not in self.cache:
            self.cache[key] = load_bank(dataset, video)
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]


class ValidationTeacher:
    """Exact L29 control path that also supports validation units."""
    def __init__(self, text: dict, device: torch.device):
        self.device = device
        self.text = text
        self.model = L29FrameMembershipSetDecoder().to(device)
        self.model.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device,
                                              weights_only=False)["model"], strict=True)
        self.model.eval()
        self.cache: dict[str, dict] = {}

    @torch.inference_mode()
    def score(self, unit: dict, bank: dict) -> torch.Tensor:
        path = str(bank["path"])
        if path not in self.cache:
            self.cache[path] = build_teacher_cache(bank)
        values = l29_score(self.model, self.cache[path], bank, unit, self.text, self.device)
        return torch.as_tensor(values, dtype=torch.float32)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row: dict) -> tuple[str, str, int, int]:
    return (str(row["dataset"]), str(row["video"]), int(row["query_id"]), int(row["frame_id"]))


def assert_split(units: list[dict], expected: str) -> None:
    if any(str(row.get("split", expected)) != expected for row in units):
        raise AssertionError(f"unexpected split in {expected} units")


def make_records(model: L51StreamingCropAdapter | None, units: list[dict], text: dict,
                 teacher: ValidationTeacher, device: torch.device, store: BankStore) -> list[dict]:
    records = []
    for unit in units:
        item = materialize_units([unit], text, teacher)[0]
        teacher_score = item["teacher"].numpy().astype(np.float32)
        if model is None:
            student_score = teacher_score.copy()
            residual = np.zeros_like(student_score)
        else:
            with torch.inference_mode():
                output, patch = forward_item(model, store.encoder, item, text, device)
            student_score = output["final_logit"].float().cpu().numpy()
            residual = output["residual"].float().cpu().numpy()
            del output, patch
        labels = item["y"].numpy().astype(bool)
        if len(student_score) != int(unit["candidate_count"]) or len(labels) != len(student_score):
            raise AssertionError(f"candidate key/count drift: {unit['unit_key']}")
        bank = store.get(str(unit["dataset"]), str(unit["video"]))
        records.append({
            "dataset": str(unit["dataset"]), "video": str(unit["video"]),
            "query_id": int(unit["query_id"]), "frame_id": int(unit["frame_id"]),
            "category": str(unit["category"]), "unit_key": str(unit["unit_key"]),
            "candidate_count": int(len(labels)), "positive_count": int(labels.sum()),
            "teacher_score": teacher_score, "score": student_score, "residual": residual,
            "label": labels,
            "sources": source_masks(bank, int(unit["begin"]), int(unit["end"])),
        })
        del item
    return records


class EncoderStore(BankStore):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


def rank_flips(records: list[dict]) -> dict:
    counts = defaultdict(int)
    for row in records:
        label = np.asarray(row["label"], dtype=bool)
        teacher = np.asarray(row["teacher_score"], dtype=np.float64)
        student = np.asarray(row["score"], dtype=np.float64)
        pos = np.flatnonzero(label); neg = np.flatnonzero(~label)
        if not len(pos) or not len(neg):
            continue
        teacher_order = (teacher[pos, None] > teacher[None, neg]).reshape(-1)
        student_order = (student[pos, None] > student[None, neg]).reshape(-1)
        counts["frame_units"] += 1
        counts["pair_count"] += int(len(teacher_order))
        counts["teacher_correct_pairs"] += int(teacher_order.sum())
        counts["student_correct_pairs"] += int(student_order.sum())
        counts["teacher_correct_student_flip"] += int((teacher_order & ~student_order).sum())
        counts["teacher_error_student_correction"] += int((~teacher_order & student_order).sum())
        counts["total_rank_flip"] += int((teacher_order != student_order).sum())
    total = max(1, counts["pair_count"])
    return {**dict(counts),
            "teacher_correct_student_flip_rate": counts["teacher_correct_student_flip"] / total,
            "teacher_error_student_correction_rate": counts["teacher_error_student_correction"] / total,
            "total_rank_flip_rate": counts["total_rank_flip"] / total}


def score_stats(records: list[dict]) -> dict:
    teacher = np.concatenate([row["teacher_score"] for row in records]) if records else np.zeros(0)
    student = np.concatenate([row["score"] for row in records]) if records else np.zeros(0)
    residual = np.concatenate([row["residual"] for row in records]) if records else np.zeros(0)
    def stats(values):
        return {"count": int(len(values)), "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std()) if len(values) else None,
                "abs_max": float(np.abs(values).max()) if len(values) else None,
                "q95_abs": float(np.quantile(np.abs(values), .95)) if len(values) else None}
    return {"teacher": stats(teacher), "l51": stats(student), "residual": stats(residual),
            "student_minus_teacher_mean": float((student - teacher).mean()) if len(student) else None,
            "student_minus_teacher_abs_max": float(np.abs(student - teacher).max()) if len(student) else None}


def audit_records(records: list[dict]) -> dict:
    keys = [key(row) for row in records]
    bad_shapes = sum(int(len(row["score"]) != len(row["teacher_score"]) or
                         len(row["score"]) != len(row["label"])) for row in records)
    return {"record_count": len(records), "duplicate_key_count": len(keys) - len(set(keys)),
            "shape_mismatch_count": bad_shapes,
            "candidate_rows": int(sum(len(row["score"]) for row in records)),
            "positive_rows": int(sum(row["label"].sum() for row in records)),
            "full_candidate_set_preserved": True, "candidate_truncation": False}


def aggregate_summary(records: list[dict], thresholds: dict[str, float]) -> dict:
    """Aggregate domain-calibrated decisions without changing frame rankings."""
    shifted = []
    for row in records:
        copy = dict(row)
        copy["score"] = np.asarray(row["score"], dtype=np.float32) - float(thresholds[row["dataset"]])
        shifted.append(copy)
    summary = summarize(shifted, 0.0)
    summary["threshold"] = "per-domain calibration thresholds (scores shifted to zero)"
    return summary


def jsonable(row: dict) -> dict:
    return {"dataset": row["dataset"], "video": row["video"], "query_id": row["query_id"],
            "frame_id": row["frame_id"], "unit_key": row["unit_key"], "category": row["category"],
            "candidate_count": row["candidate_count"], "positive_count": row["positive_count"],
            "teacher_score": row["teacher_score"].tolist(), "score": row["score"].tolist(),
            "residual": row["residual"].tolist(), "label": row["label"].tolist(),
            "sources": {name: values.tolist() for name, values in row["sources"].items()}}


def domain_records(records: list[dict], domain: str) -> list[dict]:
    return [row for row in records if row["dataset"] == domain]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    if sha256_file(FAST_MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("fixed manifest SHA mismatch")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.time()
    calibration = load_jsonl(DATA / "calibration_units.jsonl")
    validation = load_jsonl(DATA / "validation_units.jsonl")
    assert_split(calibration, "calibration"); assert_split(validation, "validation")
    expected = {"refer_kitti_v1": {"calibration": {"0016"}, "validation": {"0004", "0018"}},
                "refer_kitti_v2": {"calibration": {"0015"}, "validation": {"0016", "0017", "0020"}}}
    actual = {domain: {"calibration": {str(x["video"]) for x in calibration if x["dataset"] == domain},
                       "validation": {str(x["video"]) for x in validation if x["dataset"] == domain}}
             for domain in expected}
    if actual != expected:
        raise AssertionError(f"fixed video mismatch: {actual}")
    contract = json.loads(CONTRACT.read_text())
    overlap = contract["video_split_audit"]["cross_domain_overlap_warning"]
    shared_videos = set()
    for name, values in overlap.items():
        if name != "present" and isinstance(values, list):
            shared_videos.update(str(value) for value in values)
    text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    requested = torch.device(args.device)
    device = requested if requested.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")
    teacher = ValidationTeacher(text, device)
    eval_units = calibration[:1] + validation[:1] if args.contract_only else calibration + validation
    l29_records = make_records(None, eval_units, text, teacher, device, BankStore())
    if args.contract_only:
        result = {"format": "locatemot-l51-b1-semantic-contract-v1", "status": "pass",
                  "scope": "one calibration and one validation unit; no screening/test",
                  "calibration": audit_records(l29_records[:1]), "validation": audit_records(l29_records[1:]),
                  "official_test_labels_read": False, "screening_gt_used": False}
        (out / "semantic.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2)); return
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]["model"]
    model = L51StreamingCropAdapter(hidden=int(cfg["hidden"]), heads=int(cfg["heads"]),
                                    layers=int(cfg["layers"]), residual_bound=float(cfg["residual_bound"]))
    model.load_state_dict(checkpoint["model"], strict=True); model.to(device).eval()
    encoder = StreamingClipPatches(device)
    l51_store = EncoderStore(encoder)
    l51_records = make_records(model, calibration + validation, text, teacher, device, l51_store)
    results = {}
    for name, records in (("L29_control", l29_records), ("L51_step500", l51_records)):
        results[name] = {"calibration": {}, "validation": {}, "rank_flips": {}, "score_stats": {}}
        for domain in ("refer_kitti_v1", "refer_kitti_v2"):
            cal = [r for r in records[:len(calibration)] if r["dataset"] == domain]
            val = [r for r in records[len(calibration):] if r["dataset"] == domain]
            if name == "L29_control":
                threshold = fit_threshold(cal)["threshold"]
            else:
                threshold = fit_threshold(cal)["threshold"]
            results[name]["calibration"][domain] = {"threshold": threshold, "summary": summarize(cal, threshold)}
            results[name]["validation"][domain] = {"threshold_frozen_from_calibration": threshold,
                                                    "summary": summarize(val, threshold)}
        thresholds = {domain: results[name]["calibration"][domain]["threshold"]
                      for domain in ("refer_kitti_v1", "refer_kitti_v2")}
        results[name]["calibration"]["aggregate"] = {"summary": aggregate_summary(records[:len(calibration)], thresholds),
                                                       "thresholds_by_domain": thresholds}
        results[name]["validation"]["aggregate"] = {"summary": aggregate_summary(records[len(calibration):], thresholds),
                                                      "thresholds_by_domain": thresholds}
        results[name]["rank_flips"] = {"calibration": rank_flips(records[:len(calibration)]),
                                        "validation": rank_flips(records[len(calibration):])}
        results[name]["score_stats"] = {"calibration": score_stats(records[:len(calibration)]),
                                         "validation": score_stats(records[len(calibration):])}
        results[name]["record_audit"] = {"calibration": audit_records(records[:len(calibration)]),
                                          "validation": audit_records(records[len(calibration):])}
    # The same unit order is used for both models, so this is a direct score
    # contract check rather than a model-selection operation.
    l51_by_key = {key(row): row for row in l51_records}
    l29_by_key = {key(row): row for row in l29_records}
    shared_excluded = [row for row in l51_records[len(calibration):] if row["video"] in shared_videos]
    strict_l51 = [row for row in l51_records[len(calibration):] if row["video"] not in shared_videos]
    strict_l29 = [row for row in l29_records[len(calibration):] if row["video"] not in shared_videos]
    strict_slice = {}
    for name, rows in (("L29_control", strict_l29), ("L51_step500", strict_l51)):
        strict_slice[name] = {domain: {"summary_at_domain_calibration_threshold": None,
                                       "unit_count": len(domain_records(rows, domain))}
                              for domain in ("refer_kitti_v1", "refer_kitti_v2")}
        for domain in strict_slice[name]:
            threshold = results[name]["calibration"][domain]["threshold"]
            strict_slice[name][domain]["summary_at_domain_calibration_threshold"] = summarize(domain_records(rows, domain), threshold)
    score_records_path = out / "score_records.jsonl"
    with score_records_path.open("w") as handle:
        for split, rows in (("calibration", l51_records[:len(calibration)]), ("validation", l51_records[len(calibration):])):
            for row in rows:
                handle.write(json.dumps({"split": split, **jsonable(row)}, ensure_ascii=False) + "\n")
    result = {
        "format": "locatemot-l51-b1-semantic-validation-v1", "status": "pass",
        "scope": "calibration-fitted thresholds and fixed validation reporting; no screening/test",
        "started_at_unix": started, "completed_at_unix": time.time(), "device": str(device),
        "checkpoint": str(CHECKPOINT.resolve()), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "l29_checkpoint": str(L29_CHECKPOINT.resolve()), "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
        "manifest_sha256": sha256_file(FAST_MANIFEST), "calibration_units": len(calibration),
        "validation_units": len(validation), "fixed_videos": expected,
        "within_domain_video_disjoint": contract["video_split_audit"]["within_domain_disjoint"],
        "cross_domain_shared_videos": sorted(shared_videos),
        "strict_exclusion_definition": "descriptive validation slice excluding any video ID listed in cross-domain overlap audit",
        "strict_excluded_validation_unit_count": len(shared_excluded),
        "strict_remaining_validation_unit_count": len(strict_l51),
        "results": results, "strict_cross_domain_exclusion": strict_slice,
        "score_records": str(score_records_path.resolve()),
        "official_test_labels_read": False, "screening_gt_used": False,
        "ordinary_mot_ovmot_touched": False, "raw_cache_written": False,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id"],
        "null_false_acceptance": "N/A (L51 model has no NULL head)",
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_language_mask": "UNALIGNED/not claimed",
    }
    (out / "semantic.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "format": "locatemot-l51-b1-semantic-provenance-v1", "started_at_unix": started,
        "completed_at_unix": time.time(), "calibration_source": str((DATA / "calibration_units.jsonl").resolve()),
        "validation_source": str((DATA / "validation_units.jsonl").resolve()),
        "screening_source_opened": False, "official_test_source_opened": False,
        "manifest_sha256": sha256_file(FAST_MANIFEST), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT), "threshold_source": "calibration labels only",
    }, indent=2) + "\n")
    print(json.dumps({"status": "pass", "calibration_units": len(calibration), "validation_units": len(validation),
                      "strict_excluded_validation_unit_count": len(shared_excluded),
                      "checkpoint": str(CHECKPOINT.resolve()), "elapsed_sec": time.time() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
