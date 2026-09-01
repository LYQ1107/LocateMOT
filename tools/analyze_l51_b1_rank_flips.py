#!/usr/bin/env python3
"""CPU-only rank-flip and residual saturation audit for the L51 B1 cache."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def summarize(rows, bound):
    pair_total = teacher_errors = teacher_correct = 0
    corrected = correct_flips = student_flips = 0
    unit_total = teacher_top1_pos = student_top1_pos = top1_changed = 0
    residuals = []
    saturated = 0
    for row in rows:
        teacher = [float(x) for x in row["teacher_score"]]
        student = [float(x) for x in row["score"]]
        labels = [bool(x) for x in row["label"]]
        residuals.extend(float(x) for x in row["residual"])
        saturated += sum(abs(float(x)) >= 0.95 * bound for x in row["residual"])
        pos = [i for i, y in enumerate(labels) if y]
        neg = [i for i, y in enumerate(labels) if not y]
        if not pos or not neg:
            continue
        unit_total += 1
        ttop = max(range(len(teacher)), key=teacher.__getitem__)
        stop = max(range(len(student)), key=student.__getitem__)
        teacher_top1_pos += int(labels[ttop])
        student_top1_pos += int(labels[stop])
        top1_changed += int(ttop != stop)
        for i in pos:
            for j in neg:
                pair_total += 1
                t_ok = teacher[i] > teacher[j]
                s_ok = student[i] > student[j]
                teacher_correct += int(t_ok)
                teacher_errors += int(not t_ok)
                corrected += int((not t_ok) and s_ok)
                correct_flips += int(t_ok and not s_ok)
                student_flips += int(t_ok != s_ok)
    nres = len(residuals)
    return {
        "units_with_both_classes": unit_total,
        "pair_total": pair_total,
        "teacher_correct_pairs": teacher_correct,
        "teacher_error_pairs": teacher_errors,
        "teacher_error_corrected_pairs": corrected,
        "teacher_correct_flipped_pairs": correct_flips,
        "student_pair_flip_total": student_flips,
        "teacher_error_correction_rate": corrected / teacher_errors if teacher_errors else None,
        "teacher_correct_flip_rate": correct_flips / teacher_correct if teacher_correct else None,
        "student_pair_flip_rate": student_flips / pair_total if pair_total else None,
        "teacher_top1_positive_rate": teacher_top1_pos / unit_total if unit_total else None,
        "student_top1_positive_rate": student_top1_pos / unit_total if unit_total else None,
        "top1_changed_rate": top1_changed / unit_total if unit_total else None,
        "residual_count": nres,
        "residual_mean": sum(residuals) / nres if nres else None,
        "residual_abs_max": max((abs(x) for x in residuals), default=None),
        "residual_abs_mean": sum(abs(x) for x in residuals) / nres if nres else None,
        "residual_near_bound_fraction": saturated / nres if nres else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bound", type=float, default=0.5)
    args = ap.parse_args()
    groups = defaultdict(list)
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            groups[(row["split"], row["dataset"])].append(row)
    result = {
        "format": "locatemot-l51-b1-rank-flips-v1",
        "source": str(Path(args.input).resolve()),
        "bound": args.bound,
        "screening_or_test_read": False,
        "groups": {
            f"{split}/{dataset}": summarize(rows, args.bound)
            for (split, dataset), rows in sorted(groups.items())
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
