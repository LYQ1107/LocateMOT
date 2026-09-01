"""Stage L1-C: learnability probe (diagnostic only).

Fits a logistic regression on calibration cue events to predict:
  1) PBD candidate-selection wrong
  2) IoU candidate-selection wrong
  3) raw-PBD full-video ID continuity wrong
using prediction-side features only. Evaluates AUROC / PR-AUC on held-out
DanceTrack val events.
"""
from __future__ import annotations

import csv
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
FEATURES = ["iou_top1", "iou_margin", "pbd_top1", "pbd_margin",
            "num_candidates", "obj_size", "gap"]


def load_events(path, method):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["method"] != method:
                continue
            rows.append({k: float(r[k]) for k in FEATURES} | {
                "pbd_wrong": int(r["pbd_correct"] == "0"),
                "iou_wrong": int(r["iou_correct"] == "0"),
                "method_wrong": int(r["correct"] == "0"),
            })
    return rows


def run_probe(train, test, target_key, target_name):
    X_tr = np.asarray([[r[k] for k in FEATURES] for r in train], dtype=np.float64)
    y_tr = np.asarray([r[target_key] for r in train], dtype=np.int32)
    X_te = np.asarray([[r[k] for k in FEATURES] for r in test], dtype=np.float64)
    y_te = np.asarray([r[target_key] for r in test], dtype=np.int32)
    if y_tr.sum() == 0 or (y_tr == 0).sum() == 0:
        return {"target": target_name, "skipped": True}
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    return {
        "target": target_name,
        "n_train": len(train), "n_test": len(test),
        "pos_rate_train": round(float(y_tr.mean()), 4),
        "pos_rate_test": round(float(y_te.mean()), 4),
        "auc": round(float(roc_auc_score(y_te, p)), 4),
        "pr_auc": round(float(average_precision_score(y_te, p)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-method", default="C2_t0.0")
    ap.add_argument("--test-method", default="C2")
    args = ap.parse_args()
    train = load_events(os.path.join(ROOT, "outputs/l1_c/calibration_cue/cue_events.csv"),
                        args.train_method)
    test = load_events(os.path.join(ROOT, "outputs/l1_c/cue_events.csv"),
                       args.test_method)
    # cap test size for speed (random subsample by video is not needed; just cap)
    rng = np.random.RandomState(20260806)
    if len(test) > 200000:
        idx = rng.choice(len(test), 200000, replace=False)
        test = [test[i] for i in idx]
    results = [
        run_probe(train, test, "pbd_wrong", "pbd_selection_wrong"),
        run_probe(train, test, "iou_wrong", "iou_selection_wrong"),
        run_probe(train, test, "method_wrong", "raw_pbd_id_continuity_wrong"),
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    with open(os.path.join(ROOT, "outputs/l1_c/learnability_probe.json"),
              "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
