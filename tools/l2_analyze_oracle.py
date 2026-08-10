"""Stage L2: analyze counterfactual oracle events.

Computes oracle headroom, local-vs-future mismatch categories, action
ranking agreement across horizons, and regret statistics from the events
saved by tools/run_l2_oracle.py.

Usage:
  python tools/l2_analyze_oracle.py \
      --events outputs/l2/oracle/events_dancetrack_val.pkl \
      --out outputs/l2/oracle/analysis_dancetrack_val.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np


def local_correct_count(ev, act):
    cand_gt = [c["gt"] for c in ev["cands"]]
    snaps = ev["track_snaps"]
    n = 0
    for ti, ci in act:
        if ci >= len(cand_gt) or ti >= len(snaps):
            continue
        if cand_gt[ci] is not None and snaps[ti]["true_gt"] == cand_gt[ci]:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.events, "rb") as f:
        events = pickle.load(f)

    horizons = sorted({h for ev in events for h in ev["horizons"]})
    out = {
        "n_events": len(events),
        "horizons": horizons,
        "headroom": {},
        "mismatch": {},
        "ranking": {},
    }
    for H in horizons:
        base_u, best_u = [], []
        base_idsw, best_idsw = [], []
        regret = []
        cat = defaultdict(int)
        n_diff = 0
        for ev in events:
            acts = ev["actions"]
            n_a = len(acts)
            key = f"H{H}"
            vals = []
            for i in range(n_a):
                w = ev["action_utils"][str(i)]["utils"].get(key)
                vals.append(w["assa"] if w else 0.0)
            if not vals or ev["action_utils"]["0"]["utils"].get(key) is None:
                continue
            bu = vals[0]
            bi = max(vals)
            best_idx = int(np.argmax(vals))
            base_u.append(bu)
            best_u.append(bi)
            base_idsw.append(ev["action_utils"]["0"]["utils"][key]["idsw"])
            best_idsw.append(ev["action_utils"][str(best_idx)]["utils"][key]["idsw"])
            regret.append(bi - bu)
            if bi > bu + 1e-6:
                n_diff += 1
            base_lc = local_correct_count(ev, acts[0])
            best_lc = local_correct_count(ev, acts[best_idx])
            max_lc = max(local_correct_count(ev, a) for a in acts)
            if best_idx != 0:
                if base_lc > best_lc and bi > bu + 1e-6:
                    cat["local_correct_future_bad"] += 1
                if best_lc > base_lc and bi > bu + 1e-6:
                    cat["local_wrong_future_good"] += 1
                if base_lc == max_lc and base_lc > 0 and bi > bu + 1e-6:
                    cat["base_correct_future_suboptimal"] += 1
                if bi > bu + 1e-6:
                    cat["future_best_differs"] += 1
        n = max(1, len(base_u))
        out["headroom"][f"H{H}"] = {
            "n": len(base_u),
            "mean_base_assa": float(np.mean(base_u)),
            "mean_best_assa": float(np.mean(best_u)),
            "mean_gain_assa": float(np.mean(np.asarray(best_u) - np.asarray(base_u))),
            "mean_base_idsw": float(np.mean(base_idsw)),
            "mean_best_idsw": float(np.mean(best_idsw)),
            "frac_better": n_diff / n,
            "mean_regret": float(np.mean(regret)),
        }
        out["mismatch"][f"H{H}"] = {
            "n": len(base_u),
            "future_best_differs": cat["future_best_differs"],
            "local_correct_future_bad": cat["local_correct_future_bad"],
            "local_wrong_future_good": cat["local_wrong_future_good"],
            "base_correct_future_suboptimal": cat["base_correct_future_suboptimal"],
        }
        # ranking agreement: Kendall-like pairwise agreement of action order
        # between H and the longest horizon
        ref = f"H{max(horizons)}"
        if f"H{H}" == ref:
            continue
        agree = total = 0
        for ev in events:
            acts = ev["actions"]
            uH = [ev["action_utils"][str(i)]["utils"].get(f"H{H}", {}).get("assa", 0.0)
                  for i in range(len(acts))]
            uR = [ev["action_utils"][str(i)]["utils"].get(ref, {}).get("assa", 0.0)
                  for i in range(len(acts))]
            for i in range(len(acts)):
                for j in range(i + 1, len(acts)):
                    if abs(uH[i] - uH[j]) < 1e-9 and abs(uR[i] - uR[j]) < 1e-9:
                        continue
                    total += 1
                    if (uH[i] - uH[j]) * (uR[i] - uR[j]) > 0:
                        agree += 1
        out["ranking"][f"{H}_vs_{ref}"] = {
            "pair_agree": agree,
            "pair_total": total,
            "pair_agreement": agree / total if total else None,
        }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
