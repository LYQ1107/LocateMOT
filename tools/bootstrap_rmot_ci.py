"""CPU bootstrap CI over the 40 Refer-Dance RMOT queries.

Parses per-query HOTA/DetA/AssA from an RMOT TrackEval log and reports
mean + 95% bootstrap percentile CI.

Usage: python tools/bootstrap_rmot_ci.py outputs/l9/trackeval/l9_main_v5/rmot/trackeval.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()
    lines = Path(args.log).read_text().splitlines()
    hota, deta, assa = [], [], []
    for l in lines:
        if not l.strip().startswith("dancetrack"):
            continue
        nums = re.findall(r"-?\d+\.\d+", l)
        if len(nums) >= 12:  # HOTA rows have 12 decimal fields
            try:
                hota.append(float(nums[0]))
                deta.append(float(nums[1]))
                assa.append(float(nums[2]))
            except ValueError:
                pass
    hota, deta, assa = np.asarray(hota), np.asarray(deta), np.asarray(assa)
    print(f"queries={len(hota)}")
    rng = np.random.default_rng(args.seed)

    def ci(x):
        boot = np.array([np.mean(rng.choice(x, size=len(x), replace=True))
                         for _ in range(args.n_boot)])
        return float(np.mean(x)), float(np.percentile(boot, 2.5)), \
            float(np.percentile(boot, 97.5))

    for name, x in [("HOTA", hota), ("DetA", deta), ("AssA", assa)]:
        m, lo, hi = ci(x)
        print(f"{name}: mean={m:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
