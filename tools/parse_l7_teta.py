"""Stage L7: extract official TETA Base/Novel/All numbers from a log file
into JSON for the final report.

Usage:
  python tools/parse_l7_teta.py --log outputs/l7/logs/eval_ovmot_probe.log \
      --out outputs/l7/trackeval/ovmot_probe/teta.json
"""
from __future__ import annotations

import argparse
import json
import re


def parse(path):
    text = open(path, errors="ignore").read()
    rows = {}
    # lines like: 'Base       22.602     60.058     7.750   ...'
    for m in re.finditer(
            r"^(Base|Novel|COMBINED)\s+([\d.\-nan ]+?)\s*$", text,
            re.MULTILINE):
        name = m.group(1)
        vals = [x for x in m.group(2).split() if x]
        if len(vals) >= 4:
            rows[name] = {
                "TETA": vals[0], "LocA": vals[1], "AssocA": vals[2],
                "ClsA": vals[3],
            }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = parse(args.log)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
