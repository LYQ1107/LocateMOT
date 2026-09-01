"""Summarize paired restriction audits (U0 vs L4 model) into a table."""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audits", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    audits = [json.load(open(p)) for p in args.audits]
    specs = sorted({s for a in audits for s in a["summary"]})
    lines = [
        "| Spec | Pairs | "
        + " | ".join(f"{l} drift | {l} P0 AssA | {l} P1 AssA | "
                     f"{l} P0 IDSW | {l} P1 IDSW" for l in args.labels)
        + " |",
        "|---|--:|" + "|".join(["--:|"] * (len(args.labels) * 5)) + "|",
    ]
    for spec in specs:
        row = [spec, str(audits[0]["summary"].get(spec, {}).get("n_pairs", 0))]
        for a in audits:
            s = a["summary"].get(spec, {})
            row += [f"{s.get('drift_rate', float('nan')):.4f}",
                    f"{s['p0_mean']['assa']:.4f}",
                    f"{s['p1_mean']['assa']:.4f}",
                    str(s["p0_mean"]["idsw_sum"]),
                    str(s["p1_mean"]["idsw_sum"])]
        lines.append("| " + " | ".join(row) + " |")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
