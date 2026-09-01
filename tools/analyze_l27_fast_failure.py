#!/usr/bin/env python3
"""Make a compact TrackEval failure decomposition from formal L27 output."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")


def main():
    run = ROOT / "outputs/l27/fast_rmot_validation_formal"
    summary = json.loads((run / "summary.json").read_text())
    rows = []
    for model, strategies in summary["trackeval"].items():
        for key, result in strategies.items():
            pred = run / model / key / "uidm18"
            prediction_count = 0; gt_count = 0; empty_files = 0; gt_files = 0
            for p in pred.glob("*/*/predict.txt"):
                gt = p.with_name("gt.txt")
                prediction_count += len([x for x in p.read_text().splitlines() if x.strip()])
                if not p.read_text().strip(): empty_files += 1
                if gt.exists(): gt_count += len([x for x in gt.read_text().splitlines() if x.strip()]); gt_files += 1
            m = result.get("metrics", {})
            if m.get("DetPr___AUC", 0) < 20 and m.get("DetRe___AUC", 0) >= 45: failure = "precision: DetPr<20 while DetRe>=45"
            elif m.get("DetRe___AUC", 0) < 45 and m.get("DetPr___AUC", 0) >= 20: failure = "recall: DetRe<45 while DetPr>=20"
            elif m.get("AssA___AUC", 0) < 35: failure = "association: AssA low"
            else: failure = "mixed HOTA/DetA/precision-recall"
            rows.append({"model":model,"strategy":key,"trackeval":m,"prediction_count":prediction_count,"gt_count":gt_count,"predictions_per_gt":prediction_count/max(1,gt_count),"empty_prediction_files":empty_files,"gt_files":gt_files,"failure_class":failure,"threshold_metrics":summary["candidate_metrics"][model][key]["threshold_metrics"]})
    rows.sort(key=lambda x:(x["trackeval"].get("HOTA___AUC",0),x["trackeval"].get("DetPr___AUC",0)),reverse=True)
    report={"format":"locatemot-l27-fast-failure-decomposition-v1","formal_summary":str(run/"summary.json"),"screening_gt_used_for_selection":False,"rows":rows,"best_by_model":{m:next(x for x in rows if x["model"]==m) for m in summary["trackeval"]},"interpretation":"Best HOTA branches have adequate DetRe and non-collapsed AssA but DetPr below 20; NULL max gating trades recall for lower false acceptance and does not repair HOTA."}
    (run/"failure_decomposition.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# L27 fast RMOT failure decomposition","", "All thresholds were calibrated on the 64-query calibration split; screening labels are not used for selection.", "", "| model | strategy | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | predictions/GT | empty files | failure |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in rows:
        m=x["trackeval"]; lines.append(f"| {x['model']} | {x['strategy']} | {m.get('HOTA___AUC',0):.4f} | {m.get('DetA___AUC',0):.4f} | {m.get('AssA___AUC',0):.4f} | {m.get('DetRe___AUC',0):.4f} | {m.get('DetPr___AUC',0):.4f} | {m.get('IDF1',0):.4f} | {x['predictions_per_gt']:.4f} | {x['empty_prediction_files']} | {x['failure_class']} |")
    lines += ["", "The highest-HOTA strategy for both models is precision-first threshold. It misses the DetPr>=20 gate while DetRe is above 45 and AssA is not collapsed. NULL gating sharply increases empty outputs and lowers recall, but still does not reach Level 1.", ""]
    (run/"failure_decomposition.md").write_text("\n".join(lines))
    print(json.dumps({"out":str(run),"rows":len(rows),"best":report["best_by_model"]},indent=2))


if __name__ == "__main__": main()
