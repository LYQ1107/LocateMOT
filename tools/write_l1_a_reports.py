#!/usr/bin/env python
"""Stage L1-A: generate final report, GPT handoff, and stage decision."""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    return json.load(open(path))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v, nd=2):
    try:
        x = float(v)
        if x != x:
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    out = os.path.join(ROOT, "outputs", "l1_a")
    reports = os.path.join(ROOT, "reports")
    os.makedirs(reports, exist_ok=True)
    main_rows = {r["variant"]: r for r in load_csv(os.path.join(out, "main_results_dla.csv"))}
    ctrl_rows = {r["variant"]: r for r in load_csv(os.path.join(out, "main_results_ctrl.csv"))}
    subset_rows = {r["variant"]: r for r in load_csv(os.path.join(out, "subset_results_val.csv"))}
    manifest = load_json(os.path.join(out, "detection_manifest.json"), [])
    recall_rows = load_csv(os.path.join(out, "candidate_recall.csv"))

    def g(row, key):
        return row.get(key, "") if row else ""

    def metric(variant, key, rows=None):
        rows = rows if rows is not None else main_rows
        return fmt(g(rows.get(variant, {}), key))

    # decision
    t0 = main_rows.get("T0", {})
    t6 = main_rows.get("T6", {})
    assa_delta = num(g(t6, "HOTA_AssA(0)")) - num(g(t0, "HOTA_AssA(0)"))
    idsw_delta_pct = (
        (num(g(t6, "CLEAR_IDSW")) - num(g(t0, "CLEAR_IDSW"))) / max(1e-9, num(g(t0, "CLEAR_IDSW"))) * 100
    )
    hota_delta = num(g(t6, "HOTA_HOTA(0)")) - num(g(t0, "HOTA_HOTA(0)"))
    idf1_delta = num(g(t6, "Identity_IDF1")) - num(g(t0, "Identity_IDF1"))
    t1 = main_rows.get("T1", {})
    t2 = main_rows.get("T2", {})
    cond1 = assa_delta >= 3.0 and idsw_delta_pct <= -20.0 and hota_delta >= 2.0
    cond2 = assa_delta >= 4.0 and idf1_delta >= 2.0 and idsw_delta_pct <= -25.0
    beats_t1 = (
        num(g(t6, "HOTA_AssA(0)")) > num(g(t1, "HOTA_AssA(0)"))
        and num(g(t6, "CLEAR_IDSW")) < num(g(t1, "CLEAR_IDSW"))
    )
    beats_t2 = sum([
        num(g(t6, "HOTA_AssA(0)")) > num(g(t2, "HOTA_AssA(0)")),
        num(g(t6, "Identity_IDF1")) > num(g(t2, "Identity_IDF1")),
        num(g(t6, "CLEAR_IDSW")) < num(g(t2, "CLEAR_IDSW")),
    ]) >= 2
    if cond1 or cond2:
        decision = "L1_A_PASS"
    elif beats_t1 and beats_t2:
        decision = "L1_A_PARTIAL"
    else:
        decision = "L1_A_FAIL_TEMPORAL_VALUE_NOT_PROVEN"

    recall = recall_rows[0] if recall_rows else {}
    status = {
        "stage": "L1-A",
        "decision": decision,
        "assa_delta_pp": round(assa_delta, 2),
        "idsw_delta_pct": round(idsw_delta_pct, 1),
        "hota_delta_pp": round(hota_delta, 2),
        "idf1_delta_pp": round(idf1_delta, 2),
        "t6_vs_t1": bool(beats_t1),
        "t6_vs_t2_majority": bool(beats_t2),
        "recall_0.5": recall.get("recall_0.5"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, "final_status.json"), "w") as f:
        json.dump(status, f, indent=2)
    st = load_json(os.path.join(out, "state.json"), {})
    st["stage"] = "L1-A"
    st["state"] = "L1_A_COMPLETE" if decision == "L1_A_PASS" else "L1_A_REPORTED"
    st.setdefault("history", []).append({"state": st["state"], "decision": decision,
                                         "ts": datetime.now().isoformat(timespec="seconds")})
    with open(os.path.join(out, "state.json"), "w") as f:
        json.dump(st, f, indent=2)

    # ---- final report ----
    lines = []
    lines.append("# Stage L1-A Final Report\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Stage decision: **{decision}**\n")
    lines.append("## 1. Executive Summary\n")
    lines.append(f"在固定 detections 下比较 T0 IoU → T6 trajectory-aware association。"
                f"T6 vs T0：AssA {fmt(assa_delta, 2)} pp，IDSW {fmt(idsw_delta_pct, 1)}%，"
                f"HOTA {fmt(hota_delta, 2)} pp。LocateAnything Recall@0.5 = {recall.get('recall_0.5', 'n/a')}。\n")
    lines.append("## 2. Why Two-Frame B6 Was Not Enough\n")
    lines.append("L0-D held-out：B6 conditional +3.5pp、hard +5.7pp、IDSW -19.2%，"
                "但 HOTA/AssA 基本持平且 5-8 targets 仍低于 IoU。因此本阶段把关联从两帧升级为全视频轨迹感知。\n")
    lines.append("## 3. Scientific Question\n")
    lines.append("在固定 detections 下，trajectory history + motion prediction + short-term/anchor memory "
                "+ lost/reactivation 是否能在真实连续视频中显著减少 IDSW 并提升 HOTA/AssA/IDF1。\n")
    lines.append("## 4. Frozen L0-D Basis\n")
    lines.append("LocateAnything-3B (commit 783f656d) 与 B6 (outputs/l0_d/checkpoints/b6/best.pt) 全程冻结；"
                "B6 作为 local association kernel。\n")
    lines.append("## 5. 2025-2026 Literature and GitHub Audit\n")
    lines.append("完整审计见 docs/l1_a_reference_audit.md；主要参考 FDTA (CVPR 2026, MIT, b3b3b778)、"
                "MOTIP (CVPR 2025, MIT, ffc0e905)、MeMOTR (ICCV 2023)、OC-SORT (MIT, 8462e7e7)、MOTR；"
                "MATR (arXiv:2509.21715) 记录为 NO VERIFIED OFFICIAL CODE FOUND。\n")
    lines.append("## 6. DanceTrack Protocol\n")
    lines.append("train 40 / val 25 / test 35；本阶段固定 32 train + 8 calibration，official val 25 全程 held-out。\n")
    lines.append("## 7. Dataset Split\n")
    lines.append("seed 20260806，video-level disjoint；calibration 按 GT density 低/中/高 2/3/3 选取。\n")
    lines.append("## 8. Detection Protocols\n")
    lines.append("D-LA：LocateAnything-3B person query（calibration 固定 'person.'）；"
                "D-CTRL：ByteTrack 官方 YOLOX-X DanceTrack 权重 + OC-SORT 官方推理。\n")
    lines.append("## 9. LocateAnything Detection Quality\n")
    if recall_rows:
        r = recall_rows[0]
        lines.append(f"| query | Recall@0.3 | Recall@0.5 | Recall@0.7 | Precision | cand/frame | FPS | peak VRAM |\n")
        lines.append(f"|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in recall_rows:
            lines.append(f"| {r['query_id']} | {r['recall_0.3']} | {r['recall_0.5']} | {r['recall_0.7']} "
                         f"| {r['precision']} | {r['candidates_per_frame']} | {r['fps']} | {r['peak_gpu_gb']} |\n")
    lines.append("\n## 10. Shared Birth/Lifecycle Infrastructure\n")
    lines.append("Birth is shared evaluation infrastructure, not a proposed component. "
                "unmatched det -> tentative -> min_hits=3 -> ACTIVE；max_age=30。所有 T0-T6 相同。\n")
    for i, name in enumerate(["T0 IoU", "T1 Motion Baseline", "T2 B6 Local", "T3 Trajectory Context",
                              "T4 Motion-Aware Update", "T5 Memory", "T6 Lost/Reactivation"]):
        lines.append(f"## {11+i}. {name}\n")
        v = f"T{i}"
        lines.append(f"{name}：HOTA {metric(v, 'HOTA_HOTA(0)')}，DetA {metric(v, 'HOTA_DetA(0)')}，"
                     f"AssA {metric(v, 'HOTA_AssA(0)')}，LocA {metric(v, 'HOTA_LocA(0)')}，"
                     f"MOTA {metric(v, 'CLEAR_MOTA')}，MOTP {metric(v, 'CLEAR_MOTP')}，"
                     f"IDF1 {metric(v, 'Identity_IDF1')}，IDSW {metric(v, 'CLEAR_IDSW')}。\n")
    lines.append("## 18. Architecture Summary\n")
    lines.append("T3: TrajectoryEncoder(2-layer causal temporal transformer, K=8, raw-space fusion) -> frozen B6。\n"
                 "T4: + MotionPredictor(2-layer MLP, dx/dy/dw/dh, SmoothL1) + bounded motion residual。\n"
                 "T5: + MemoryFusion(anchor + EMA, 高可信写入)。\n"
                 "T6: + ReactivationResidualHead(lost>=2, trajectory/PBD similarity + motion-weighted IoU)。\n")
    lines.append("## 19. Training Setup\n")
    lines.append("仅训练 TrajectoryEncoder/MotionPredictor/MemoryFusion/residual heads + nm_bias；"
                "B6 冻结；AdamW lr=2e-4，bf16，SmoothL1 motion loss lambda=0.1。\n")
    lines.append("## 20. Main Full-Video Results (D-LA val)\n")
    lines.append("| variant | HOTA | DetA | AssA | LocA | MOTA | MOTP | IDF1 | IDP | IDR | IDSW | FP | FN | Frag | MT | PT | ML |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for v in ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]:
        r = main_rows.get(v, {})
        lines.append(f"| {v} | {fmt(g(r,'HOTA_HOTA(0)'))} | {fmt(g(r,'HOTA_DetA(0)'))} | {fmt(g(r,'HOTA_AssA(0)'))} "
                     f"| {fmt(g(r,'HOTA_LocA(0)'))} | {fmt(g(r,'CLEAR_MOTA'))} | {fmt(g(r,'CLEAR_MOTP'))} "
                     f"| {fmt(g(r,'Identity_IDF1'))} | {fmt(g(r,'Identity_IDP'))} | {fmt(g(r,'Identity_IDR'))} "
                     f"| {fmt(g(r,'CLEAR_IDSW'),0)} | {fmt(g(r,'CLEAR_FP'),0)} | {fmt(g(r,'CLEAR_FN'),0)} "
                     f"| {fmt(g(r,'CLEAR_Frag'),1)} | {fmt(g(r,'CLEAR_MT'),0)} | {fmt(g(r,'CLEAR_PT'),0)} "
                     f"| {fmt(g(r,'CLEAR_ML'),0)} |\n")
    lines.append("## 21. Incremental Ablation\n")
    lines.append(f"T6 vs T0: AssA {fmt(assa_delta,2)}pp, IDSW {fmt(idsw_delta_pct,1)}%, HOTA {fmt(hota_delta,2)}pp；"
                 f"T6 vs T1: AssA {fmt(num(g(t6,'HOTA_AssA(0)'))-num(g(t1,'HOTA_AssA(0)')),2)}pp, "
                 f"IDSW {fmt(num(g(t6,'CLEAR_IDSW'))-num(g(t1,'CLEAR_IDSW')),0)}；"
                 f"T6 vs T2 至少两项更优: {beats_t2}。\n")
    lines.append("## 22. Low-IoU Results\n")
    sr = subset_rows.get("dla|T6", {})
    sr0 = subset_rows.get("dla|T0", {})
    lines.append(f"T0 iou_<0.1 acc={sr0.get('iou_<0.1_acc','n/a')}，0.1-0.3 acc={sr0.get('iou_0.1-0.3_acc','n/a')}；"
                 f"T6 iou_<0.1 acc={sr.get('iou_<0.1_acc','n/a')}，0.1-0.3 acc={sr.get('iou_0.1-0.3_acc','n/a')}。\n")
    lines.append("## 23. Crowd/Density Results\n")
    lines.append(f"T0 density low/med/high acc={sr0.get('density_low_acc','n/a')}/"
                 f"{sr0.get('density_medium_acc','n/a')}/{sr0.get('density_high_acc','n/a')}；"
                 f"T6 low/med/high acc={sr.get('density_low_acc','n/a')}/"
                 f"{sr.get('density_medium_acc','n/a')}/{sr.get('density_high_acc','n/a')}。\n")
    lines.append("## 24. Ambiguous Association Results\n")
    lines.append(f"T0 ambiguous acc={sr0.get('ambiguous_acc','n/a')} (n={sr0.get('ambiguous_n','n/a')})；"
                 f"T6 ambiguous acc={sr.get('ambiguous_acc','n/a')} (n={sr.get('ambiguous_n','n/a')})。\n")
    lines.append("## 25. Reactivation Results\n")
    lines.append(f"T1 events={sr0 and 'n/a'}；T6 events={sr.get('reactivation_events','n/a')}，"
                 f"id_kept={sr.get('reactivation_accuracy','n/a')}，mean_gap={sr.get('reactivation_mean_gap','n/a')}。\n")
    lines.append("## 26. Sequence-wise Results\n")
    lines.append("见 outputs/l1_a/per_sequence_results_dla.csv。\n")
    lines.append("## 27. LocateAnything vs Controlled Detection\n")
    lines.append("D-CTRL 固定 YOLOX-X 只运行 T0/T1（T2-T6 需要 ObjectToken 特征）；结果见 main_results_ctrl.csv。\n")
    lines.append("## 28. Why Not IoU?\n")
    lines.append("结论由 low-IoU 子集与整体 IDSW 决定：详见第 22 节与第 20 节真实数值，不以口头解释代替实验。\n")
    lines.append("## 29. Failure Cases\n")
    lines.append("见 reports/l1_a_failure_analysis.md（若生成）。\n")
    lines.append("## 30. Resource Usage\n")
    lines.append("见 outputs/l1_a/tracker_runtime_*.json 与 cache meta（peak VRAM ~10.7GB/进程）。\n")
    lines.append("## 31. Scientific Interpretation\n")
    lines.append("以固定 detections 下 association-only 差异解释；不声称 detection 能力。\n")
    lines.append("## 32. Claim Boundary\n")
    lines.append("可以说：full-video trajectory context/motion/memory/reactivation 对 DanceTrack 固定检测的关联影响；"
                "不能说：LocateAnything 检测性能、open-vocabulary、跨数据集泛化。\n")
    lines.append(f"## 33. Stage Decision\n{decision}\n")
    lines.append("## 34. Next Recommended Stage\n")
    lines.append("依据 decision 决定：PASS -> Visual Prompt LoRA / candidate generation；"
                "否则 -> 先诊断 ObjectToken 判别力与 trajectory/memory 污染。\n")
    lines.append("## 35. Important Paths\n")
    lines.append("configs/stage_l1_a.yaml；outputs/l1_a/{main_results_dla.csv, per_sequence_results_dla.csv, "
                "detection_manifest.json, final_status.json}；reports/STAGE_L1_A_GPT_HANDOFF.md\n")
    with open(os.path.join(reports, "STAGE_L1_A_FINAL_REPORT.md"), "w") as f:
        f.write("\n".join(lines))

    # ---- GPT handoff ----
    h = []
    h.append("# Stage L1-A GPT 交接报告\n")
    h.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | Stage decision: {decision}\n")
    h.append("## 1. Stage L0-D 结论\n")
    h.append("B6 两帧关联：conditional 0.7783 vs IoU 0.7432（+3.5pp），hard +5.7pp，IDSW -19.2%；"
            "但 HOTA 0.6592 vs 0.6607、AssA 0.8127 vs 0.8128 基本持平，5-8 targets 仍低于 IoU。\n")
    h.append("## 2. 为什么转 full-video\n")
    h.append("两帧模型只使用上一帧 token，无法利用 trajectory；需要全视频实验回答“为什么不用 IoU”。\n")
    h.append("## 3. GitHub 2025/2026 审计\n")
    h.append("FDTA (CVPR 2026, MIT, b3b3b778)、MOTIP (CVPR 2025, ffc0e905)、MeMOTR、OC-SORT、MOTR 已读；"
            "MATR 无官方代码。\n")
    h.append("## 4. DanceTrack 规模\n")
    h.append("train 32 (33,772 帧) / calibration 8 (8,024 帧) / official val 25 (25,508 帧)，video-level disjoint。\n")
    h.append(f"## 5. Detection recall\nD-LA person query Recall@0.5 = {recall.get('recall_0.5','n/a')}\n")
    h.append("## 6. T0-T6 架构\n")
    h.append("T0 IoU；T1 OC-SORT 风格 Kalman+OCM；T2 冻结 B6 local；T3 TrajectoryEncoder(K=8)；"
            "T4 + MotionPredictor；T5 + anchor/EMA memory；T6 + lost/reactivation。\n")
    h.append("## 7. HOTA/DetA/AssA/MOTA/IDF1/IDSW 完整表\n")
    h.append("| variant | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for v in ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]:
        r = main_rows.get(v, {})
        h.append(f"| {v} | {fmt(g(r,'HOTA_HOTA(0)'))} | {fmt(g(r,'HOTA_DetA(0)'))} | {fmt(g(r,'HOTA_AssA(0)'))} "
                 f"| {fmt(g(r,'CLEAR_MOTA'))} | {fmt(g(r,'Identity_IDF1'))} | {fmt(g(r,'CLEAR_IDSW'),0)} |\n")
    h.append("## 8. Low-IoU 结果\n")
    h.append(f"T0 vs T6：<0.1 acc {sr0.get('iou_<0.1_acc','n/a')} vs {sr.get('iou_<0.1_acc','n/a')}；"
             f"0.1-0.3 {sr0.get('iou_0.1-0.3_acc','n/a')} vs {sr.get('iou_0.1-0.3_acc','n/a')}。\n")
    h.append("## 9. High-density 结果\n")
    h.append(f"T0 vs T6 high density acc：{sr0.get('density_high_acc','n/a')} vs {sr.get('density_high_acc','n/a')}。\n")
    h.append("## 10. Reactivation\n")
    h.append(f"T6 events={sr.get('reactivation_events','n/a')}，id kept={sr.get('reactivation_accuracy','n/a')}。\n")
    h.append("## 11. T6 vs IoU\n")
    h.append(f"AssA {fmt(assa_delta,2)}pp，IDF1 {fmt(idf1_delta,2)}pp，IDSW {fmt(idsw_delta_pct,1)}%，HOTA {fmt(hota_delta,2)}pp。\n")
    h.append("## 12. T6 vs Kalman/OC-SORT\n")
    h.append(f"T6 vs T1：AssA {fmt(num(g(t6,'HOTA_AssA(0)'))-num(g(t1,'HOTA_AssA(0)')),2)}pp，"
             f"IDSW {fmt(num(g(t6,'CLEAR_IDSW'))-num(g(t1,'CLEAR_IDSW')),0)}。\n")
    h.append("## 13. T6 vs B6-local\n")
    h.append(f"T6 vs T2：AssA {fmt(num(g(t6,'HOTA_AssA(0)'))-num(g(t2,'HOTA_AssA(0)')),2)}pp，"
             f"IDF1 {fmt(num(g(t6,'Identity_IDF1'))-num(g(t2,'Identity_IDF1')),2)}pp，"
             f"IDSW {fmt(num(g(t6,'CLEAR_IDSW'))-num(g(t2,'CLEAR_IDSW')),0)}；"
             f"至少两项更优：{beats_t2}。\n")
    h.append(f"## 14. 是否真正证明“为什么不用 IoU”\n{decision}\n")
    h.append("## 15. 剩余最大瓶颈\n")
    h.append("根据 low-IoU / reactivation / ambiguous 数据定位（见 l1_a_failure_analysis.md）。\n")
    h.append("## 16. 下一阶段唯一建议\n")
    h.append("若 PASS：Visual Prompt LoRA；否则：先修复 ObjectToken 判别力与 trajectory/memory 训练-推理一致性。\n")
    with open(os.path.join(reports, "STAGE_L1_A_GPT_HANDOFF.md"), "w") as f:
        f.write("\n".join(h))
    with open(os.path.join(reports, "LATEST_GPT_HANDOFF.md"), "w") as f:
        f.write("最新交接报告：reports/STAGE_L1_A_GPT_HANDOFF.md\n")

    # auxiliary reports
    with open(os.path.join(reports, "l1_a_dancetrack_candidate_report.md"), "w") as f:
        f.write("# Stage L1-A DanceTrack Candidate Report\n\n")
        f.write("| query | Recall@0.5 | Recall@0.7 | cand/frame | fps |\n|---|---:|---:|---:|---:|\n")
        for r in recall_rows:
            f.write(f"| {r['query_id']} | {r['recall_0.5']} | {r['recall_0.7']} "
                    f"| {r['candidates_per_frame']} | {r['fps']} |\n")
    with open(os.path.join(reports, "l1_a_training_report.md"), "w") as f:
        f.write("# Stage L1-A Training Report\n\n仅训练 temporal modules；B6 冻结；checkpoint: outputs/l1_a/checkpoints/temporal/best.pt\n")
    with open(os.path.join(reports, "l1_a_full_video_evaluation.md"), "w") as f:
        f.write("# Stage L1-A Full-Video Evaluation\n\n见 outputs/l1_a/main_results_dla.csv 与 per_sequence_results_dla.csv\n")
    with open(os.path.join(reports, "l1_a_low_iou_analysis.md"), "w") as f:
        f.write("# Stage L1-A Low-IoU Analysis\n\n见 outputs/l1_a/subset_results_val.csv\n")
    with open(os.path.join(reports, "l1_a_reactivation_analysis.md"), "w") as f:
        f.write("# Stage L1-A Reactivation Analysis\n\n见 outputs/l1_a/reactivation_results_val.csv\n")
    with open(os.path.join(reports, "l1_a_failure_analysis.md"), "w") as f:
        f.write("# Stage L1-A Failure Analysis\n\n待数值定位；主瓶颈见 final_status.json。\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
