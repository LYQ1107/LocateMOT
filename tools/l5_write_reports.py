"""Stage L5: generate Route A reports from training/drift JSONs."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")


def load(p):
    if not Path(p).exists():
        return None
    with open(p) as f:
        return json.load(f)


def drift_table(paths):
    lines = ["| 集合/域 | common | drift | drift rate |",
             "|---|---:|---:|---:|"]
    for p, label in paths:
        d = load(p)
        if d is None:
            continue
        lines.append(f"| {label} | {d['common_candidates']} | "
                     f"{d['drift_candidates']} | {d['drift_rate']:.4f} |")
        for dom, v in d.get("per_domain", {}).items():
            lines.append(f"| {label} / {dom} | {v['common']} | {v['drift']} | "
                         f"{v['drift_rate']:.4f} |")
    return "\n".join(lines)


def learning_curve(marker):
    ck = ROOT / f"outputs/l5/checkpoints/{marker}"
    lc = load(ck / "learning_curve.json")
    if not lc:
        return "NOT_EXECUTED"
    rows = ["| epoch | loss | train_row_acc | val_row_acc |",
            "|---|---:|---:|---:|"]
    for r in lc:
        rows.append(f"| {r['epoch']} | {r['loss']:.4f} | "
                    f"{r['train_row_acc']:.4f} | "
                    f"{r['val_row_acc']:.4f} |")
    return "\n".join(rows)


def main():
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    small = ROOT / "outputs/l5/checkpoints/route_a_small"
    base = ROOT / "outputs/l5/checkpoints/route_a_base"
    lc_small = load(small / "learning_curve.json") or []
    lc_base = load(base / "learning_curve.json") or []
    best_small = max(lc_small, key=lambda r: r.get("val_row_acc", 0) or 0)
    best_base = max(lc_base, key=lambda r: r.get("val_row_acc", 0) or 0)
    drift = {
        "base_val": load(ROOT / "outputs/l5/drift_base_val.json"),
        "base_train": load(ROOT / "outputs/l5/drift_base_train.json"),
        "small_val": load(ROOT / "outputs/l5/drift_small_ep5_val.json"),
    }
    reports = {
        "l5_overfit_test.md": f"""# Stage L5 — Overfit / Memorization Capability Test

日期：2026-08-11。

## 协议

训练集：8 个 BDD 视频 + 3 个 DanceTrack calibration 视频（gt+u0 混合，
target 均 GT-anchored）；验证集：2 BDD + 1 Dance。每个模型 120 epoch，
batch 16，OneCycleLR（pct_start=0.05），seed 20260806。

判据（用户要求）：train drift 能否明显压到 <10–15%；若 train 能压但
val 不能 → generalization 问题；若 epoch20 后仍持续改善 → L4 的
20 epoch 太早。

## Small（1.48M）

{learning_curve('route_a_small')}

best val_row_acc = {best_small.get('val_row_acc') if best_small else 'N/A'}

## Base（7.71M）

{learning_curve('route_a_base')}

best val_row_acc = {best_base.get('val_row_acc') if best_base else 'N/A'}

## Train/Val drift（u0 source，离线 decode）

{drift_table([
    ('outputs/l5/drift_base_train.json', 'baseline train'),
    ('outputs/l5/drift_base_val.json', 'baseline val'),
    ('outputs/l5/drift_small_ep5_val.json', 'Small ep5 val'),
])}

## 结论

（训练结束后填写。）
""",
        "l5_learning_curve.md": f"""# Stage L5 — Learning Curve

日期：2026-08-11。

## 数据来源

`outputs/l5/checkpoints/route_a_small/learning_curve.json` 与
`route_a_base/learning_curve.json`（每 epoch 保存）。

## Small

{learning_curve('route_a_small')}

## Base

{learning_curve('route_a_base')}

## 观察

（训练结束后填写。）
""",
        "l5_capacity_scaling.md": f"""# Stage L5 — Capacity Scaling Ladder

日期：2026-08-11。

| 档位 | d_model | temporal layers | set layers | ffn | 参数量 |
|---|---:|---:|---:|---:|---:|
| Small | 128 | 2 | 2 | 512 | 1.48M |
| Base | 256 | 4 | 4 | 1024 | 7.71M |
| Large | 384 | 6 | 6 | 1536 | 计划 |

判据：Base 是否优于 Small；train 能否 fit；val 是否随容量提升。

（训练结束后填写结论。）
""",
        "l5_route_a.md": """# Stage L5 — Route A: GT-Anchored Temporal Identity Transformer

日期：2026-08-11。

## 假设

drift 主要来自缺少 persistent temporal identity state；GT-anchored
temporal state + set-level decoder + cross-spec relation-structure
consistency 能同时改善 identity consistency 与标准 tracking 指标。

## 架构

见 `docs/l5_temporal_identity_design.md`。

## 训练

- 数据：small_bdd_train + small_dance_train（gt+u0 混合，GT-anchored）；
- 损失：row/col ranking CE + cross-spec relation MSE（0.2）+ residual
  preservation；relation BCE 权重 0（单视图内同 GT 对不存在，无正样本）；
- 优化：AdamW 3e-4，OneCycle pct_start=0.05，clip 5.0，120 epochs。

## 结果

（训练结束后填写。）
""",
    }
    for name, content in reports.items():
        (out_dir / name).write_text(content)
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
