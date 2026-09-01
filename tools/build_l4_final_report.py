"""Append all Stage L4 sub-reports verbatim into the final report."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "reports/STAGE_L4_FINAL_REPORT.md")

APPENDICES = [
    ("A", "2025–2026 官方代码审计", "docs/l4_reference_audit.md"),
    ("B", "Novelty Collision Audit", "reports/l4_novelty_collision_audit.md"),
    ("C", "Specification / Restriction 任务定义", "docs/l4_specification_task_definition.md"),
    ("D", "Specification Paired-View Dataset", "reports/l4_spec_view_dataset.md"),
    ("E", "TAO Cache Recovery Plan", "docs/l4_tao_cache_recovery_plan.md"),
    ("F", "Cross-Spec Consistency Metric Audit", "docs/l4_consistency_metric_audit.md"),
    ("G", "U0 Restriction Audit (P0 vs P1)", "reports/l4_u0_restriction_audit.md"),
    ("H", "Cross-Spec Identity Inconsistency", "reports/l4_cross_spec_inconsistency.md"),
    ("I", "BDD100K Multi-Spec Results", "reports/l4_bdd_multispec.md"),
    ("J", "TAO / Open-World Results", "reports/l4_tao_openworld.md"),
    ("K", "Implementation Evidence", "docs/l4_implementation_evidence.md"),
    ("L", "Pilot Results", "reports/l4_pilot.md"),
    ("M", "Full Multi-Domain Training", "reports/l4_full_training.md"),
    ("N", "Association-Controlled Results", "reports/l4_ac_results.md"),
    ("O", "Efficiency", "reports/l4_efficiency.md"),
    ("P", "LODO", "reports/l4_lodo.md"),
    ("Q", "Leave-One-Spec-Type-Out", "reports/l4_loso_spec.md"),
    ("R", "Ablations", "reports/l4_ablation.md"),
    ("S", "Failure Analysis", "reports/l4_failure_analysis.md"),
]


def main():
    with open(REPORT, encoding="utf-8") as f:
        text = f.read()
    if "## 附录 A" in text:
        print("appendices already present; skip")
        return
    parts = [text.rstrip(), ""]
    for letter, title, rel in APPENDICES:
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as f:
            body = f.read().strip()
        parts.append(f"## 附录 {letter} — {title}")
        parts.append("")
        parts.append(body)
        parts.append("")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"appended {len(APPENDICES)} appendices -> {REPORT}")


if __name__ == "__main__":
    main()
