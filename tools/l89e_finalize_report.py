#!/usr/bin/env python3
"""Write the bounded L89E evidence reports after all authorized replays."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from l89d_fullvideo_common import MANIFEST_SHA, THREAD, WORK_ROOT, sha256_file, write_json  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def pct(result: dict[str, Any], key: str) -> float | None:
    return result.get("metrics_percent", {}).get(key)


def _dataset_result(result: dict[str, Any], dataset: str) -> dict[str, Any]:
    for item in result.get("per_dataset", []):
        if str(item.get("dataset")) == dataset:
            return item
    raise KeyError(f"missing dataset {dataset}")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    stage_s = load(args.stage_s / "summary.json")
    merged = load(args.merged / "summary.json")
    shortlist = load(args.shortlist / "shortlist.json")
    dev_matrix = load(args.dev_trackeval / "trackeval_matrix.json")
    selection = load(args.selection / "checkpoint_selection.json")
    semantic = load(args.semantic / "semantic.json")
    gate = load(args.semantic / "gate_decision.json")
    internal = load(args.internal_trackeval / "trackeval_matrix.json")
    if stage_s.get("status") != "complete" or stage_s.get("record_count") != 1992:
        raise AssertionError("Stage-S report input incomplete")
    if merged.get("status") != "complete" or merged.get("record_count") != 9960:
        raise AssertionError("merged report input incomplete")
    if shortlist.get("status") != "complete" or not shortlist.get("phase_consistent_temporal"):
        raise AssertionError("shortlist report input incomplete")
    if dev_matrix.get("status") != "complete" or len(dev_matrix.get("results", [])) != int(shortlist["shortlist_count"]) * 3:
        raise AssertionError("dev TrackEval input incomplete")
    if selection.get("status") != "complete" or not selection.get("selection_frozen_before_fixed_validation"):
        raise AssertionError("selection input incomplete")
    if semantic.get("status") != "complete" or gate.get("status") != "complete":
        raise AssertionError("semantic input incomplete")
    if internal.get("status") != "complete" or len(internal.get("results", [])) != 1:
        raise AssertionError("internal TrackEval input incomplete")
    final = selection["final_selection"]
    selected_epoch = int(final["checkpoint_info"]["epoch"])
    selected_phase = str(final.get("phase_policy", {}).get("phase") or final["checkpoint_info"].get("phase"))
    selected_rule = str(final["rule"])
    policy = dict(final.get("phase_policy") or {})
    semantic_validation = semantic["validation"]
    dev_rows = dev_matrix["results"]
    internal_result = internal["results"][0]
    v1 = _dataset_result(internal_result, "refer_kitti_v1")
    v2 = _dataset_result(internal_result, "refer_kitti_v2")
    report_root = args.out.resolve().parent
    _write(report_root / "L89E_PROTOCOL_AUDIT.md", f"""# L89E protocol audit

L89E is a zero-training phase/history replay from L89D base commit
`d0960e1d1679414765f79203bef444f0f9968928`.

The confirmed training contract is S=`temporal_enabled=false` plus zero
history for epochs 1–8; T=`true` plus last-four causal history for epochs
9–20; and J=`true` plus last-four causal history for epochs 21–40.  The
phase policy was implemented once in `tools/l89e_phase_policy.py` and checked
against frozen `L86ClipStore` packing by the required CPU equality test.

L89D had a valid native timeline but used temporal-on/raw history for every
checkpoint.  L89E changed only this evaluation contract.  No L89 model,
loss, bank, tracker, candidate rows, UIDM, ordinary MOT or OVMOT source was
changed.  No dense Z1 or language cache was rebuilt.

The fixed manifest SHA remained `{MANIFEST_SHA}`.  Token/span-region and
static/motion alignment remain `UNALIGNED`.
""")
    _write(report_root / "L89E_STAGE_S_RESCORE.md", f"""# L89E Stage-S sparse rescore

Stage-S epochs `{stage_s['checkpoint_epochs']}` were rescored with
`temporal_enabled=false` and `history_contract=zero_history`.  The output has
{stage_s['dev_group_count']} legal dev groups, exactly {stage_s['record_count']}
records, and {stage_s['record_count_per_checkpoint']} records per checkpoint.
The corrected B/R/P rule fits are fit/dev-only; fixed calibration and
validation were not read in this stage.  Candidate rows were retained with no
deletion or truncation, and `zero_training=true`.

The historical T/J epochs 10–40 were not re-forwarded: their original sparse
path already used `build_frame(..., temporal_enabled=True)` and L86's last-four
causal packing.  Their score values were copied unchanged at merge time.
""")
    _write(report_root / "L89E_DEV_SELECTION.md", f"""# L89E dev selection

The merged fit/dev pool contains 20 epochs and {merged['record_count']} rows.
The registered shortlist contains {shortlist['shortlist_count']} candidates
with epochs `{shortlist['shortlist_epochs']}`.  The true-native-timeline dev
matrix contains {dev_matrix['result_count']} complete TrackEval results
(each candidate under B/R/P).

The preregistered tuple
`(HOTA, DetA, AssA, distinct_target_recall, -inactive_false_acceptance,
-epoch, deterministic_rule_order)` selected epoch `{selected_epoch}`,
phase `{selected_phase}`, Rule `{selected_rule}`.  Selection was frozen before
fixed validation and no fixed calibration/validation labels were used here.

Phase policy: `{json.dumps(policy, sort_keys=True)}`.
""")
    _write(report_root / "L89E_FIXED_SEMANTIC.md", f"""# L89E fixed semantic replay

This is a fixed 16-calibration/24-validation semantic diagnostic, not
screening, official-test, HOTA or a production RMOT result.  The selected
epoch/rule/threshold was frozen before validation.  The unchanged corrected
candidate-vs-NULL emission contract was used.

| metric | validation |
|---|---:|
| recall | {semantic_validation['legacy_candidate_recall']:.7f} |
| precision | {semantic_validation['legacy_candidate_precision']:.7f} |
| FP/frame | {semantic_validation['legacy_fp_per_frame']:.7f} |
| predictions/positive | {semantic_validation['legacy_predictions_per_positive']:.7f} |
| hard violation | {semantic_validation['legacy_row_hard_violation']:.7f} |
| multi-positive recall | {semantic_validation['legacy_row_multi_positive_recall']:.7f} |
| inactive false acceptance | {semantic_validation['inactive_false_acceptance']:.7f} |
| empty rate | {semantic_validation['empty_rate']:.7f} |

Decision: `{gate['decision']}`.  The registered recall and multi-positive
floors are not met, so this remains a semantic-gate failure despite lower
volume and improved precision.  No threshold rescue was performed.
""")
    _write(report_root / "L89E_INTERNAL_TRACKEVAL.md", f"""# L89E internal TrackEval

The frozen selection was replayed over the complete native internal timeline:
V1 videos 0004/0018 and V2 videos 0016/0017/0020.  This is internal
validation-scope TrackEval only; screening and official-test labels were not
read.

| dataset | HOTA | DetA | AssA | DetRe | DetPr | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| V1 | {pct(v1, 'HOTA___AUC'):.4f} | {pct(v1, 'DetA___AUC'):.4f} | {pct(v1, 'AssA___AUC'):.4f} | {pct(v1, 'DetRe___AUC'):.4f} | {pct(v1, 'DetPr___AUC'):.4f} | {v1.get('metrics_counts', {}).get('IDSW', 'n/a')} |
| V2 | {pct(v2, 'HOTA___AUC'):.4f} | {pct(v2, 'DetA___AUC'):.4f} | {pct(v2, 'AssA___AUC'):.4f} | {pct(v2, 'DetRe___AUC'):.4f} | {pct(v2, 'DetPr___AUC'):.4f} | {v2.get('metrics_counts', {}).get('IDSW', 'n/a')} |

The phase/history contract passed for the selected checkpoint: `{json.dumps(policy, sort_keys=True)}`.
""")
    _write(args.out.resolve(), f"""# L89E — phase-consistent temporal/history replay final report

## Executive decision

`L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY / {gate['decision']} / STOPPED_PENDING_SUPERVISOR_REVIEW`

L89E repaired the checkpoint phase/history evaluation contract and completed
the authorized true-full-video dev selection, fixed semantic replay, and
internal V1/V2 TrackEval.  It did not train or change the QSC-D model.

## Source and boundary

- base L89D commit: `d0960e1d1679414765f79203bef444f0f9968928`
- branch: `codex/l89e-phase-consistent-temporal-replay-20260909`
- thread: `{THREAD}`
- fixed manifest SHA: `{MANIFEST_SHA}`
- selected checkpoint: `{final['checkpoint_info']['path']}`
- selected checkpoint SHA: `{final['checkpoint_info']['sha256']}`
- selected epoch/phase/rule: `{selected_epoch}` / `{selected_phase}` / `{selected_rule}`
- selected policy: `{json.dumps(policy, sort_keys=True)}`

The older L89D report contains historical base-commit text from before its
final commit; this report uses the actual L89D base commit above.

## Phase/history correction

| checkpoint | train phase | train temporal | eval temporal | eval history | consistent |
|---|---|---:|---:|---|---|
| 2/4/6/8 | S | off | off | zero | yes |
| 10–20 | T | on | on | last4 causal | yes |
| 22–40 | J | on | on | last4 causal | yes |

Stage-S produced exactly {stage_s['record_count']} sparse records.  The
historical T/J score rows were reused unchanged, producing {merged['record_count']}
records over 20 epochs.  No new checkpoint or cache was created.

## True-full-video dev and fixed semantic

The dev matrix has {dev_matrix['result_count']} complete TrackEval results.
Selection used only fit/dev TrackEval and the registered tuple, then froze
epoch `{selected_epoch}` / Rule `{selected_rule}` before fixed validation.

Fixed validation metrics:

| recall | precision | FP/frame | pred/positive | hard | multi-positive | inactive FA | gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| {semantic_validation['legacy_candidate_recall']:.7f} | {semantic_validation['legacy_candidate_precision']:.7f} | {semantic_validation['legacy_fp_per_frame']:.7f} | {semantic_validation['legacy_predictions_per_positive']:.7f} | {semantic_validation['legacy_row_hard_violation']:.7f} | {semantic_validation['legacy_row_multi_positive_recall']:.7f} | {semantic_validation['inactive_false_acceptance']:.7f} | `{gate['decision']}` |

Recall and multi-positive recall fail the registered floors.  The result is
therefore not a semantic correspondence pass and is not used to authorize a
new training run.

## Internal TrackEval and historical comparison

| evidence | V1 HOTA | V2 HOTA | authority |
|---|---:|---:|---|
| L86 | 29.1663 | 21.6467 | historical full-video comparison |
| L87-A | 28.5752 | 22.1300 | historical full-video comparison |
| L89D phase/history-inconsistent | 25.0799 | 22.6374 | valid timeline, inconsistent phase/history |
| L89E phase-consistent | {pct(v1, 'HOTA___AUC'):.4f} | {pct(v2, 'HOTA___AUC'):.4f} | valid internal validation-scope TrackEval |

L89E internal metrics:

- V1: HOTA `{pct(v1, 'HOTA___AUC'):.4f}%`, DetA `{pct(v1, 'DetA___AUC'):.4f}%`,
  AssA `{pct(v1, 'AssA___AUC'):.4f}%`, DetRe `{pct(v1, 'DetRe___AUC'):.4f}%`,
  DetPr `{pct(v1, 'DetPr___AUC'):.4f}%`.
- V2: HOTA `{pct(v2, 'HOTA___AUC'):.4f}%`, DetA `{pct(v2, 'DetA___AUC'):.4f}%`,
  AssA `{pct(v2, 'AssA___AUC'):.4f}%`, DetRe `{pct(v2, 'DetRe___AUC'):.4f}%`,
  DetPr `{pct(v2, 'DetPr___AUC'):.4f}%`.

These are not screening or official-test results.  No HOTA fast-screening
claim is made; they are legal internal TrackEval measurements only.

## Final boundaries and next action

`zero_training=true`, `new_checkpoint_created=false`,
`checkpoint_weights_changed=false`, `screening_gt_used=false`,
`official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
Token/span-region and static/motion alignment remain `UNALIGNED`.

The unique next action is supervisor review of one separately authorized
absence/volume-calibration study.  Do not extend L89/L89C/L89E, change the
threshold, add a new NULL head, rebuild the bank, or modify ordinary
MOT/OVMOT in this stage.

## Artifact paths

- Stage-S: `{args.stage_s.resolve()}`
- merged scores: `{args.merged.resolve()}`
- shortlist: `{args.shortlist.resolve()}`
- dev true-full-video TrackEval: `{args.dev_trackeval.resolve()}`
- frozen selection: `{args.selection.resolve()}`
- fixed semantic: `{args.semantic.resolve()}`
- internal TrackEval: `{args.internal_trackeval.resolve()}`
""")
    final_status = {
        "format": "locatemot-l89e-final-status-v1",
        "status": f"L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY / {gate['decision']} / STOPPED_PENDING_SUPERVISOR_REVIEW",
        "base_l89d_sha": "d0960e1d1679414765f79203bef444f0f9968928",
        "l89e_code_sha": None,
        "zero_training": True,
        "phase_consistent_temporal": True,
        "stage_s_sparse_rescored": True,
        "tj_sparse_scores_reused": True,
        "dense_z1_rebuilt": False,
        "selected_epoch": selected_epoch,
        "selected_phase": selected_phase,
        "selected_rule": selected_rule,
        "selected_temporal_enabled": bool(policy.get("temporal_enabled")),
        "selected_history_mode": policy.get("history_mode"),
        "selected_history_length": policy.get("history_length"),
        "fixed_semantic_gate": gate["decision"],
        "fixed_semantic_validation": semantic_validation,
        "v1_hota": pct(v1, "HOTA___AUC"),
        "v1_deta": pct(v1, "DetA___AUC"),
        "v1_assa": pct(v1, "AssA___AUC"),
        "v1_detre": pct(v1, "DetRe___AUC"),
        "v1_detpr": pct(v1, "DetPr___AUC"),
        "v2_hota": pct(v2, "HOTA___AUC"),
        "v2_deta": pct(v2, "DetA___AUC"),
        "v2_assa": pct(v2, "AssA___AUC"),
        "v2_detre": pct(v2, "DetRe___AUC"),
        "v2_detpr": pct(v2, "DetPr___AUC"),
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": True,
        "no_hota_screening_claim": True,
        "manifest_sha256": MANIFEST_SHA,
        "report": str(args.out.resolve()),
    }
    write_json(args.final_status, final_status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-s", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--shortlist", type=Path, required=True)
    parser.add_argument("--dev-trackeval", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--internal-trackeval", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final-status", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
