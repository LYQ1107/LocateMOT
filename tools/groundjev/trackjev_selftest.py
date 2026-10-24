"""Self-test for the TrackJEV decision layer (no GPU, no model, no data).

Run: python tools/groundjev/trackjev_selftest.py
Exits non-zero on the first contract violation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from locatemot.groundjev.trackjev import candidate_competition as cc
from locatemot.groundjev.trackjev import decision_schema as ds
from locatemot.groundjev.trackjev import global_assignment as ga


def main() -> int:
    report = {}

    # two real objects; candidate 1 is a duplicate of object A; candidate 4 is background
    boxes = [
        [100.0, 100.0, 50.0, 100.0],   # 0: object A (good box)
        [104.0, 102.0, 50.0, 100.0],   # 1: duplicate of A
        [400.0, 120.0, 60.0, 90.0],    # 2: object B
        [402.0, 118.0, 62.0, 92.0],    # 3: duplicate of B
        [700.0, 500.0, 40.0, 40.0],    # 4: background
    ]
    gt_boxes = [[100.0, 100.0, 50.0, 100.0], [400.0, 120.0, 60.0, 90.0]]
    gt_ids = [11, 22]

    edges = cc.sparse_edges(boxes, None)
    clusters = cc.entity_clusters(boxes, edges)
    targets = cc.admission_targets(boxes, gt_boxes, gt_ids, clusters=clusters)
    rows = cc.competition_rows(boxes, clusters, edges)
    consistency = cc.cluster_consistency(clusters, edges)

    labels = [t.label for t in targets]
    assert labels[0] == "valid_entity", labels
    assert labels[1] == "duplicate" and targets[1].duplicate_of == 0, labels
    assert labels[2] == "valid_entity" and labels[3] == "duplicate", labels
    assert labels[4] == "background", labels
    assert max(len(c) for c in clusters) <= cc.MAX_GROUP
    assert consistency["max_cluster"] == 2, consistency
    report["competition"] = {
        "labels": labels,
        "clusters": clusters,
        "edges": len(edges),
        "consistency": consistency,
        "iou_row0": rows["competition_iou_rows_milli"][0],
    }

    # Native GroundAnything/OnlineTracker geometry is xyxy.  Verify the
    # explicit boundary gives the same overlap as the equivalent xywh case.
    native_boxes = [[100.0, 100.0, 150.0, 200.0],
                    [104.0, 102.0, 154.0, 202.0],
                    [400.0, 120.0, 460.0, 210.0]]
    native_edges = cc.sparse_edges(native_boxes, None, box_format="xyxy")
    native_clusters = cc.entity_clusters(native_boxes, native_edges)
    native_rows = cc.competition_rows(native_boxes, native_clusters, native_edges,
                                      box_format="xyxy")
    assert native_rows["competition_box_format"] == "xyxy", native_rows
    assert native_rows["competition_iou_rows_milli"][0].split(",")[1] == rows["competition_iou_rows_milli"][0].split(",")[1]
    report["native_xyxy_contract"] = {
        "box_format": native_rows["competition_box_format"],
        "iou_row0": native_rows["competition_iou_rows_milli"][0],
    }

    # partial annotation must never produce background labels
    partial = cc.admission_targets(boxes, gt_boxes, gt_ids, clusters=clusters,
                                   annotation_exhaustive=False)
    assert partial[4].label == "unknown", partial[4]
    report["partial_annotation_guard"] = {"candidate4": partial[4].label}

    # decision vocabulary
    options = ds.build_identity_options(
        [{"track_id": 11, "lost_age": 0}, {"track_id": 33, "lost_age": 12}])
    tokens = [o.token for o in options]
    assert tokens == ["11", "33", ds.NEW_TOKEN, ds.NO_MATCH_TOKEN, ds.REJECT_TOKEN], tokens
    assert len(options) <= ds.MAX_IDENTITY_OPTIONS

    parsed = ds.parse_identity_answer(0, {"11": 0.62, "33": 0.11, "NEW": 0.2,
                                          "NO_MATCH": 0.04, "REJECT": 0.03}, options)
    assert parsed.decision == "MATCH" and parsed.track_id == 11, parsed
    parsed_reactivate = ds.parse_identity_answer(
        0, {"11": 0.1, "33": 0.7, "NEW": 0.1, "NO_MATCH": 0.05, "REJECT": 0.05}, options)
    assert parsed_reactivate.decision == "REACTIVATE" and parsed_reactivate.track_id == 33
    parsed_new = ds.parse_identity_answer(0, {"11": 0.1, "33": 0.1, "NEW": 0.7,
                                              "NO_MATCH": 0.05, "REJECT": 0.05}, options)
    assert parsed_new.decision == "NEW" and parsed_new.track_id is None
    report["decisions"] = {"match": parsed.to_dict(), "reactivate": parsed_reactivate.decision,
                           "new": parsed_new.decision}

    # supervised decision contract
    assert ds.supervised_decision(physical_identity=11, identity_valid=True, is_duplicate=True,
                                  is_valid_entity=True, identity_in_shortlist=True)[0] == "REJECT"
    assert ds.supervised_decision(physical_identity=11, identity_valid=True, is_duplicate=False,
                                  is_valid_entity=True,
                                  identity_in_shortlist=False) == ("MATCH", 11)
    assert ds.supervised_decision(physical_identity=None, identity_valid=True, is_duplicate=False,
                                  is_valid_entity=True,
                                  identity_in_shortlist=False)[0] == "NEW"
    assert ds.supervised_decision(physical_identity=11, identity_valid=False, is_duplicate=False,
                                  is_valid_entity=True,
                                  identity_in_shortlist=False)[0] == "NO_MATCH"
    assert ds.supervised_decision(physical_identity=None, identity_valid=False, is_duplicate=False,
                                  is_valid_entity=False,
                                  identity_in_shortlist=None)[0] == "UNKNOWN"
    assert ds.slot_targets("MATCH", 11) == {"identity_choice": "11"}
    assert ds.slot_targets("NEW", None) == {"identity_choice": ds.NEW_TOKEN}
    assert ds.slot_targets("NO_MATCH", None) == {"identity_choice": ds.NO_MATCH_TOKEN}
    assert ds.shortlist_coverage([11, 22], 22) is True
    assert ds.shortlist_coverage([11], 22) is False
    assert ds.shortlist_coverage([11], None) is None
    report["labels"] = {"reject_priority": True, "no_match_not_new": True,
                        "coverage": "measured"}

    # assignment: two candidates competing for identity 11 -> exactly one wins
    probs = [{"11": 0.9, "NEW": 0.05, "REJECT": 0.05},
             {"11": 0.8, "NEW": 0.1, "REJECT": 0.1},
             {"11": 0.05, "NEW": 0.9, "REJECT": 0.05}]
    result = ga.assign(probs, options)
    assigned_ids = [t for t in result.assignment.values()
                    if t not in (ds.NEW_TOKEN, ds.REJECT_TOKEN)]
    assert assigned_ids == ["11"], result.to_dict()
    assert result.diagnostics["one_identity_per_frame"] is True
    assert result.assignment[2] == ds.NEW_TOKEN, result.to_dict()
    # the losing duplicate must not receive an identity of its own
    assert result.assignment[1] in (ds.REJECT_TOKEN,) or 1 in result.unmatched, result.to_dict()
    assert result.solver in ("hungarian", "greedy")
    # a solver that cannot honour uniqueness must be corrected by the post-pass
    tight_options = [ds.IdentityOption(token="11", track_id=11)]
    tight = ga.assign([{"11": 1.0}, {"11": 1.0}], tight_options, solver="greedy")
    owners = [t for t in tight.assignment.values() if t == "11"]
    assert len(owners) == 1, tight.to_dict()
    ops = ga.assignment_to_memory_ops(result)
    assert any(op["op"] == "ASSIGN" and op["track_id"] == 11 for op in ops), ops
    report["assignment"] = result.to_dict()
    report["uniqueness_postpass"] = tight.to_dict()
    report["memory_ops"] = ops

    print(json.dumps({"status": "TRACKJEV_DECISION_LAYER_SELFTEST_PASS", **report},
                     indent=1, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
