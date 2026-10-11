#!/usr/bin/env python3
"""Targeted premature-readiness rejection, with no actual eligibility claim."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main() -> int:
    from tools.unified.verify_u2_ready import REQUIREMENTS,readiness_decision
    from tools.unified.build_u2_candidate_cache import write_json
    expected={"verified_u1_foundation_and_base","complete_perception_portability","mot_only_learned_baseline",
              "complete_registered_training_budgets","same_identity_core","every_ordinary_dataset_retained",
              "tao_teta_and_association","full_u1_generic_expression_language_retention","actual_packed_integration_retention",
              "actual_selected_causal_and_serialization","actual_non_head_parameter_sharing","self_contained_actual_checkpoint",
              "required_gpu_execution_proofs","official_test_labels_untouched"}
    assert set(REQUIREMENTS)==expected, "The final requirement set must preserve the explicit U2 specification"
    gates={name:{"pass":True,"source":"explicitly synthetic gate fixture"} for name in REQUIREMENTS}
    assert readiness_decision(gates)["ready_for_u3"]
    missing=copy.deepcopy(gates)
    missing.pop("actual_selected_causal_and_serialization")
    result=readiness_decision(missing)
    assert not result["ready_for_u3"] and result["status"]=="LOCATEMOT_U_U2_IN_PROGRESS"
    rejected={}
    cases={"every_ordinary_dataset_retained":"NEGATIVE_TRANSFER","tao_teta_and_association":"OVMOT",
           "actual_packed_integration_retention":"FOUNDATION_FORGETTING","actual_selected_causal_and_serialization":"CAUSAL_CONTRACT"}
    for name,cause in cases.items():
        changed=copy.deepcopy(gates)
        changed[name]["pass"]=False
        decision=readiness_decision(changed)
        assert not decision["ready_for_u3"] and decision["failed_gates"]==[name]
        assert decision["status"]=="LOCATEMOT_U_BLOCKED_U2_"+cause
        rejected[name]=decision["status"]
    report={"format":"locatemot-u2-readiness-gate-contract-v1","status":"PASS","synthetic_only":True,
            "explicit_complete_requirement_set_preserved":True,
            "missing_selected_audit_does_not_become_ready":True,"independent_material_failure_cases":rejected,
            "actual_experiments_used_to_claim_readiness":False,"ready_for_u3":False}
    write_json(ROOT / "outputs/unified/u2/eval/readiness_gate_contract.json",report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
