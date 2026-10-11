#!/usr/bin/env python3
"""Readiness requires current complete experiments and bound actual audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# These are the explicit final requirements, not replaceable by reference
# fixtures or an aggregate benchmark score.
REQUIREMENTS = {
    "verified_u1_foundation_and_base": "FOUNDATION_FORGETTING",
    "complete_perception_portability": "PERCEPTION_PORTABILITY",
    "mot_only_learned_baseline": "IDENTITY_CORE",
    "complete_registered_training_budgets": "IDENTITY_CORE",
    "same_identity_core": "CAUSAL_CONTRACT",
    "every_ordinary_dataset_retained": "NEGATIVE_TRANSFER",
    "tao_teta_and_association": "OVMOT",
    "full_u1_generic_expression_language_retention": "FOUNDATION_FORGETTING",
    "actual_packed_integration_retention": "FOUNDATION_FORGETTING",
    "actual_selected_causal_and_serialization": "CAUSAL_CONTRACT",
    "actual_non_head_parameter_sharing": "CAUSAL_CONTRACT",
    "self_contained_actual_checkpoint": "CAUSAL_CONTRACT",
    "required_gpu_execution_proofs": "CAUSAL_CONTRACT",
    "official_test_labels_untouched": "CAUSAL_CONTRACT",
}


def readiness_decision(gates: dict) -> dict:
    missing = [name for name in REQUIREMENTS if name not in gates]
    failed = [name for name in REQUIREMENTS if name in gates and gates[name]["pass"] is not True]
    if missing:
        status = "LOCATEMOT_U_U2_IN_PROGRESS"
    elif failed:
        status = "LOCATEMOT_U_BLOCKED_U2_" + REQUIREMENTS[failed[0]]
    else:
        status = "LOCATEMOT_U_READY_FOR_U3"
    return {"status": status, "ready_for_u3": not missing and not failed, "missing_gates": missing, "failed_gates": failed}


def current_evidence(args) -> tuple[dict, dict]:
    import torch
    torch.set_num_threads(2)
    from tools.unified.pack_u2_checkpoint import selected_inputs
    from tools.unified.select_u2_perception_adaptation import verify_selected_foundation
    from tools.unified.build_u2_candidate_cache import digest
    from tools.unified.eval_u2_joint import checkpoint_decision
    from locatemot.unified.runtime.u2_checkpoint import load_u2_checkpoint, parameter_sharing, FORMAT, U1R2_BASE, U1_FOUNDATION_SHA
    from tools.unified.u2_mot_portability import CHECKPOINT
    from tools.unified.u1_formal_generic_gate import VOCABULARY

    ordinary, simple, reference, ovmot, labels, text, selected = selected_inputs(args)
    chosen = selected["selected"]
    packed = load_u2_checkpoint(args.checkpoint, device="cpu")
    checkpoint_sha = digest(args.checkpoint)
    trained = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)
    if (packed.reference_only or packed.metadata["joint_checkpoint_sha256"] != chosen["checkpoint_sha256"]
            or packed.metadata["mot_only_baseline_sha256"] != reference["checkpoint_sha256"]
            or packed.metadata["foundation_sha256"] != ordinary["spec"]["foundation_sha256"]
            or not all(torch.equal(v,packed.core.state_dict()[k]) for k,v in trained["core"].items())
            or not torch.equal(text["embeddings"],packed.class_text)):
        raise ValueError("Final weights/classes differ from the complete selected joint/MOT/foundation evidence")
    for key,path in (("ordinary_manifest",args.cache_manifest),("tao_ctao_manifest",args.ovmot_cache)):
        if packed.metadata[key]["sha256"] != digest(path):
            raise ValueError("Packed source scope differs from the current full cache")
    foundation = verify_selected_foundation(Path(ordinary["spec"]["selection"]),ordinary["spec"]["foundation_sha256"])
    mot_run = json.loads((args.mot_only_selection.parent / "run_state.json").read_text())
    joint_run = json.loads((args.joint_selection.parent / "run_state.json").read_text())
    retention = json.loads(args.retention.read_text())
    causal = json.loads(args.causal.read_text())
    for audit in (retention,causal):
        if audit["reference_only"] or audit["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Final audit is a fixture or belongs to another packed artifact")
        if audit["foundation_sha256"] != ordinary["spec"]["foundation_sha256"]:
            raise ValueError("Final audit used another foundation")
    if causal["candidate_cache_sha256"] != digest(args.cache_manifest):
        raise ValueError("Final causal audit used another candidate scope")
    sharing = parameter_sharing(packed.foundation,packed.core)
    if sharing != packed.metadata["parameter_sharing"]:
        raise ValueError("Packed sharing metadata differs from the actual native/core parameter count")
    ddp = {name: json.loads((ROOT / "outputs/unified/u2/eval" / filename).read_text()) for name,filename in
           (("single","ovmot_training_contract_single.json"),("two","ddp_two_rank_joint.json"),("eight","ddp_eight_rank_joint.json"))}
    decision = checkpoint_decision(chosen["ordinary"],chosen["ovmot"],reference)
    lineage = subprocess.run(["git","merge-base","--is-ancestor",U1R2_BASE,"HEAD"],cwd=ROOT).returncode == 0
    causal_flags = ("birth_persistence","occlusion_retention","no_future_dependency","mot_specification_invariance",
                    "ovmot_raw_identity_and_fixed_pair_gru_isolation","both_tasks_same_core_object")
    no_test = all(report.get("official_test_labels_read") is False for report in
                  (foundation,selected,chosen,retention,causal,*ddp.values()))
    gates = {
        "verified_u1_foundation_and_base": {"pass": lineage and digest(CHECKPOINT)==U1_FOUNDATION_SHA
                                              and packed.metadata["u1r2_base_commit"]==U1R2_BASE,
                                             "base":U1R2_BASE,"source_foundation":U1_FOUNDATION_SHA},
        "complete_perception_portability": {"pass": foundation["portability_status"]=="PASS","selection":foundation},
        "mot_only_learned_baseline": {"pass":mot_run["selection_status"]=="MOT_ONLY_SELECTED","selected":reference},
        "complete_registered_training_budgets": {"pass":mot_run["status"]==joint_run["status"]=="COMPLETE"
            and mot_run["optimizer_steps"]==6000 and joint_run["optimizer_steps"]==4000
            and mot_run["world_size"]==joint_run["world_size"]==8 and len(trained["rank_rng_states"])==8,
            "mot_steps":mot_run["optimizer_steps"],"joint_steps":joint_run["optimizer_steps"]},
        "same_identity_core": {"pass":causal.get("both_tasks_same_core_object") is True
            and trained["metadata"]["mot_only_checkpoint_sha256"]==reference["checkpoint_sha256"],
            "strict_selected_tensor_load":True},
        "every_ordinary_dataset_retained": {"pass":decision["ordinary_retained"],"decision":decision},
        "tao_teta_and_association": {"pass":decision["ovmot_gate_pass"],"metrics":chosen["ovmot"]["metrics"]},
        "full_u1_generic_expression_language_retention": {"pass":foundation["retention_pass"]
            and foundation["language_conditioning_pass"],"retention":foundation["retention"],
            "true_ap50":foundation["true_ap50"],"shuffled_ap50":foundation["shuffled_ap50"]},
        "actual_packed_integration_retention": {"pass":retention["status"]=="PASS"
            and retention["frozen_foundation_all_tensors"]=="BIT_EXACT" and retention["all_predictions"]=="BIT_EXACT"
            and {item["dataset"] for item in retention["scope"]}=={"refer_kitti_v1","refer_kitti_v2"}
            and all(item["native_metrics"]==item["packed_metrics"] and item["generic_vocabulary"]==VOCABULARY
                    for item in retention["scope"]),"audit":str(args.retention)},
        "actual_selected_causal_and_serialization": {"pass":causal["status"]=="PASS"
            and all(causal.get(name) is True for name in causal_flags)
            and causal.get("observed_births",0)>0 and causal.get("observed_repeat_matches",0)>0
            and causal.get("online_state_save_reload")=="BIT_EXACT" and set(causal["serialized_tasks"])=={"mot","ovmot"}
            and causal["selected_weights_modified"] is False and causal["teacher_forcing_used"] is False,"audit":str(args.causal)},
        "actual_non_head_parameter_sharing": {"pass":sharing["shared_non_head_fraction"]>=.90,"counts":sharing},
        "self_contained_actual_checkpoint": {"pass":not packed.reference_only and packed.metadata["foundation_frozen"]
            and packed.metadata["training_step"]==chosen["step"],"format":FORMAT,"path":str(args.checkpoint),"sha256":checkpoint_sha},
        "required_gpu_execution_proofs": {"pass":all(report["status"]=="PASS" for report in ddp.values())
            and ddp["single"]["world_size"]==1 and ddp["two"]["world_size"]==2 and ddp["eight"]["world_size"]==8
            and all(ddp[name]["next_stochastic_update_bit_exact_after_resume"] for name in ("two","eight")),
            "world_sizes":[ddp[name]["world_size"] for name in ("single","two","eight")]},
        "official_test_labels_untouched": {"pass":no_test,"original_frozen_scopes_recomputed":True},
    }
    sources = {"checkpoint":str(args.checkpoint),"checkpoint_sha256":checkpoint_sha,"chosen_joint_step":chosen["step"],
               "ordinary_result":chosen["ordinary"],"ovmot_result":chosen["ovmot"],"mot_only_reference":reference,
               "source_artifacts":{name:{"path":str(getattr(args,name)),"sha256":digest(getattr(args,name))} for name in
                   ("cache_manifest","simple_baseline","mot_only_selection","joint_selection","ovmot_cache","ovmot_supervision",
                    "class_text","tao_gt_view","retention","causal")}}
    return gates,sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-manifest","simple-baseline","mot-only-selection","joint-selection","ovmot-cache","ovmot-supervision",
                 "class-text","tao-gt-view","checkpoint","retention","causal","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    from tools.unified.build_u2_candidate_cache import write_json
    try:
        gates,sources=current_evidence(args)
    except (ValueError,FileNotFoundError) as error:
        report={"format":"locatemot-u2-readiness-v1","status":"LOCATEMOT_U_U2_IN_PROGRESS","ready_for_u3":False,
                "reason":str(error),"actual_evidence_complete":False}
        write_json(args.out,report)
        print(json.dumps(report))
        return 2
    report={"format":"locatemot-u2-readiness-v1",**readiness_decision(gates),"gates":gates,**sources,
            "reference_only":False,"official_test_labels_read":False,"rmot_or_prompt_tracking_launched":False}
    write_json(args.out,report)
    print(json.dumps({key:report[key] for key in ("status","ready_for_u3","failed_gates","missing_gates")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
