#!/usr/bin/env python3
"""One blocking owner: selected joint -> actual pack/audits -> final evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.unified.build_u2_candidate_cache import DIRECTORY,digest,write_json
from tools.unified.queue_u2_ordinary_baseline import wait_process


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-joint-pid",type=int,required=True)
    args=parser.parse_args()
    directory=ROOT / "outputs/unified/u2/finalization"
    state_path=directory / "run_state.json"
    state={"format":"locatemot-u2-finalization-queue-v1","state":"WAITING_ORIGINAL_JOINT_OWNER",
           "controller_pid":os.getpid(),"joint_controller_pid":args.wait_joint_pid,"pack_started":False,"ready_for_u3":False}
    write_json(state_path,state)
    print(json.dumps(state),flush=True)
    wait_process(args.wait_joint_pid)
    try:
        joint=json.loads((ROOT / "outputs/unified/u2/joint_queue/run_state.json").read_text())
        if joint["state"]!="COMPLETE" or joint["selection_status"]!="U2_B_JOINT_SELECTED":
            state.update(state="HELD_UPSTREAM_NOT_SELECTED",upstream_terminal_state=joint,
                         pending="bounded U2-C remains required if eligible; no U2-B pack or readiness claim")
            write_json(state_path,state)
            print(json.dumps(state),flush=True)
            return 0
        ordinary=json.loads((ROOT / "outputs/unified/u2/ordinary_baseline/run_state.json").read_text())
        mot=json.loads((ROOT / "outputs/unified/u2/mot_only_queue/run_state.json").read_text())
        simple_path=Path(ordinary["result"])
        simple=json.loads(simple_path.read_text())
        cache_path=Path(simple["candidate_cache_manifest"])
        spec=json.loads(cache_path.read_text())["spec"]
        inputs={"cache-manifest":cache_path,"simple-baseline":simple_path,"mot-only-selection":Path(mot["selection"]),
                "joint-selection":Path(joint["selection"]),"ovmot-cache":DIRECTORY / spec["foundation_sha256"] / "ovmot/complete_manifest.json",
                "ovmot-supervision":ROOT / "outputs/unified/u2/ovmot_supervision/manifest.json",
                "class-text":ROOT / "outputs/unified/u2/ovmot_semantics" / spec["foundation_sha256"] / "class_text.pt",
                "tao-gt-view":ROOT / "outputs/unified/u2/teta/gt/gt_view.json"}
        from tools.unified.pack_u2_checkpoint import selected_inputs
        *_,selection=selected_inputs(argparse.Namespace(**{key.replace("-","_"):value for key,value in inputs.items()}))
        checkpoint=ROOT / "outputs/unified/u2/final" / (selection["selected"]["checkpoint_sha256"][:16]+".pth")
        common=[item for key,value in inputs.items() for item in ("--"+key,str(value))]

        def run(name,script,arguments):
            command=[sys.executable,"-u",script,*arguments]
            state.update(state=name,command=command)
            write_json(state_path,state)
            print(json.dumps(state),flush=True)
            with (directory / (name.lower()+".log")).open("a") as log:
                subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)

        pack_report=checkpoint.with_suffix(".contract.json")
        if pack_report.is_file():
            receipt=json.loads(pack_report.read_text())
            if (receipt["sha256"]!=digest(checkpoint) or receipt["reference_only"]
                    or receipt["metadata"]["joint_checkpoint_sha256"]!=selection["selected"]["checkpoint_sha256"]):
                raise ValueError("Existing packed artifact differs from its actual selected receipt")
        else:
            state["pack_started"]=True
            run("PACKING_SELECTED_NATIVE_MODEL","tools/unified/pack_u2_checkpoint.py",common+["--out",str(checkpoint)])
        retention=directory / "retention.json"
        causal=directory / "causal.json"
        for name,script,output,extra in (
            ("AUDITING_PACKED_U1_RETENTION","tools/unified/audit_u2_checkpoint_retention.py",retention,[]),
            ("AUDITING_SELECTED_CORE_CONTRACT","tools/unified/audit_u2_selected_core.py",causal,["--cache-manifest",str(cache_path)])):
            if output.is_file():
                previous=json.loads(output.read_text())
                if (previous["status"]=="PASS" and previous["checkpoint_sha256"]==digest(checkpoint)
                        and previous["reference_only"] is False):
                    continue
            run(name,script,["--checkpoint",str(checkpoint),"--out",str(output),*extra])
        ready_path=directory / "readiness.json"
        run("VERIFYING_ALL_CURRENT_U2_GATES","tools/unified/verify_u2_ready.py",common+[
            "--checkpoint",str(checkpoint),"--retention",str(retention),"--causal",str(causal),"--out",str(ready_path)])
        ready=json.loads(ready_path.read_text())
        summary_path=ROOT / "outputs/unified/u2/final_status.json"
        summary=json.loads(summary_path.read_text())
        summary.update(status=ready["status"],ready_for_u3=ready["ready_for_u3"],execution_state="U2_FINAL_EVIDENCE_COMPLETE",
                       u2_final_checkpoint={"path":str(checkpoint),"sha256":digest(checkpoint)},
                       final_readiness_evidence=str(ready_path),foundation_retention=str(retention),
                       causal_contract=str(causal),shared_non_head_fraction=ready["gates"]["actual_non_head_parameter_sharing"]["counts"]["shared_non_head_fraction"],
                       stop=None if ready["ready_for_u3"] else ready["failed_gates"])
        write_json(summary_path,summary)
        state.update(state="COMPLETE",readiness=str(ready_path),status=ready["status"],ready_for_u3=ready["ready_for_u3"],checkpoint=str(checkpoint))
        write_json(state_path,state)
        print(json.dumps(state),flush=True)
        return 0
    except Exception as error:
        # Preserve a genuine bound causal failure as the explicit research
        # result; other command failures remain actionable execution errors.
        audit_files={"AUDITING_SELECTED_CORE_CONTRACT":directory / "causal.json",
                     "AUDITING_PACKED_U1_RETENTION":directory / "retention.json"}
        audit_path=audit_files.get(state["state"])
        failure=json.loads(audit_path.read_text()) if audit_path is not None and audit_path.is_file() else {}
        if (failure.get("status")=="FAIL" and failure.get("blocked_status") in
                {"LOCATEMOT_U_BLOCKED_U2_CAUSAL_CONTRACT","LOCATEMOT_U_BLOCKED_U2_FOUNDATION_FORGETTING"}
                and failure["checkpoint_sha256"]==digest(checkpoint)):
            summary_path=ROOT / "outputs/unified/u2/final_status.json"
            summary=json.loads(summary_path.read_text())
            summary.update(status=failure["blocked_status"],ready_for_u3=False,stop=failure["error"],failed_final_audit=str(audit_path))
            write_json(summary_path,summary)
        state.update(state="FAILED",error=str(error))
        write_json(state_path,state)
        raise


if __name__=="__main__":
    raise SystemExit(main())
