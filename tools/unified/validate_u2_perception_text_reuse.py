#!/usr/bin/env python3
"""Actual unchanged TAO/C-TAO smoke rows under optional fixed-text reuse."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main() -> int:
    from tools.unified.build_u2_candidate_cache import write_json
    from tools.unified.u2_mot_portability import FOUNDATION_SHA
    reports=[]
    with tempfile.TemporaryDirectory(prefix="locatemot-u2-perception-text-reuse-") as directory:
        for mode in ("native","reuse"):
            out=Path(directory)/mode
            command=[sys.executable,"-u","tools/unified/u2_tao_portability.py","--smoke","--worker-index","0",
                     "--worker-count","1","--gpu","0","--extraction-pid","625610","--out",str(out)]
            if mode=="reuse":
                command.append("--reuse-frozen-text")
            subprocess.run(command,cwd=ROOT,check=True)
            prefix=out/"worker_0"
            rows=[json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
            result=json.loads(prefix.with_suffix(".result.json").read_text())
            assert result["state"]=="SMOKE_COMPLETE" and result["metadata"]["smoke"] is True
            reports.append({"rows":rows,"terminal":result})
        native,reused=reports
        assert native["rows"]==reused["rows"],"Native proposal/recall rows changed under text reuse"
        assert native["terminal"]["metadata"]==reused["terminal"]["metadata"]
        assert {row["dataset"] for row in native["rows"]}=={"tao","ctao"}
        assert all(row["backbone_calls"]==1 for row in reused["rows"])
        stats=reused["terminal"]["frozen_text_cache"]
        assert stats["bert_calls"]==38 and stats["bert_hits"]==38
        report={"format":"locatemot-u2-perception-text-reuse-contract-v1","status":"PASS","reference_only":True,
                "foundation_sha256":FOUNDATION_SHA,"actual_native_rows":native["rows"],"actual_reused_rows":reused["rows"],
                "frozen_scope_metadata_identical":True,"all_proposal_recall_counts_identical":True,
                "frozen_text_cache":stats,"native_elapsed_seconds":native["terminal"]["elapsed_seconds"],
                "reuse_elapsed_seconds":reused["terminal"]["elapsed_seconds"],"backbone_once_per_image":True,
                "scope":"original --smoke: one original frame from each of TAO/C-TAO; not a full audit",
                "current_full_f0_jobs_restarted_or_modified":False,"actual_adapted_full_audit_complete":False,
                "official_test_labels_read":False,"ready_for_u3":False}
        write_json(ROOT/"outputs/unified/u2/eval/perception_text_reuse_contract.json",report)
        print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
