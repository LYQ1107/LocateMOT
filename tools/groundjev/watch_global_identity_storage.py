"""One blocking budget observer; no writes to experiments, no deletions/signals."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.global_identity_storage import inventory
    cap=json.loads((ROOT/'configs/groundjev/global_identity_storage_closed_v2.json').read_text())
    root=ROOT/'outputs/groundjev/global_identity_v1';source=Path('/data2/user/LocateMOT_GroundJEV/outputs/groundjev')
    tables=[root/'R2_all_five_scopes.json',root/'memory_controls/all_five_scopes.json',root/'A4_all_five_scopes.json',root/'A4b_mixed_all_five_scopes.json']
    warned=False;first=True
    while True:
        stage=inventory(root)['conservative_temporary_total'];old=inventory(source)['conservative_temporary_total']
        stage_remaining=cap['global_identity_stage_cap_bytes']-stage;global_remaining=cap['global_limit_bytes']-old-stage
        complete=all(p.exists() for p in tables)
        status='VALIDATED' if stage_remaining>=0 and global_remaining>=0 else 'BUDGET_RECONCILIATION_REQUIRED'
        near=stage_remaining<256*1024**2 or global_remaining<512*1024**2
        if first or complete or status!='VALIDATED' or near and not warned:
            result=dict(status=status,actual_source_conservative_bytes=old,actual_new_stage_conservative_bytes=stage,
                actual_combined_conservative_bytes=old+stage,remaining_stage_cap_bytes=stage_remaining,
                remaining_global_limit_bytes=global_remaining,accounting_config='configs/groundjev/global_identity_storage_closed_v2.json',
                all_four_registered_evaluation_tables_exist=complete,files_deleted=0,workers_signaled=0)
            (root/'storage_live_observer.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        assert status=='VALIDATED','Budget reconciliation required; preserve all files and workers, investigate completed unused caps'
        if complete:return
        first=False;warned=warned or near;time.sleep(60)
if __name__=='__main__':main()
