"""One blocking collector for completed original R2/R3/A4 owners."""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from tools.groundjev.analyze_global_identity import analyze
    plan=json.loads((ROOT/'configs/groundjev/global_identity_v1.json').read_text());source=Path(plan['immutable_source_root'])
    scopes=json.loads((source/plan['evaluation_plan']).read_text())['scopes'];root=ROOT/plan['output']
    for stage,table,owner in [('R2',root/'R2_all_five_scopes.json',root/'R2_owner/owner_state.json'),
        ('A3',root/'memory_controls/all_five_scopes.json',root/'memory_controls/owner_state.json'),
        ('A4',root/'A4_all_five_scopes.json',root/'A4_owner/owner_state.json')]:
        print(json.dumps(dict(status='WAITING_ON_ORIGINAL_COMPLETION_OWNER',stage=stage)),flush=True)
        while not table.exists():
            if owner.exists():
                state=json.loads(owner.read_text())
                assert 'FAILED' not in state['status'],state
                pid=Path('/proc')/str(state['owner_pid'])
                assert pid.exists(),f'{stage} owner exited without completion receipt; retain partial output and investigate'
            time.sleep(10)
        registered=json.loads(table.read_text());output=root/(stage+'_all_five_identity_diagnostics_v2.json')
        assert not output.exists();result=dict(status='FULL_REGISTERED_TABLE_DIAGNOSTICS_RUNNING',diagnostic_version=2,source_table=str(table),completed={},full_goal_complete=False)
        for name,record in registered['completed'].items():
            scope=next(s for s in scopes if name.endswith('_'+s))
            manifest=json.loads((source/scopes[scope]['manifest']).read_text());diagnostic=analyze(manifest,Path(record['run']))
            result['completed'][name]=dict(official_metrics=record['official_metrics'],diagnostics=diagnostic)
            output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(stage=stage,completed=name)),flush=True)
        result['status']='FULL_REGISTERED_TABLE_DIAGNOSTICS_COMPLETE';output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
