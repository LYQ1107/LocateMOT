"""Read-only official prediction diagnostics; write only a new scalar table."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    from tools.groundjev.analyze_global_identity import analyze
    p=argparse.ArgumentParser();p.add_argument('--table',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--only-before',action='store_true');a=p.parse_args()
    plan=json.loads((ROOT/'configs/groundjev/global_identity_v1.json').read_text());source=Path(plan['immutable_source_root'])
    scopes=json.loads((source/plan['evaluation_plan']).read_text())['scopes'];table=json.loads(a.table.read_text())
    result=dict(status='POSTHOC_OFFICIAL_TABLE_DIAGNOSTICS_RUNNING',source_table=str(a.table),completed={},full_goal_complete=False)
    assert not a.output.exists()
    for name,record in table['completed'].items():
        if a.only_before and not name.endswith('_before'):continue
        scope=next(s for s in scopes if name.startswith(s+'_') or name.endswith('_'+s))
        manifest=json.loads((source/scopes[scope]['manifest']).read_text())
        diagnostic=analyze(manifest,Path(record['run']))
        result['completed'][name]=dict(official_metrics=record.get('official_metrics',record.get('metrics')),diagnostics=diagnostic)
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=name,counts=diagnostic['counts'])),flush=True)
    result['status']='POSTHOC_OFFICIAL_TABLE_DIAGNOSTICS_COMPLETE';a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
