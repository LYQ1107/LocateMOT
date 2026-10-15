"""Wait for verified surviving native clients; recover aggregation, never relaunch them."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'external/GroundAnything'
sys.path[:0]=[str(ROOT),str(SOURCE)]


def alive(identity):
    try:
        proc=Path('/proc')/str(identity['pid'])
        fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z' or fields[19]!=identity['start_ticks']:return False
        assert (proc/'cwd').resolve().is_relative_to(ROOT)
        return True
    except FileNotFoundError:return False


def main():
    from scripts import evaluate as native
    from tools.groundjev.service_process_group import members,stop_owned_service
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    clients=json.loads((ROOT/'outputs/groundjev/ground_retention/existing_native_clients_20261015.json').read_text())
    specs=[('original','8101','groundjev_original_dense_ref9_queue32_v3_20261014','original_dense_ref9_v3_queue32'),
           ('adapted200','8102','groundjev_trackproj200_dense_ref9_sync_v3_20261015','trackproj200_dense_ref9_v3_sync')]
    roots={name:{'clients':[c for c in clients if ':'+port in c['port_or_url']],
                 'output':SOURCE/'outputs/groundjev_retention'/run,
                 'owner':ROOT/'outputs/groundjev/ground_retention'/owner}
           for name,port,run,owner in specs}
    assert all(r['clients'] and any(alive(c) for c in r['clients']) for r in roots.values())
    state={'status':'WAITING_VERIFIED_EXISTING_NATIVE_RUNNERS_AND_CLIENTS','owner_pid':os.getpid(),
           'new_model_or_evaluation_processes_launched':False,'existing_client_count':len(clients),'completed':{}}
    def save():(args.out/'owner_state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();pending=set(roots)
    while pending:
        for name in tuple(pending):
            item=roots[name]
            if any(alive(c) for c in item['clients']):continue
            run=json.loads((item['output']/'run.json').read_text())
            tasks=run['plan']['tasks']
            result=native.summarize(item['output'],tasks)
            samples={}
            for record in result['records']:
                source=json.loads((item['output']/record['source']).read_text())
                n=source['n-samples'][record['task']]
                assert n['original']==n['effective'] and n['original']>0
                if record['task'] in samples:assert samples[record['task']]==n
                samples[record['task']]=n
            result.update(native_summarizer_unchanged=True,n_samples=samples,
                actual_runner_exit_code_unknown_after_parent_exit=True,
                original_failure_or_summary_overwritten=False,
                completed_all9=not result['missing_tasks'] and len(samples)==9)
            if result['completed_all9']:
                assert sum(n['original'] for n in samples.values())==91124
            (args.out/(name+'_native_summary.json')).write_text(json.dumps(result,indent=2)+'\n')
            parent=json.loads((item['owner']/'owner_state.json').read_text())
            group=parent['owned_server_process_group']
            class ExistingGroup:
                pid=group
                def wait(self,timeout):assert not members(self.pid)
            stop_owned_service(ExistingGroup())
            state['completed'][name]={'complete_all9':result['completed_all9'],'missing_tasks':result['missing_tasks'],
                'full_native_row_count':sum(n['original'] for n in samples.values()),
                'owned_service_group_stopped':group,'all_group_members_gone':not members(group)}
            pending.remove(name);save()
        if pending:time.sleep(20)
    state['status']='EXISTING_NATIVE_RUNNERS_FINISHED_RECOVERED_SUMMARIES_VALIDATED';save()


if __name__=='__main__':main()
