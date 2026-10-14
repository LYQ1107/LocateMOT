"""Actual18concurrent HTTP requests to the owned bounded32request native queue."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen, Request
import base64
import argparse
import socket
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.groundjev.service_process_group import stop_owned_service


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=False)
    server=None
    state={'status':'STARTING_NATIVE_QUEUE_REGRESSION','physical_gpu':7,'max_queued_requests':32,
        'max_running_requests':1,'concurrent_clients':18,'GT_read':False,'benchmark_score_claim':False}
    def save():
        (out/'receipt.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    save()
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind(('127.0.0.1',8101))
        with (out/'server.log').open('w') as log:
            server=subprocess.Popen(['python3','tools/groundjev/run_with_resources.py','--gpus','7','--',
                str(ROOT/'external/GroundAnything/.venv-serve/bin/python'),'tools/groundjev/serve_groundanything.py',
                '--port','8101','--max-queued-requests','32','--mem-fraction-static','.4','--out',str(out/'service')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['owned_server_process_group']=server.pid;save()
            deadline=time.monotonic()+600
            while True:
                if server.poll() is not None:raise RuntimeError('Queue-regression native server failed')
                try:
                    with urlopen('http://127.0.0.1:8101/v1/models',timeout=5) as response:
                        if any(m['id']=='groundinganything' for m in json.load(response)['data']):break
                except OSError:pass
                if time.monotonic()>deadline:raise RuntimeError('Queue regression readiness timeout')
                time.sleep(2)
            image=Path(json.loads((ROOT/'outputs/groundjev/pilot_data/mot17_first4.json').read_text())['images'][0])
            payload=base64.b64encode(image.read_bytes()).decode()
            body={'model':'groundinganything','temperature':0,'max_tokens':512,'skip_special_tokens':False,
                'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+payload}},
                    {'type':'text','text':'Locate the person and return its bounding box.'}]}]}
            def send(index):
                request=Request('http://127.0.0.1:8101/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
                with urlopen(request,timeout=600) as reply:
                    response=json.load(reply);assert response['choices']
                    return {'client':index,'HTTP_status':reply.status}
            with ThreadPoolExecutor(max_workers=18) as pool:
                replies=list(pool.map(send,range(18)))
            assert all(r['HTTP_status']==200 for r in replies)
            state.update(status='ACTUAL18_CONCURRENT_REQUESTS_QUEUE32_PASS',replies=replies);save()
    except Exception as error:
        state.update(status='QUEUE32_REGRESSION_FAILED',error=repr(error));save();raise
    finally:
        if server is not None:
            stop_owned_service(server)
        state['owned_regression_service_stopped']=True;save()


if __name__=='__main__':main()
