"""Prefetch the largest pinned official PyPI wheels using parallel byte ranges."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).parent))
from download_models import ranged_download

ROOT = Path(__file__).resolve().parents[2]
SPECS = [('torch','2.9.1'),('nvidia-cudnn-cu12','9.10.2.21'),
         ('nvidia-cudnn-cu12','9.16.0.29'),('nvidia-cublas-cu12','12.8.4.1')]


def fetch(spec):
    name,version = spec
    with urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60) as response:
        info = json.load(response)
    choices = [f for f in info['urls'] if f['filename'].endswith('.whl') and
               (('-py2.py3-none-any.whl' in f['filename'] or '-py3-none-any.whl' in f['filename']) or
                ('x86_64' in f['filename'] and 'manylinux' in f['filename'] and
                 ('-cp312-cp312-' in f['filename'] or '-py3-none-' in f['filename'] or '-cp310-abi3-' in f['filename'])))]
    if len(choices)!=1:
        raise RuntimeError(f'Choose the actual pinned CPython3.12/Linux wheel: {name} {version}')
    entry = choices[0]
    folder = ROOT/'.download-cache/wheelhouse'
    folder.mkdir(parents=True,exist_ok=True)
    logdir = ROOT/'outputs/groundjev/model_setup/wheel_prefetch'
    logdir.mkdir(parents=True,exist_ok=True)
    path = folder/entry['filename']
    print(json.dumps({'state':'OFFICIAL_WHEEL_PREFETCH_STARTED','name':name,'version':version,'bytes':entry['size']}),flush=True)
    if not path.exists() or path.stat().st_size!=entry['size']:
        ranged_download(entry['url'],path,entry['size'],logdir)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8<<20),b''):
            h.update(chunk)
    if h.hexdigest()!=entry['digests']['sha256']:
        raise RuntimeError(f'Official wheel digest differs: {path.name}')
    parts = path.parent/('.'+path.name+'.parts')
    if parts.exists():
        for part in parts.glob('*.part'):
            part.unlink()
        parts.rmdir()
    receipt = {'status':'OFFICIAL_WHEEL_PREFETCH_COMPLETE','name':name,'version':version,
               'path':str(path),'size_bytes':entry['size'],'sha256':h.hexdigest(),'proxy':False}
    (logdir/f'{name}_{version}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packages',nargs='+',help='explicit pinned name==version specs; default largest Torch/CUDA wheels')
    args = parser.parse_args()
    specs = [tuple(x.split('==')) for x in args.packages] if args.packages else SPECS
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch,specs))
