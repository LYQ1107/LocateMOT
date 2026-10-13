"""Download pinned official model bundles directly, retaining original file hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
MODELS = [('OmniJev/OneJev-0.8B', 'c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8', 'OneJev-0.8B'),
          ('GroundingPI/GroundAnything', 'e44e8f1bd518323db31c551aeaa07e370e12d264', 'GroundAnything-DLM')]


def ranged_download(url, path, size, logdir):
    """Parallel official HTTP ranges; reuse completed prefix/parts on resume."""
    chunk_size = 8 << 20
    parts = path.parent / ('.' + path.name + '.parts')
    parts.mkdir(exist_ok=True)
    ranges = [(i, start, min(start+chunk_size, size)) for i,start in enumerate(range(0,size,chunk_size))]
    # Reuse the valid consecutive byte prefix from the prior curl owner.
    if path.exists() and path.stat().st_size < size:
        prefix = path.stat().st_size
        with path.open('rb') as original:
            for index,start,end in ranges:
                part = parts / f'{index:06d}.part'
                if end <= prefix and not part.exists():
                    original.seek(start)
                    part.write_bytes(original.read(end-start))

    def fetch(item):
        index,start,end = item
        part = parts / f'{index:06d}.part'
        if part.exists() and part.stat().st_size == end-start:
            return
        temp = parts / f'{index:06d}.tmp'
        with (logdir / f'{path.parent.name}_part_{index:06d}.log').open('a') as log:
            result = subprocess.run(['curl','--http1.1','--noproxy','*','--fail','--location',
                '--retry','5','--retry-all-errors','--retry-delay','5','--connect-timeout','30',
                '--max-time','300','--range',f'{start}-{end-1}','--output',str(temp),
                '--write-out','%{http_code}',url],stdout=subprocess.PIPE,stderr=log,text=True,check=True)
        if result.stdout.strip() != '206' or temp.stat().st_size != end-start:
            raise RuntimeError(f'Official byte range incomplete: {index} {result.stdout}')
        temp.replace(part)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(fetch,ranges))
    with path.open('wb') as output:
        for index,_,_ in ranges:
            with (parts/f'{index:06d}.part').open('rb') as part:
                shutil.copyfileobj(part,output,8<<20)
    assert path.stat().st_size == size


def download(model):
    repo, revision, name = model
    directory = ROOT / 'models' / name
    directory.mkdir(parents=True, exist_ok=True)
    with urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=60) as response:
        metadata = json.load(response)
    assert metadata['sha'] == revision
    files = [f for f in metadata['siblings'] if '/' not in f['rfilename'] and f['rfilename'] != '.gitattributes']
    required = sum(f.get('size', 0) for f in files)
    if shutil.disk_usage(directory).free < required + (20 << 30):
        raise RuntimeError('Insufficient free disk space for model download')
    logdir = ROOT / 'outputs/groundjev/model_setup'
    logdir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'state': 'MODEL_DOWNLOAD_STARTED', 'repo': repo, 'revision': revision,
                      'bytes': required, 'directory': str(directory)}), flush=True)
    receipts = []
    for entry in files:
        path = directory / entry['rfilename']
        expected = entry.get('lfs', {}).get('sha256')
        if not path.exists() or path.stat().st_size != entry.get('size'):
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{path.name}'
            if entry.get('size',0) > 64 << 20:
                print(json.dumps({'state':'PARALLEL_OFFICIAL_RANGES','repo':repo,'workers':16,'chunk_bytes':8<<20}),flush=True)
                ranged_download(url,path,entry['size'],logdir)
            else:
                with (logdir / f'{name}_{path.name}.download.log').open('a') as log:
                    subprocess.run(['curl', '--http1.1', '--noproxy', '*', '--fail', '--location', '--retry', '5',
                        '--retry-all-errors', '--retry-delay', '5', '--connect-timeout', '30',
                        '--continue-at', '-', '--output', str(path),url], stdout=log, stderr=subprocess.STDOUT, check=True)
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b''):
                h.update(chunk)
        actual = h.hexdigest()
        if expected and actual != expected:
            raise RuntimeError(f'Official weight digest mismatch: {path}')
        partdir = path.parent / ('.'+path.name+'.parts')
        if partdir.exists():
            for part in partdir.glob('*.part'):
                part.unlink()
            partdir.rmdir()
        receipts.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual,
                         'official_lfs_sha256': expected})
    result = {'status': 'DOWNLOAD_COMPLETE_NOT_MODEL_LOAD', 'repo': repo, 'revision': revision,
              'files': receipts, 'proxy': False, 'gpu_used': False, 'large_weight_transport':'16 verified parallel official HTTP byte ranges'}
    (logdir / f'{name}_download.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'state': result['status'], 'repo': repo, 'bytes': sum(x['bytes'] for x in receipts)}), flush=True)
    return result


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, MODELS))
