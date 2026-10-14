"""Direct parallel downloads of complete native dense/referring benchmark inputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    plan = json.loads((ROOT/'configs/groundjev/ground_retention_dense_ref9_data.json').read_text())
    out = ROOT/'outputs/groundjev/ground_retention/dense_ref9_data'
    out.mkdir(parents=True, exist_ok=True)
    bundle, data_root = Path(plan['bundle']), Path(plan['data_root'])
    bundle.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(bundle).free > 2*plan['archive_bytes']
    state = {'status': 'DOWNLOADING_ORIGINAL_DENSE_REF9_INPUTS', 'revision': plan['revision'],
        'tasks': plan['tasks'], 'GT_used_for_training': False, 'GPU_usage': 0,
        'direct_without_proxy': True, 'complete_30_benchmark_bundle_claim': False,
        'model_evaluation_performed': False, 'verified_archives': []}
    def save():
        (out/'owner_state.json').write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state), flush=True)
    save()
    base = 'https://huggingface.co/datasets/'+plan['repository']+'/resolve/'+plan['revision']+'/'
    def download(name, expected=None):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if expected:
                assert target.stat().st_size == expected['bytes'] and digest(target) == expected['sha256']
            return target
        partial = target.with_name(target.name+'.part')
        log = out/(name.replace('/', '_')+'.download.log')
        with log.open('a') as output:
            subprocess.run(['curl', '--noproxy', '*', '--fail', '--location', '--silent', '--show-error',
                '--retry', '3', '--retry-delay', '1', '--connect-timeout', '30', '--continue-at', '-',
                '--output', str(partial), base+name], stdout=output, stderr=subprocess.STDOUT, check=True)
        if expected:
            assert partial.stat().st_size == expected['bytes'] and digest(partial) == expected['sha256']
        partial.replace(target)
        return target
    try:
        metadata = ['package_manifest.json', 'BENCHMARKS.json', 'DATA_SOURCES.json', 'LICENSE',
                    'THIRD_PARTY_NOTICES.md', 'SHA256SUMS', 'scripts/prepare_data.py']+plan['annotations']
        with ThreadPoolExecutor(max_workers=3) as pool:
            for _ in pool.map(download, metadata):
                pass
        checksums = {line.split(maxsplit=1)[1].lstrip('*'): line.split(maxsplit=1)[0]
                     for line in (bundle/'SHA256SUMS').read_text().splitlines()}
        for name in metadata:
            if name in checksums:
                assert digest(bundle/name) == checksums[name]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {entry['path']: pool.submit(download, entry['path'], entry)
                       for entry in plan['archive_manifest']}
            for entry in plan['archive_manifest']:
                futures[entry['path']].result()
                state['verified_archives'].append(entry['path']); save()
        module_spec = importlib.util.spec_from_file_location('groundjev_official_eval_data', bundle/'scripts/prepare_data.py')
        module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
        state['status'] = 'EXTRACTING_VERIFIED_DENSE_REF9_ARCHIVES'; save()
        state['extracted'] = {}
        for entry in plan['archive_manifest']:
            written, existing = module.extract_archive(bundle/entry['path'], data_root/entry['extract_to'])
            state['extracted'][entry['path']] = {'written_files': written, 'reused_identical_files': existing}
            save()
        paths = module.make_data_paths(bundle, data_root)
        module.write_config(out/'datasets.local.yaml', paths)
        state.update(status='ORIGINAL_DENSE_REF9_INPUT_DATA_PREPARED', bundle=str(bundle), data_root=str(data_root),
            data_paths=str(out/'datasets.local.yaml'), declared_evaluation_rows=plan['benchmark_rows'],
            original_scorer_input_validation='pending dry-run and actual unmodified benchmark evaluation')
        save()
    except Exception as error:
        state.update(status='ORIGINAL_DENSE_REF9_DATA_PREPARATION_FAILED', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
