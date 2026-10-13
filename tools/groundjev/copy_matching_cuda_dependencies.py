"""Reuse only ABI-neutral NVIDIA wheels that exactly match the new Torch wheel."""
from email import message_from_string
from importlib import metadata
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path('/data2/user/LocateMOT_GroundJEV')
WHEEL = ROOT/'.download-cache/wheelhouse/torch-2.9.1-cp312-cp312-manylinux_2_28_x86_64.whl'
TARGET = ROOT/'external/GroundAnything/.venv-serve/lib/python3.12/site-packages'


def main():
    with zipfile.ZipFile(WHEEL) as archive:
        filename = next(p for p in archive.namelist() if p.endswith('.dist-info/METADATA'))
        info = message_from_string(archive.read(filename).decode())
    expected = [r.split(';',1)[0].strip().split('==') for r in info.get_all('Requires-Dist') if r.startswith('nvidia-')]
    copied,pending = [],[]
    for name,version in expected:
        distribution = metadata.distribution(name)
        if distribution.version != version:
            pending.append({'name':name,'required':version,'available':distribution.version})
            continue
        if 'Tag: py3-none-' not in distribution.read_text('WHEEL'):
            raise RuntimeError('Do not copy CPython-specific modules across interpreter versions')
        base = Path(distribution.locate_file('')).resolve()
        count,size = 0,0
        for item in distribution.files:
            source = Path(distribution.locate_file(item)).resolve()
            if not source.is_relative_to(base) or not source.is_file():
                continue
            destination = TARGET/source.relative_to(base)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            count += 1
            size += source.stat().st_size
        copied.append({'name':name,'version':version,'files':count,'bytes':size,'wheel_python_tag':'py3-none'})
    result = {'status':'EXACT_VERSION_PY3_NONE_CUDA_DEPENDENCIES_COPIED','copied':copied,'pending':pending,
              'source_modified':False,'target':str(TARGET),'torch_wheel':str(WHEEL)}
    (ROOT/'outputs/groundjev/model_setup/ground_cuda_snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
