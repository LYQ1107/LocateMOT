"""Copy installed CPython3.10 CUDA framework files into an isolated new environment."""
import json
from importlib import metadata
from pathlib import Path
import shutil
import sys

ROOT = Path('/data2/user/LocateMOT_GroundJEV')
SOURCE = Path(metadata.distribution('torch').locate_file('')).resolve()
TARGET = ROOT/'.venv-onejev-py310/lib/python3.10/site-packages'


def main():
    if sys.version_info[:2]!=(3,10) or not TARGET.is_dir():
        raise RuntimeError('Copy only matching CPython3.10 ABI into the fresh isolated environment')
    names = ['torch','torchvision','triton','cuda-bindings','cuda-pathfinder',
             'numpy','pillow','scipy','sympy','networkx','filelock','typing-extensions','jinja2','fsspec','mpmath','markupsafe']
    names += [req.split('==',1)[0] for req in metadata.requires('torch') if req.startswith('nvidia-')]
    receipts = []
    for name in sorted(set(names)):
        distribution = metadata.distribution(name)
        distribution_root = Path(distribution.locate_file('')).resolve()
        count,size = 0,0
        for item in distribution.files or []:
            source = Path(distribution.locate_file(item)).resolve()
            # Console scripts from the old environment are not copied; launch
            # distributed training with the new python -m torch.distributed.run.
            if not source.is_relative_to(distribution_root) or not source.is_file():
                continue
            relative = source.relative_to(distribution_root)
            destination = TARGET/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            count += 1
            size += source.stat().st_size
        if not count:
            raise RuntimeError(f'No framework files copied: {name}')
        receipts.append({'distribution':name,'version':distribution.version,'source_root':str(distribution_root),'files':count,'bytes':size})
    result = {'status':'LOCAL_FRAMEWORK_SNAPSHOT_COPIED_NOT_MODEL_LOAD','source':str(SOURCE),'target':str(TARGET),
              'source_modified':False,'shared_package_symlinks':False,'distributions':receipts,
              'note':'Only CUDA/framework dependencies copied; no old Transformers/MMDetection or user data'}
    out = ROOT/'outputs/groundjev/model_setup/local_framework_snapshot.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
