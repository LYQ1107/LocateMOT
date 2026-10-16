"""Apply an allocation-only CUDA cap before the unchanged native SFT entry."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-fraction', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[0] == '--' else args.command
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '7' and 0 < args.memory_fraction < 1
    root = Path(__file__).resolve().parents[2]
    script = (root/command[0]).resolve()
    assert script == root/'tools/groundjev/run_source_manifest_sft.py'
    import torch
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    total = torch.cuda.get_device_properties(0).total_memory
    print(json.dumps(dict(status='NATIVE_SFT_CUDA_ALLOCATION_CAP_SET',
        physical_gpu=7, torch_allocator_fraction=args.memory_fraction,
        torch_allocator_limit_bytes=int(total*args.memory_fraction),
        scope='Allocator cap only, not all-driver-memory guarantee; model/data/loss/optimizer/seed unchanged.')), flush=True)
    sys.argv = [str(script), *command[1:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
