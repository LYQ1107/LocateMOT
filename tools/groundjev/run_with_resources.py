"""Launch a GroundJEV job only on the common four-device allowlist."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/groundjev/resources.json').read_text())
    if not args.command or len(set(args.gpus)) != len(args.gpus):
        parser.error('provide a command and distinct physical devices')
    if len(args.gpus) > config['max_physical_gpus'] or not set(args.gpus) <= set(config['physical_gpu_allowlist']):
        parser.error('all GroundJEV jobs combined must remain within the registered four GPUs')
    command = args.command[1:] if args.command[0] == '--' else args.command
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    env['GROUNDJEV_FEATURE_CACHE_BUDGET_BYTES'] = str(config['feature_cache_budget_bytes'])
    env['PYTHONUNBUFFERED'] = '1'
    print(json.dumps({'physical_gpus': args.gpus, 'command': command}), flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
