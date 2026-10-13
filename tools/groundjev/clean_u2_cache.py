"""Inventory then remove only the stopped epoch-1 U2 feature shards."""
import argparse
import json
import os
from pathlib import Path
import re
import time

ROOT = Path('/data2/user/LocateMOT')
SHA = '1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688'
CACHE = ROOT / 'outputs/unified/u2/perception_cache' / SHA
OUT = Path(__file__).resolve().parents[2] / 'outputs/groundjev/resource_cleanup'


def disk():
    return {str(p): {'used_bytes': (os.statvfs(p).f_blocks - os.statvfs(p).f_bfree) * os.statvfs(p).f_frsize,
                     'available_bytes': os.statvfs(p).f_bavail * os.statvfs(p).f_frsize}
            for p in [Path(f'/data{i}/user') for i in range(4)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete-verified', action='store_true')
    args = parser.parse_args()
    stop = json.loads((OUT / 'process_stop.json').read_text())
    assert not stop['remaining_verified_pids'] and not stop['remaining_u2_processes']
    assert not CACHE.is_symlink() and CACHE.resolve() == CACHE
    spec = json.loads((CACHE / 'ordinary/cache_spec.json').read_text())
    assert spec['foundation_sha256'] == SHA and spec['phase'] == 'ordinary'
    inventory = [json.loads(line) for line in (CACHE.parent / 'frame_inventory.jsonl').read_text().splitlines()]
    ordinary = {g['key']: g for g in inventory if g['dataset'] != 'ovmot'}
    files = []
    for path in sorted(CACHE.rglob('*')):
        if not path.is_file() or not re.fullmatch(r'frames_\d{6}\.(pt|tmp)', path.name):
            continue
        relative = path.relative_to(CACHE)
        key = str(relative.parent)
        assert key in ordinary and not path.is_symlink()
        assert path.resolve().is_relative_to(CACHE) and path.stat().st_dev == CACHE.stat().st_dev
        st = path.stat()
        files.append({'path': str(path), 'resolved': str(path.resolve()), 'foundation_sha256': SHA,
                      'group': key, 'size_bytes': st.st_size, 'allocated_bytes': st.st_blocks * 512,
                      'inode': st.st_ino, 'device': st.st_dev, 'symlink': False, 'crosses_mount': False})
    names = {f['path'] for f in files}
    references = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target in names:
                        references.append({'pid': int(proc.name), 'path': target})
                except (OSError, PermissionError):
                    pass
        except (OSError, PermissionError):
            pass
    assert not references, references
    before = disk()
    plan = {'status': 'VERIFIED_DRY_RUN', 'allowed_root': str(CACHE), 'files': files,
            'file_count': len(files), 'allocated_bytes': sum(f['allocated_bytes'] for f in files),
            'open_process_references': references, 'disk_before': before,
            'retained': 'All non-feature files, manifests, protocols, datasets, checkpoint and completed evaluations'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cache_delete_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({k: v for k, v in plan.items() if k != 'files'}), flush=True)
    if not args.delete_verified:
        return
    for row in files:
        p = Path(row['path'])
        st = p.stat()
        assert st.st_ino == row['inode'] and st.st_dev == row['device'] and st.st_size == row['size_bytes']
        p.unlink()
    after = disk()
    result = {'status': 'CACHE_FEATURES_REMOVED_BY_USER_RESEARCH_PIVOT', 'deleted_files': len(files),
              'deleted_allocated_bytes': plan['allocated_bytes'], 'disk_before': before, 'disk_after': after,
              'freed_bytes_by_volume': {k: before[k]['used_bytes'] - after[k]['used_bytes'] for k in before},
              'remaining_feature_files': sum(1 for p in CACHE.rglob('frames_*.pt')),
              'preserved_checkpoint': str(ROOT / 'outputs/unified/u2/perception_adaptation/training/epoch_1.pth'),
              'timestamp_unix': time.time(), 'recoverable': 'Not retained; rebuild from preserved images and checkpoint'}
    (OUT / 'cache_cleanup_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
