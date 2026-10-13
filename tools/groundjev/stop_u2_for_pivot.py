"""Stop the verified LocateMOT U2 process tree for the user's research pivot."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/data2/user/LocateMOT')
OUT = ROOT / 'outputs/groundjev/resource_cleanup'
NAMES = ('build_u2_candidate_cache.py', 'queue_u2_ordinary_baseline.py',
         'queue_u2_mot_only.py', 'queue_u2_joint.py', 'queue_u2_finalize.py',
         'run_u2_perception_adaptation.py', 'train_u2_mot_only.py', 'train_u2_joint.py')


def processes():
    rows = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            rows[int(entry.name)] = dict(pid=int(entry.name), ppid=int(stat[1]),
                pgid=int(stat[2]), start_ticks=int(stat[19]), state=stat[0],
                argv=(entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],
                cwd=str((entry / 'cwd').resolve()))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total',
                                    '--format=csv'], text=True)


def same_alive(row):
    now = processes().get(row['pid'])
    return now and now['start_ticks'] == row['start_ticks'] and now['state'] != 'Z'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = processes()
    owners = json.loads((ROOT / 'outputs/unified/u2/recovery_20261012/owners.json').read_text())
    supervisor = rows.get(owners['supervisor_pid'])
    direct = {pid for pid, row in rows.items()
              if row['cwd'] == str(ROOT) and any(Path(arg).name in NAMES for arg in row['argv'])
              and row['argv'] and 'python' in Path(row['argv'][0]).name}
    targets = set(direct)
    if supervisor and supervisor['cwd'] == str(ROOT) and any(rows[p]['ppid'] == supervisor['pid'] for p in direct):
        targets.add(supervisor['pid'])
    while True:
        descendants = {pid for pid, row in rows.items() if row['ppid'] in targets}
        added = descendants - targets
        if not added:
            break
        targets.update(added)
    before = {'status': 'VERIFIED_BEFORE_STOP', 'gpu': gpu(),
              'targets': [rows[p] for p in sorted(targets)], 'timestamp_unix': time.time()}
    (OUT / 'process_before.json').write_text(json.dumps(before, indent=2) + '\n')
    print(json.dumps({'verified_pids': sorted(targets), 'count': len(targets)}), flush=True)
    priority = {'queue_u2_finalize.py': 1, 'queue_u2_joint.py': 2, 'queue_u2_mot_only.py': 3,
                'queue_u2_ordinary_baseline.py': 4, 'build_u2_candidate_cache.py': 5}
    order = sorted(targets, key=lambda p: (0 if supervisor and p == supervisor['pid'] else
                   min((priority.get(Path(a).name, 6) for a in rows[p]['argv']), default=6), p))
    sent = []
    for pid in order:
        if same_alive(rows[pid]):
            os.kill(pid, signal.SIGTERM)
            sent.append({'pid': pid, 'signal': 'SIGTERM'})
    deadline = time.monotonic() + 30
    while any(same_alive(rows[p]) for p in targets) and time.monotonic() < deadline:
        time.sleep(1)
    for pid in order:
        if same_alive(rows[pid]):
            os.kill(pid, signal.SIGKILL)
            sent.append({'pid': pid, 'signal': 'SIGKILL'})
    time.sleep(2)
    after_rows = processes()
    remaining = [p for p in targets if same_alive(rows[p])]
    newly_started = [row for row in after_rows.values() if row['cwd'] == str(ROOT)
                     and row['argv'] and 'python' in Path(row['argv'][0]).name
                     and any(Path(a).name in NAMES for a in row['argv']) and row['state'] != 'Z']
    result = {'status': 'STOPPED_BY_USER_RESEARCH_PIVOT', 'signals': sent,
              'remaining_verified_pids': remaining, 'remaining_u2_processes': newly_started,
              'gpu': gpu(), 'timestamp_unix': time.time(), 'checkpoint_or_data_deleted': False}
    (OUT / 'process_stop.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if remaining or newly_started:
        raise RuntimeError('Old U2 process shutdown incomplete')


if __name__ == '__main__':
    main()
