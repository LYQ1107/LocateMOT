"""One blocking read-only follower for the original registered training owner."""
import argparse
import json
import math
import os
from pathlib import Path
import select

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--owner',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--after-step',type=int,default=0)
    args=parser.parse_args()
    state_path=args.owner/'owner_state.json'
    state=json.loads(state_path.read_text())
    pid,tick=state['owner_pid'],state['owner_starttick']
    proc=Path('/proc')/str(pid)
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    assert fields[0]!='Z' and fields[19]==tick and (proc/'cwd').resolve()==ROOT
    assert 'tools/groundjev/run_full_joint_source_training.py' in (proc/'cmdline').read_bytes().decode().split('\0')
    descriptor=os.pidfd_open(pid);waiter=select.poll();waiter.register(descriptor,select.POLLIN)
    last_step=args.after_step;phase=None
    try:
        with (args.training/'log.jsonl').open() as log:
            while True:
                position=log.tell();line=log.readline()
                if line.endswith('\n'):
                    row=json.loads(line)
                    if 'loss' in row and row['step']>args.after_step:
                        assert row['step']>last_step
                        assert all(math.isfinite(row[k]) for k in ('loss','ce','brier','grad_norm'))
                        last_step=row['step']
                        print(json.dumps(dict(source='ORIGINAL_FOUR_GPU_TRAINING',**row)),flush=True)
                    elif row.get('event')=='done':
                        print(json.dumps(dict(status='ACTUAL_OPTIMIZER_DONE_SEEN_PARENT_FINAL_VALIDATION_PENDING',record=row)),flush=True)
                        return 0
                    continue
                log.seek(position)
                state=json.loads(state_path.read_text())
                if state['status']!=phase:
                    print(json.dumps(dict(status='FOLLOWING_ORIGINAL_TRAINING_WITHOUT_RESTART',
                        source_phase=state['status'],owner=[pid,tick],last_forwarded_step=last_step)),flush=True)
                    phase=state['status']
                assert state['status']!='FULL_SOURCE_JOINT_OWNER_FAILED',state.get('error')
                if waiter.poll(10000):
                    raise RuntimeError('Original owner ended before a complete optimizer done record; inspect original terminal state')
    finally:os.close(descriptor)


if __name__=='__main__':raise SystemExit(main())
