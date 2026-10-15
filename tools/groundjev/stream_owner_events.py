"""Read-only inotify/pidfd progress stream; no periodic job/file polling."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
IN_CLOSE_WRITE, IN_MOVED_TO, IN_CREATE = 8, 128, 256


def summary(data):
    completed = data.get('completed',data.get('completed_controls',{}))
    detail = {}
    for name,record in completed.items():
        detail[name] = {k:record[k] for k in
            ('metrics','complete_all9','missing_tasks','full_native_row_count','controls','steps') if k in record}
        if 'official' in record:detail[name]['metrics'] = record['official']['metrics']
    return {'status':data.get('status'),'steps':data.get('steps',data.get('step')),
        'target_steps':data.get('target_steps'),'completed':detail,
        'completed_training':list(data.get('completed_training',{})),
        'failed':data.get('failed'),'error':data.get('error')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    log = (args.out/'events.jsonl').open('w',buffering=1)
    paths = {name:ROOT/path for name,path in plan['states'].items()}
    last,watched,pidfds = {},{},{}
    libc = ctypes.CDLL(None,use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd < 0:raise OSError(ctypes.get_errno(),'inotify_init1')
    poll = select.poll()
    poll.register(fd,select.POLLIN)

    def emit(event,**fields):
        value = {'event':event,'observed_utc_unix':time.time(),**fields}
        text = json.dumps(value,ensure_ascii=False)
        log.write(text+'\n')
        print(text,flush=True)

    def read(name):
        path = paths[name]
        if not path.is_file():return
        try:data = json.loads(path.read_text())
        except json.JSONDecodeError:
            # Some existing writers use write_text rather than atomic replace;
            # defer an in-progress write until its next close event.
            return
        current = summary(data)
        if current != last.get(name):
            last[name] = current
            emit('state_change',name=name,source=str(path),**current)

    def register_directories():
        for path in paths.values():
            parent = path.parent
            while not parent.is_dir():parent = parent.parent
            if str(parent) in watched.values():continue
            watch = libc.inotify_add_watch(fd,os.fsencode(parent),IN_CLOSE_WRITE|IN_MOVED_TO|IN_CREATE)
            if watch < 0:raise OSError(ctypes.get_errno(),'inotify_add_watch',str(parent))
            watched[watch] = str(parent)

    register_directories()
    for name,record in plan['owners'].items():
        pid,start = record['identity']
        proc = Path('/proc')/str(pid)
        try:
            fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
            if fields[19] != start or fields[0] == 'Z':raise ProcessLookupError
            assert (proc/'cwd').resolve() == ROOT
            assert record['command'] in (proc/'cmdline').read_bytes().decode().split('\0')
            pidfd = os.pidfd_open(pid)
        except (FileNotFoundError,ProcessLookupError):
            emit('owner_already_exited',name=name,identity=record['identity'])
            continue
        pidfds[pidfd] = name
        poll.register(pidfd,select.POLLIN)
        emit('owner_registered_live',name=name,identity=record['identity'])
    for name in paths:read(name)
    emit('observer_ready',pid=os.getpid(),live_owners=list(pidfds.values()),periodic_polling=False)
    try:
        while pidfds:
            for ready,_ in poll.poll():
                if ready == fd:
                    data = os.read(fd,65536)
                    offset,changed = 0,set()
                    while offset < len(data):
                        watch,mask,cookie,length = struct.unpack_from('iIII',data,offset)
                        offset += 16
                        filename = data[offset:offset+length].split(b'\0',1)[0].decode()
                        offset += length
                        parent = watched[watch]
                        for name,path in paths.items():
                            if str(path.parent) == parent and path.name == filename:changed.add(name)
                    # Directory creation may make a future child-state path
                    # available. Register it and take its first real snapshot.
                    register_directories()
                    for name in paths:
                        if name in changed or name not in last:read(name)
                else:
                    name = pidfds.pop(ready)
                    poll.unregister(ready)
                    os.close(ready)
                    read(name)
                    emit('owner_exited',name=name,last_saved_status=last.get(name,{}).get('status'),
                         note='Read the original blocking command exit code and validate its outputs before claiming success.')
        emit('all_registered_owners_exited',scientific_goal_complete=False)
    finally:
        for pidfd in pidfds:os.close(pidfd)
        os.close(fd)
        log.close()


if __name__ == '__main__':
    main()
