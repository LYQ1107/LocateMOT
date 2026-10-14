"""Stop every member of a service group created with start_new_session=True."""
import os
from pathlib import Path
import signal
import time

ROOT=Path(__file__).resolve().parents[2]


def members(group):
    live=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group and fields[0]!='Z':
                assert (proc/'cwd').resolve().is_relative_to(ROOT), f'Unexpected member in owned service group: {proc.name}'
                live.append(int(proc.name))
        except (FileNotFoundError,PermissionError):pass
    return live


def stop_owned_service(server):
    group=server.pid
    if members(group):
        os.killpg(group,signal.SIGTERM)
        deadline=time.monotonic()+20
        while members(group) and time.monotonic()<deadline:
            time.sleep(.2)
        if members(group):
            os.killpg(group,signal.SIGKILL)
            deadline=time.monotonic()+5
            while members(group) and time.monotonic()<deadline:
                time.sleep(.1)
    assert not members(group), f'Owned native service group still alive: {group}'
    server.wait(timeout=2)
