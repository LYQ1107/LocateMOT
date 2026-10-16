"""Serialize future GPU7 launches around the registered paired training window.

The already-running four-GPU epoch is an explicitly permitted co-resident.
An inherited descriptor lets the window owner's own sequential stages enter.
"""
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT/'outputs/groundjev/resources/gpu7_temporal_training_window.lock'
FD_ENV = 'GROUNDJEV_GPU7_TRAINING_WINDOW_FD'


def acquire_training_window():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    stream = LOCK.open('a+')
    print(json.dumps(dict(status='WAITING_GPU7_TEMPORAL_TRAINING_WINDOW')), flush=True)
    fcntl.flock(stream, fcntl.LOCK_EX)
    return stream


def wait_for_training_window(gpus):
    if 7 not in gpus:
        return None
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if FD_ENV in os.environ:
        descriptor = int(os.environ[FD_ENV])
        held, expected = os.fstat(descriptor), LOCK.stat()
        assert (held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino)
        return None
    stream = LOCK.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(dict(status='WAITING_REGISTERED_GPU7_TEMPORAL_TRAINING_WINDOW',
                              existing_four_GPU_training_unaffected=True)), flush=True)
        fcntl.flock(stream, fcntl.LOCK_SH)
    return stream
