# GroundJEV resource gate audit — 2026-10-20

The registered OneJev resume queue requires exclusive physical GPUs 4/5/6/7 and at least 32768 MiB free on each. This read-only snapshot shows that the gate is not yet satisfied because compute PIDs are present on the requested devices. Existing jobs were preserved and no process was stopped.

- GPU4: 58549 MiB free; compute PIDs 2594782 (ljh/.venv/bin/python)
- GPU5: 61833 MiB free; compute PIDs 1799469 (lyq//data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python)
- GPU6: 64919 MiB free; compute PIDs 4004560 (ljh/python)
- GPU7: 59370 MiB free; compute PIDs 4008192 (ljh/python), 4058728 (lyq/python3)

This is a resource gate snapshot, not a training or metric result. The queue remains waiting rather than sharing devices and risking interference.
