# G1 GPU5 waiter runtime-path repair — 2026-10-21

The G1 owner was waiting safely, but its future launch command resolved the
model, GroundAnything source and Python 3.12 environment under the code
checkout `/data2/user/LocateMOT`. Those artifacts are stored in the runtime
checkout `/data2/user/LocateMOT_GroundJEV`; the old paths did not exist.

The queue now runs G1 training, projector export and isolated controls from
the runtime checkout while retaining the code checkout for the registered
plan and data/protocol files. The runtime model, external GroundAnything tree,
and `.venv-ground-train-py312` were verified before replacement.

The old wait-only owner PID 579251 was terminated only after its state was
verified as `WAITING_G2_CLEAN_SUCCESS`. A new wait-only owner PID 631075 now
records the same status, physical GPU5, zero optimizer updates, and no GPU
launch. G2 Clean and all historical G0/G200/G1 artifacts were untouched.
