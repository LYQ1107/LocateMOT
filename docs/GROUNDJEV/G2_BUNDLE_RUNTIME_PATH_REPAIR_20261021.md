# G2 downstream bundle path repair

The G2 bundle owner was waiting correctly, but its future exporter command was
resolved under `/data2/user/LocateMOT`, where the base GroundAnything bundle is
not staged. The owner now uses the dedicated runtime tree:

- Python: `/data2/user/LocateMOT_GroundJEV/.venv-ground-train-py312/bin/python`
- exporter: `/data2/user/LocateMOT_GroundJEV/tools/groundjev/g2/export_g2_checkpoint.py`
- base/output bundles: `/data2/user/LocateMOT_GroundJEV/models`
- working directory: `/data2/user/LocateMOT_GroundJEV`

The old wait-only owner (PID 511425) was replaced by PID 634486 with the same
G2 dependency and output arguments. It remains in
`WAITING_G2_CLEAN_SUCCESS`, has no GPU allocation and has not changed G2.

`py_compile` and a runtime Torch import (`2.10.0+cu128`) passed. This is an
execution-path repair only; G2 has no terminal receipt yet and no performance
claim follows from it.
