# TrackJEV runtime preflight — 2026-10-20

The GroundJEV worktree passed `compileall` for `locatemot/groundjev`,
`scripts/groundjev`, and the TrackJEV tools under the registered runtime
`/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python`.

The decision-layer, model, and native OneJev adapter self-tests all passed:

- `TRACKJEV_DECISION_LAYER_SELFTEST_PASS`
- `TRACKJEV_MODEL_SELFTEST_PASS`
- `TRACKJEV_ADAPTER_SELFTEST_PASS`

The checks used the external OneJev source through `PYTHONPATH` and allocated
no GPU. They validate imports and structured decision behavior only; they do
not claim tracking performance or replace the queued real-frame evaluation.
