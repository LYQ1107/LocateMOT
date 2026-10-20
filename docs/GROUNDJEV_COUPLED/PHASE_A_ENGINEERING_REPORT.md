# Phase A engineering report

## Real G2 forward/backward gate

Receipt: `/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_a_g2_step11000/phase_a_receipt.json`.

The gate loaded the base GroundAnything bundle plus the protected G2
step-11000 projector, encoded one real MOT20-03 TRAIN image once, pooled two
regions on the official spatial grid, ran the typed readout, and completed a
finite backward pass.  The receipt records:

- model feature dimension: 2,560; spatial layout: 27x36;
- visual forward calls: 1; second VLM loaded: false;
- typed readout trainable parameters: 1,980,425;
- typed readout gradient norm: 15.877484321594238;
- peak allocated GPU memory: 2,930,523,648 bytes;
- no benchmark or quality claim.

The gate proves the intended gradient path into the new readout and the
one-forward provenance contract.  It does not prove identity accuracy, HOTA,
RMOT membership, OVMOT TETA, or end-to-end detector quality.

## CPU contract tests

`tests/groundjev_coupled/test_phase_a_contracts.py` runs with the standard
library `unittest` runner.  It covers finite shared state, dynamic slots,
candidate-local multi-birth assignment, assignment-conditioned memory, causal
timestamps, safe lifecycle termination and typed-readout backward.  The current
run passed all five tests.
