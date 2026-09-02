# L83 factorized energy phase

Status: **not run**.

The factorized candidate-prior/query-presence/semantic-interaction model was
conditional on the faithful target-bag gate.  That gate is
`faithful_target_bag_training_gate_fail` for L59, L81, and L82, despite the
implementation and reload contracts passing.  Therefore no factorized code,
training, checkpoint, dev gate, historical 16/24 semantic run, screening, or
TrackEval evidence is claimed in L83.

This is an intentional evidence stop, not a claim that all GroundingDINO
representations are impossible.  The decoder sharpness audit was still
completed as required and is reported separately.
