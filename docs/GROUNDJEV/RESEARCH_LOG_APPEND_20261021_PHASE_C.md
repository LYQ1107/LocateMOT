# Research log append — Phase C memory input contract

The isolated single-backbone branch now has an opt-in
`inject_identity_memory` contract. Projected causal identity tokens are
inserted before the current-answer index, while `memory_enabled=False`
returns the original prompt tensors exactly. Eight focused CPU tests passed,
including projector gradients, insertion order and mask length. This is an
engineering input contract only; the GroundAnything decoder hook, real GPU
C0/C1/C2 replay, generated-box recall and HOTA/IDF1 effects remain pending.
