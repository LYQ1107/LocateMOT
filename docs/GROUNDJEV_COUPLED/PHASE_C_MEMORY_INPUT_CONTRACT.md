# Phase C memory-conditioned input contract — 2026-10-21

`locatemot/groundjev/coupled/memory_tokens.py` now exposes
`inject_identity_memory`. It accepts the already prepared GroundAnything
`inputs_embeds` and attention mask, projects only causal identity memories,
and inserts those tokens immediately before the registered current-answer
index. The helper returns the insertion offset so the DLM caller can shift its
loss labels without changing the original G2 label contract.

The disabled path returns the original tensors without cloning. Thus
`memory_enabled=False` is an exact input baseline, rather than a second
approximation of the G2 prompt. Batched memory tokens are shape-checked and
cast to the prompt device/dtype; empty memory is a no-op. No future labels or
future frame features can enter this helper.

The focused CPU contract suite passed three test methods covering eight assertions, including projector
gradient flow, insertion order, mask length, and disabled-path identity. This
is an engineering/input-contract receipt only. It is not yet a GroundAnything
decoder run, box-recall result, HOTA/IDF1 result, or evidence that memory
changes generated boxes. That requires the isolated G2 DLM hook and a real
GPU C0/C1/C2 video replay after the protected G2 course reaches a terminal
checkpoint.
