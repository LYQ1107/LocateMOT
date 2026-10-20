# Compute audit

The Phase-A real-image gate used one A800 selected as CUDA device 0 inside
`CUDA_VISIBLE_DEVICES=7`; the formal G2 owner continued on its protected GPU3 /
GPU5 pair.  The visual bridge has one 428,482,048-parameter native visual
module and reports one encoder call for the frame.  The typed readout adds
1,980,425 trainable parameters in the gate.

The gate's measured visual-plus-readout wall time was 25.65873762499541 s for
initialization, one image forward and one backward.  This is not an FPS claim:
initial model loading dominates it.  A future steady-state profile must report
warm per-frame latency, memory-conditioned overhead, number of visual forwards,
and end-to-end FPS separately.

SGLang contextual hidden-state extraction and identity-conditioned Grounding
are not yet enabled, so no production real-time claim is made.

The bounded Phase-B pilot used the same protected G2 visual path on six native
TRAIN examples and made six visual forward calls. It ran 20 readout-only
AdamW updates in 5.0875 s after model construction, reached 3,072,477,696
bytes peak allocated memory, and saved the readout checkpoint externally.
This is a learnability measurement; it is not a steady-state FPS or tracking
throughput measurement.
