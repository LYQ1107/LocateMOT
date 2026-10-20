# Phase C causal memory prefill contract — 2026-10-21

The GroundAnything DLM hybrid stream requires `input_ids`, labels, embeddings
and positions to have the same logical sequence.  The new adapter therefore
targets the already-supported causal generation prefill boundary, immediately
before bbox answer tokens are generated.  It inserts causal identity tokens,
extends the attention mask and position IDs, and performs one language prefill
through the existing GroundAnything language tower.  It does not instantiate
OneJev, YesTrack or a second vision encoder.

With `enabled=False`, the adapter returns the original embedding, mask and
position tensors by identity.  A seven-test CPU contract suite passes,
including a fake language tower where the next-token logits change only when a
memory token is inserted.  This verifies the intended causal influence path,
but it is not evidence that the real G2 decoder uses the token.

The real G2 prefill remains pending an exclusive GPU window.  GPU3/5 are
reserved by the formal G2 Clean run, and the other devices have foreign
compute owners; no process was stopped or co-located.  The required next
experiment is a same-image comparison of memory disabled, correct memory and
shuffled memory, reporting bbox-token probabilities, generated boxes and
whether the one-prefill contract is preserved.
