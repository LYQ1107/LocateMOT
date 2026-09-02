# L85 architecture and training contract

## Factorized energy

For every complete current-frame candidate row, the model uses frozen Z1
`[Q,N,256]`, query-independent observation/history features, and a text/frame
summary. The causal history encoder reads at most eight observations ending at
the current frame. The factorization is:

```text
R_static = MLP(Z1)
h_i       = causal history encoder(obs_i[<= current_frame])
C_iq      = MLP([z_i, h_i, z_i * h_i])
g_i       = sigmoid(gate([z_i, h_i]))
R_total   = R_static + g_i * C_iq
A_i       = centered query-independent prior(obs_i, history_i)
B_q       = query/frame presence energy(text_global, frame_global)
S_iq      = A_i + B_q + R_total
NULL_q    = -B_q + learned_bias
```

There is no source/pool/group/query/state/track ID feature, old score,
candidate deletion, top-k, NMS, or tracker logic. Multiple positive rows are
allowed. The current implementation uses the registered compact Z1 path; it
does not claim a detector-side LoRA update unless a separate artifact proves
one.

## Curriculum and loss

The registered curriculum is S=8 epochs single-frame, T=12 epochs clip length
4, and J=20 epochs clip length 4, total 40 epochs. Loss weights are
`1.0 semantic_rank_Rtotal + .30 semantic_rank_Rstatic + 1.0 membership_S +
.50 presence_B + .50 null_rank + .10 temporal`. Present-uncovered units mask
membership negatives; inactive units are explicit no-match examples. Every
positive in a multi-positive target bag participates.

The actual training report must disclose whether the compact implementation
could supply an explicit clip-4 batch or used causal per-row history. It may
not describe a fallback as a full temporal clip experiment.

## Resource and reproducibility contract

Four-GPU DDP is permitted for the registered full run after a single-GPU
finite/memory check. BF16 is enabled only after that check. Query tiles are
chosen label-free from `{8,16,24,32}`. Checkpoints contain only adapter/model,
optimizer/scheduler/scaler/RNG/sampler/epoch and provenance hashes; no full
detector checkpoint is copied. All candidates and native row keys remain in
order. Text/region fine-grained alignment remains `UNALIGNED`.

## Implementation qualification

The registered S/T/J curriculum is represented in the actual compact trainer
as causal per-row history: S exposes only the current observation, while T and
J expose the last four valid observations ending at the current frame. The
model enforces the eight-observation bound and future-frame assertion, but the
trainer does not construct a separate batched clip-4 tensor or a temporal
Transformer window. Therefore the run is reported as a causal-history
curriculum with a temporal auxiliary, not as a full explicit clip-batch
experiment. This is an implementation limitation, not a reason to relabel
the completed result.
