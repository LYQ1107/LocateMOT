# L88 Runtime Deviations and Preserved Failures

## Approved deviations

| item | registered value | executed value | consequence |
|---|---|---|---|
| adapter precision | BF16 autocast | FP32 | Required by the local MMCV deformable-attention CUDA kernel; finite training completed. |
| formal training | world 4 if available | world 4 | No deviation. |
| internal inference scheduling | free GPUs | GPU0--2 in the final serial schedule | GPU3 was occupied; outputs are complete and no model/protocol change was made. |

The local MMDetection source tree and the local TrackEval checkout do not have
verifiable Git HEADs in this environment. Their paths, versions, and this
limitation are retained in the L88 literature/provenance files; this is not a
claim of an official reproduction.

## Preserved implementation failures

The following directories remain intact and are not semantic evidence of model
failure by themselves:

- `outputs/l88/train/joint40_world4/`: first formal run, causal history-key
  drift at epoch 9;
- `outputs/l88/internal/video_parts_attempt1/`: launcher created nonempty task
  directories before the writer, so the writer correctly refused them;
- `outputs/l88/eval/fixed_semantic_attempt1/`: CPU observation tensors reached
  a CUDA evaluator;
- `outputs/l88/eval/fixed_semantic_attempt2/`: score shape `[1,N]` violated the
  evaluator contract;
- sparse-sequence TrackEval and merge retries documented in the corresponding
  output provenance and commit history.

Each was corrected only at its first actionable implementation point, then
rerun in a new directory. No old asset, checkpoint, bank, GT, or production
entrypoint was overwritten.

