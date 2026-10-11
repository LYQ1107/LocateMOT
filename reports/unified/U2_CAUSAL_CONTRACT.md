# LocateMOT-U U2 causal contract

`tools/unified/u2_causal_contract.py` passed all required checks and wrote
`outputs/unified/u2/eval/causal_contract.json`:

- birth persists across the next matched frame;
- one-frame occlusion retains the identity and does not recycle its ID;
- changing frame *t+1* does not change frame *t* state;
- arbitrary MOT specification features leave identity logits, new/no-match
  decisions, and memory identical;
- OVMOT class text changes only the semantic residual path while raw identity
  logits and identity memory remain isolated;
- continuous and save/reload online rollouts are identical.

`official_test_labels_read=false` and
`refer_kitti_official_eval_labels_read=false`.

A two-process CUDA forward/backward smoke passed in the original stage. The
original eight-process attempt hung during collective initialization while
devices were occupied and produced no accepted result.

On 2026-10-12, the actual eight-GPU temporal core contract passed using
`tools/unified/u2_ddp_temporal_contract.py`. All ranks bind to distinct devices
0–7 and use the same task on each global step. Two four-frame clips per rank
(global 16) keep the recurrent graph connected throughout each clip. MOT and
OVMOT both produce finite losses/gradients; identity GRU, set interaction and
association receive nonzero gradients, as do semantic GRU/text projection on
OVMOT. After each scratch optimizer update, all model parameters are bit-exact
across the eight ranks. Evidence: `eval/ddp_eight_rank_core.json`.

This is a synthetic contract check with a probe objective and scratch weights.
It does not substitute for the registered 6,000-step learned MOT baseline or
measure tracking accuracy. The shared foundation was not modified, and no
formal track training was launched.
