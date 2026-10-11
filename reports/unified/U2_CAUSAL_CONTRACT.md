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

A two-process CUDA forward/backward DDP smoke also passed with the same MOT
task on both ranks.  The requested eight-process smoke was not accepted as
evidence: all eight A800 devices were already occupied by an unrelated
process, and the short launch hung during collective initialization before
producing a new result.  It was terminated; no training job was started.
