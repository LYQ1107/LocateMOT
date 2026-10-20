# Real MOT/RMOT/OVMOT OneJev source v2

The first three-task manifest rendered 131,990 rows but dropped 2,099 high-candidate MOT identity rows at the 32,768-token limit. The native 1.9GB JSONL is immutable; it is not rewritten and its SHA remains `b60d029b28a1d5bb101361ea548cf07e48b41033ccfcbe20d78f32ccfbe20d78f32cc7f0ba7b4`.

The v2 manifest is:

`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/real_three_task_joint_v2_compact/train.sources.json`

Manifest SHA-256: `cbdafce9e5a587c00b053ee58ea6435db4221d38036ee7574fe8a3f0f854e54c`.

It explicitly applies `compact_competition_v1` to native MOT identity rows while loading them. Candidate IDs, target labels, media paths, and candidate box geometry are unchanged. The two dense competition matrices are removed from the text state and replaced by every positive-IoU edge with its native rounded IoU and center-distance values. Zero-IoU geometry is derivable from the exact retained boxes. The v1 source and manifest remain available for replay.

The registered CPU plan used the same model, renderer, media root, seed and 32,768-token limit:

- 131,990 rows rendered; 0 over-length rows.
- p50/p90/p99/max: 7,939 / 12,173 / 16,050 / 28,156 tokens.
- 1,059,541,548 total tokens; 131,990 micro-batches; 8,250 planned steps for world size 4.
- No GPU, optimizer update, checkpoint or metric was produced by this preflight.

The v2 owner is `scripts/groundjev/queue_real_three_task_joint_v2_compact.py`, waiting on the existing G2/G1/bundle/held-out gates and exclusive GPUs 4/5/6/7. It reports full source exposure only after the actual training receipt contains all 131,990 examples. OVMOT remains the registered Base teacher-control component; native TAO category training rows are still pending.

Rebuild the manifest with `scripts/groundjev/prepare_real_three_task_joint_v2_compact.py`.
