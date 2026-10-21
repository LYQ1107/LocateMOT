# YesTrack native validation v2 — 2026-10-21

## Scope

This run is the first real GPU YesTrack experiment registered for the current
GroundJEV evidence chain. It uses the native GroundAnything candidate rows in
`native_yesno_v2_phaseb.jsonl`, with a fixed `train`/`val` split and no ground
truth in the model prompt. The held-out split is materialized without relabeling
or reading official test labels.

| item | registered value |
|---|---|
| base checkpoint | `onejev_unified600_native_v2/ckpt-step600` |
| train / validation rows | 726 / 390 |
| train corpus SHA-256 | `065337d2fedca2377d02b94e57c9eb3778aeca09aeb1a8b9463429fa15bbee66` |
| validation SHA-256 | `d63a9fb0d54b8a1dc5d12b0960a48cb75c56ab786671f955fb317e196b3033d8` |
| families | OVMOT, RMOT, identity-pair, occlusion-recovery |
| training | 3 epochs, native multimodal LoRA, bfloat16, GPU7 |
| validation | frozen base and fine-tuned adapter on the same 390 rows |
| reported metrics | AUROC, PR-AUC, accuracy, Brier, ECE, false rejection, false acceptance |

The CPU plan completed before queue registration: p50/p90/p99/max rendered
lengths were 1,133/10,000/25,429/26,159, with zero rows over the 32,768-token
limit and 91 planned optimizer steps per epoch (273 over three epochs) for the
registered single-GPU recipe.

## Resource and execution contract

`scripts/groundjev/queue_yestrack_native_v2.py` is the sole owner. It waits
internally until GPU7 has no compute PID and at least 32 GiB free, then launches
real OneJev training through the registered resource wrapper. It never stops or
shares an unrelated process. After training it evaluates the frozen base and
the trained adapter with the same native renderer, media root, validation rows,
and token budget. The owner writes `owner_state.json`, training/evaluation logs,
and two JSON receipts under
`/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/yestrack_native_v2_queue/`.

At registration time GPUs 4–7 were occupied by unrelated or protected compute
clients, so no GPU update was claimed. Until the owner reaches its terminal
receipt, YesTrack has no performance result; existing 24-row readout controls
remain engineering evidence only.

## Scientific boundary

The corpus is a native candidate/identity verification study. It does not by
itself establish tracking HOTA, AssA, IDF1, IDSW, or end-to-end FPS. Those
metrics require the adapter to be connected to a valid physical tracking
baseline on an independent held-out sequence. The queue therefore records the
base-versus-adapter probability metrics first and leaves the tracking claim
open until that later fixed-candidate evaluation is complete.
