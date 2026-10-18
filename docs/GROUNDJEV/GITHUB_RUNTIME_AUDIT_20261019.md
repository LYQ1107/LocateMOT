# GroundJEV GitHub and runtime audit — 2026-10-19

**Status:** `IMPLEMENTED` / `RUNNING` / `WAITING_DEPENDENCY`; no new
scientific result is claimed.

The project identity was confirmed as `/data2/user/LocateMOT`. This audit was
performed from the independent YesTrack worktree
`/data2/user/LocateMOT_YesTrack`, which contains the YesTrack preparation
changes and does not own any active GPU process. `git fetch --all --prune`
completed in this worktree. The root worktree's remote-tracking ref had a
concurrent stale lock, so its current remote head was checked from the clean
YesTrack worktree rather than force-updating a shared ref.

## Remote references

| Ref | Audited value | Interpretation |
| --- | --- | --- |
| `origin/research/groundanything-g2-multidomain-20261018` | `a3fc905ee4f1450dc155dc40f84d86030a8e079f` | latest clean G2 lineage used as the YesTrack base |
| `origin/codex/u2-shared-persistent-track-core-20261001` | `2e07baa26bb8df85e6688c9538d49ee17512a60b` | latest root evidence/log branch |
| local YesTrack branch | `4e29143f8b4ddcf323d98d81bc54fe25b2ada347` | two local Stage-0 commits ahead of the clean G2 base; ready for the next GitHub push |
| official YesTrack | `ed42b8315b28c94dc595d83b0d49bf57002bb281` | read-only audit source |

The previously named J1/G2 branch heads in the task note were historical and
were not used to overwrite newer work. The local active training worktrees
were read separately; no active code, checkpoint, optimizer, sampler, or
owner state was changed.

## Server state observed once

| Component | State | Consequence |
| --- | --- | --- |
| Clean G2 `G2_CLEAN_FULL_COVERAGE_V2` | `RUNNING`; latest JSONL event step `145/18227`, two ranks on GPU 3/5; terminal receipt absent | `RUNNING`; do not restart or change loaded code. The active process still uses the pre-v2 event schema, so its historical `samples_global=16` is audited read-only as 32. |
| J1 `trackjev_3task_J1_v4_birth` | `RUNNING` on GPU 4/6; durable state step 700; latest progress log had step 720/1200; no final receipt | `RUNNING`; step 700 is the latest resumable checkpoint, not a final result. |
| J1 held-out gate | owner state `WAITING_J1_FINAL`, registered wait PID still matches J1 command | `WAITING_DEPENDENCY`; the repaired gate is not duplicated and does not stop J1. |
| Fresh GPU5 G1 queue | `WAITING_G2_CLEAN_SUCCESS` | `WAITING_DEPENDENCY`; GPU5 is intentionally reserved behind clean G2, not idle due to an error. |
| G2 bundle exporter | `WAITING_G2_AND_G1_TERMINAL` | `WAITING_DEPENDENCY`; no clean bundle or G2 quality evaluation exists. |
| MOT-only queue | no matching live process | stopped by the prior controlled pivot; it was not restarted. |
| G0 MOT20-05 native export | terminal TRAIN-scope reference exists with 1,658 frames, all input frames accounted for, 224 rejected perception frames | `IMPLEMENTED` reference only; it is not held-out and does not validate J1/G2. |

## Accounting and evaluation contracts

`tools/groundjev/g2/train_g2_dlm.py` now defines
`g2-training-event-v2-global-accounting`: loss is averaged across ranks while
effective tokens and samples remain all-reduced sums. The active old process
was not restarted, and `audit_global_event_accounting.py` never writes its
JSONL. The clean evaluation preflight rejects historical G2-PL/formal-1200
names, validates the 569-tensor bundle and clean manifest SHA, rejects GT or
future-frame access, and preserves rejected frames in the denominator.

The G0 report's historical `183/1658` claim is `11.04%` when the denominator
is the full fixed manifest. The current authoritative G0 result records 224
rejected frames; both values remain in their original receipts and must not
be silently substituted.

## YesTrack stage state

The branch contains CPU-only Yes/No request/result adapters, selective
verification policy, causal semantic cache, candidate corpus preparers,
corpus/evaluation validators, the G2 accounting and J1 gate tests, and the
official implementation audit. The default policy is disabled, so existing
MOT/RMOT/OVMOT behavior is unchanged. No GPU Yes/No training or full-video
YesTrack comparison was started while J1/G2 are active.
