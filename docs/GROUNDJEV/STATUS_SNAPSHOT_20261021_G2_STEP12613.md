# GroundJEV runtime snapshot — G2 step 12613

Recorded from the authoritative formal event log at approximately
`2026-10-21 05:11 +08:00`.

- Formal run: `G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final`.
- Progress: `12613/18227` events, with finite loss `0.2061017351`.
- Durable checkpoints: `adapter-step012500.pt` and
  `latest_training_state.pt` are present; the latest complete checkpoint is
  still step 12500.
- Terminal state: `final_receipt.json` is absent. Therefore G2 is still
  running and no downstream gate is open.
- Resources: the protected two-rank run remains on physical GPUs 3 and 5.
  The registered G1, bundle-export, held-out, native TAO and YesTrack owners
  remain dependency/resource waiters; no optimizer update is claimed for them.
- Scope boundary: this is a progress snapshot only. It is not a detector,
  MOT, RMOT or OVMOT metric result and does not promote step 12613 to a final
  model.

The preceding public snapshot was step 12610. This file records the newer
event without changing the training contract, checkpoint policy or any
historical G2-PL result.
