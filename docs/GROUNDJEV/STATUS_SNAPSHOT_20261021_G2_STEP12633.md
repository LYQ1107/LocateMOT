# GroundJEV runtime snapshot — G2 step 12633

Recorded from the authoritative formal event log at `2026-10-21 05:15:19
+08:00`.

- Formal run: `G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final`.
- Progress: `12633/18227` events, finite loss `0.28367656865157187`, finite
  gradient norm `10.239775657653809`, and peak allocated memory `12.8289 GiB`.
- Durable checkpoints: `adapter-step012500.pt` and
  `latest_training_state.pt` remain present; the latest complete numbered
  checkpoint is step 12500.
- Terminal state: `final_receipt.json` is absent. G2 is still running, so no
  dependent training or evaluation gate is open.
- Resources: the protected two-rank run remains on physical GPUs 3 and 5.
  Registered G1, bundle-export, held-out, native TAO and YesTrack owners are
  still alive in their waiting states.
- Scope boundary: this is a progress snapshot only. It is not a detector,
  MOT, RMOT or OVMOT metric result and does not promote step 12633 to a final
  model.

This snapshot supersedes the previous step-12613 runtime snapshot while
preserving that earlier record and all historical G2-PL outputs.
