# GroundJEV goal completion matrix — 2026-10-20

Overall goal: **not complete**. This matrix is generated from current receipts and queue state; it does not promote preliminary checks into performance claims.

- **g0_mot20_long_export** — `COMPLETE` — G0 registered MOT20-05 validation-half 1657-frame export and official receipt
- **mot_only_long_training** — `COMPLETE_SAFE_NO_LIVE_OWNER` — no live independent MOT-only optimizer process in active GroundJEV scope
- **g2_clean_training** — `RUNNING` — /data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final/training_events.jsonl
- **g1_gpu5_adaptation** — `WAITING_G2_CLEAN_SUCCESS` — /data2/user/LocateMOT_GroundJEV/outputs/groundjev/ground_adaptation/native_tracking200_gpu5_g1_v1/queue_owner_state.json
- **g2_clean_bundle_export** — `WAITING_G2_AND_G1_TERMINAL` — /data2/user/LocateMOT_GroundJEV/outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/owner_state.json
- **independent_heldout** — `WAITING_G2_CLEAN_BUNDLE_AND_GPU7` — /data2/user/LocateMOT_GroundJEV/outputs/groundjev/heldout/g2_clean_independent_gpu7_v1/owner_state.json
- **onejev_joint_mot_rmot_ovmot** — `WAITING_G2_G1_AND_EXCLUSIVE_GPU_SET` — /data2/user/LocateMOT_GroundJEV/outputs/groundjev/training/full_joint_source_epoch1_resume_queue_v2/owner_state.json
- **trackjev_code_and_real_input** — `COMPLETE_PRELIMINARY` — TrackJEV self-tests plus real OneJev/MOT input receipts
- **github_evidence_sync** — `COMPLETE` — codex/groundjev-independent-heldout-complete-20261020

Remaining terminal dependencies:
- G2 final receipt
- G1 GPU5 adaptation and controls
- G2 deployment bundle
- independent heldout official metrics
- OneJev full source joint terminal receipt
