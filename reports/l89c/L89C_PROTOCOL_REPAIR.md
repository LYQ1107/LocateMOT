# L89C protocol-repair plan

Date: 2026-09-08
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
L89C worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89C`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`

## Scope

L89C is a zero-training correction of the L89 evaluation/deployment
contract. The frozen QSC-D model, all 20 checkpoints, L85 Z1 cache, L89
language cache, L69 bank, labels, loss, tracker and ordinary MOT/OVMOT paths
are unchanged. The only scientific correction is to use the registered
candidate-energy-versus-NULL emission rule from `tools/l88c_eval_metrics.py`:
`candidate_energy >= candidate_threshold`,
`candidate_energy - null_logit >= null_margin`, and
`presence_logit >= presence_threshold`.

The old L89 presence-versus-NULL rule fits, fixed semantic values and internal
TrackEval outputs remain historical buggy-deployment evidence. No training,
forward pass, cache rebuild, screening, official-test labels, or production
benchmark is authorized.

## Frozen inputs and outputs

- Base L89 commit: `23f36bc3eefa4e891ff38ba1e7e3bd870b171e0e`.
- Existing raw dev records: `outputs/l89/eval/dev_scores_joint40/score_records.jsonl`.
- Existing L89 checkpoints: `outputs/l89/train/joint40/**` (read-only).
- Fixed manifest SHA256:
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- New code is limited to the five L89 evaluation/deployment files and
  `tools/l89c_boundary_guard.py`.
- New evidence is under `outputs/l89c/` and `reports/l89c/`.

## Execution gates

First compile the affected files, run the boundary guard, one deterministic
emission assertion and static source checks, then commit the correction before
any replay. Reuse the existing 20 checkpoints and 9,960 raw dev records; do
not run the scorer or any training. Refit corrected dev rules, run corrected
dev TrackEval, freeze the corrected checkpoint/rule, then run the fixed
16-calibration/24-validation replay and final internal V1/V2 TrackEval. The
internal scope must remain V1 videos `0004,0018` (86 sequences) and V2 videos
`0016,0017,0020` (537 sequences).

All output must explicitly retain
`zero_training=true`, `screening_gt_used=false`,
`official_test_labels_read=false`, and
`ordinary_mot_ovmot_touched=false`. After the corrected verdict, stop at
`STOPPED_PENDING_SUPERVISOR_REVIEW`; do not launch a new model or benchmark.
