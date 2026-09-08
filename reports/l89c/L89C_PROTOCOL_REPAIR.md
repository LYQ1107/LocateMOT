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

## Completed evidence (2026-09-08)

The implementation and replay are complete. The first fixed-semantic attempt
was preserved at
`outputs/l89c/eval/fixed_semantic_corrected_attempt1/` with its complete CUDA
OOM traceback. Because GPU0 was occupied by unrelated processes, the exact
same evaluator and frozen selection were rerun on CPU in the new authoritative
directory
`outputs/l89c/eval/fixed_semantic_corrected_attempt2/`; this is a resource
execution correction, not a protocol or model change.

The corrected developer selection used 20 existing raw checkpoints, the
corrected `l88c_eval_metrics.fit_rule_set` implementation, five shortlisted
checkpoints, three rules (B/R/P), and the completed 15-result developer
TrackEval matrix. It froze epoch 2, Rule R, before fixed-slice labels were
attached. The selected checkpoint is the immutable L89 checkpoint at
`outputs/l89/train/joint40/checkpoint_l89_epoch002.pt`, SHA256
`f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`.
The frozen corrected thresholds are candidate `-0.75`, presence `-1.0`, and
NULL margin `0.0`.

The fixed replay completed 16 calibration and 24 validation units with 40/40
ordered score records. Validation was not used to choose the checkpoint or
threshold. The corrected validation gate failed: recall `0.8387097`,
precision `0.0869565`, FP/frame `11.375`, predictions/positive `9.6452`,
hard violation `0.6923077`, multi-positive recall `0.8333333`, and inactive
false acceptance `1.0`. Thus the corrected candidate-vs-NULL contract removes
the historical evaluator ambiguity but does not produce deployable emission.

The final internal full-video replay used only this frozen epoch-2/Rule-R
strategy over V1 videos `0004,0018` (86 sequences) and V2 videos
`0016,0017,0020` (537 sequences). Corrected internal TrackEval HOTA is
V1 `2.244718%` and V2 `0.583059%`; these are internal validation-scope
TrackEval results, not screening or official-test results. L89C is therefore
`STOPPED_PENDING_SUPERVISOR_REVIEW` with no new training authorized.
