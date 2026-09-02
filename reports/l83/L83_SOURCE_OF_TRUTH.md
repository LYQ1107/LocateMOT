# L83 source of truth

Date: 2026-09-02
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Base commit: `75e9f9cd0482645c07a9f71ad4419b0c5f57132b`

The machine-readable snapshot is
`outputs/l83/preregister/source_of_truth.json`.  It records the actual file
hashes, mtimes and the intentionally unverified local MMDetection checkout.
The worktree had pre-existing historical changes; this branch does not modify
them and no L82 file is edited.

## Frozen assets

| Asset | Frozen path | Expected/current status |
|---|---|---|
| fast manifest | `outputs/l19/protocol/kitti_fast_eval_manifest.json` | SHA256 `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa` |
| UIDM | `outputs/l11/checkpoints/uidm_l11_main/step11000.pt` | SHA256 `f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343` |
| L69 bank manifest | `outputs/l69/attempt9/budget40_features/kitti/manifest.json` | SHA256 `11169c30f8eacf23a4462c96665a5b31c921fcb2478191f798f0837c318045ce` |
| L48 text cache | `outputs/l48/data/text_cache.pt` | SHA256 recorded by snapshot |
| L81 step100 | `outputs/l81/train/probe500_retry1/checkpoint_l81_step100.pt` | SHA256 `2b6131584f4fe0fe018ee4494d61f481ac8eacb5f7ed7abe1125bc4a37c46915` |
| L82 L59 control | `outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l59_fused_roi_checkpoint_epoch10.pt` | SHA256 `bc957ed71af26716a060395f9bbf2e23dddb41c1e555e02d19f77ece64f2eb2b` |
| L82 L81 control | `outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l81_candidate_evidence_checkpoint_epoch10.pt` | SHA256 `9fca222c37694ac760aa6244408a58648b369e305a45413acd3ec2c7092bb5e9` |
| L82 candidate-reference | `outputs/l82/train/frozen_rank_probe_retry3/checkpoints/l82_candidate_reference_checkpoint_epoch10.pt` | SHA256 `371edd45f5715a46ce839052be67d98705f1021bc9737d8431974fde03a6fcc6` |
| L69 feature contract | `outputs/l69/attempt9/budget40_features/kitti/` | 17 frozen video files; per-file manifest is recorded |

GroundingDINO config, detector weight and local BERT snapshot are read-only
runtime dependencies.  The local MMDetection checkout has no verifiable git
HEAD in this environment; it is recorded as `unverified`, not as the official
v3.3.0 commit.  The fixed manifest is the hard stop if its hash changes.

## Boundary flags

`screening_gt_used=false`; `official_test_labels_read=false`;
`ordinary_mot_ovmot_touched=false`; `hota_trackeval_run=false`;
`candidate_deletion=false`; `candidate_truncation=false`;
`l81_modified=false`; `l82_modified=false`;
`uidm_shared_checkpoint_modified=false`;
`token_span_region_alignment=UNALIGNED`;
`static_motion_alignment=UNALIGNED`.
