# Stage L89 final report — query-conditioned set correspondence decoder

Date: 2026-09-07
Project: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
Execution worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`

## Executive decision

L89 completed its registered RMOT-only evidence chain and failed the semantic
gate. The QSC-D head produced a fixed-slice hard-negative improvement and
preserved multi-positive recall, but it failed precision, false-positive
volume, predictions per positive, and inactive/no-match behavior. The frozen
strategy then scored far below L87-A on the final internal V1/V2 TrackEval.
L89 is therefore not ordinary-level RMOT completion and is stopped pending
supervisor review. No new L89 training or architecture was started after the
internal result.

## Evidence chain

| evidence type | authoritative path | result |
|---|---|---|
| pure language cache | `outputs/l89/cache/language_tokens_retry1/` | 3,418 label-free entries, dim 256 |
| contract smoke | `outputs/l89/audit/contract_smoke_retry2/` | finite/reload/positive-negative gradients passed |
| fit | `outputs/l89/train/joint40/` | 40 epochs, 2,640 optimizer steps, finite |
| dev checkpoint scoring | `outputs/l89/eval/dev_scores_joint40/` | 20 checkpoints, 138 groups, 498 records/checkpoint |
| dev selection | `outputs/l89/eval/dev_selection_joint40_retry1/` | epoch 4, Rule R frozen before fixed validation |
| fixed semantic | `outputs/l89/eval/fixed_semantic_retry4/` | `semantic_gate_fail` |
| internal full video | `outputs/l89/eval/fullvideo_internal_selected_retry1/` | V1 86 and V2 537 sequences, complete |
| internal TrackEval | `outputs/l89/eval/trackeval_internal_selected_retry1/` | complete; below L87-A on both domains |
| screening / official test | not run | intentionally forbidden before a passing gate |

## Fixed semantic result

The final frozen epoch-4/Rule-R values were candidate threshold `-1`, presence
threshold `-1`, and null margin `0`, inherited from the legal dev selection.
They were not chosen from validation.

| metric | L29 control | L89 validation | requirement |
|---|---:|---:|---:|
| recall | 0.7333 | 0.8710 | >= 0.7233 — pass |
| precision | 0.0830 | 0.0794 | >= 0.0830 — fail |
| FP/frame | 10.1250 | 13.0417 | <= 11.125 — fail |
| predictions/positive | 8.8333 | 10.9677 | <= 4.069 — fail |
| hard violation | 0.9167 | 0.6923 | <= 0.8667 — pass |
| multi-positive recall | 0.8194 | 0.9028 | >= 0.7894 — pass |
| inactive false acceptance | — | 1.0000 | < 1.0 — fail |
| empty rate | — | 0.0000 | no empty-collapse — pass |

The result is a valid fixed calibration/validation semantic failure, not an
oracle result and not HOTA. V1 was recall `0.9375`, precision `0.0955`,
FP/frame `11.8333`, hard violation `0.7143`; V2 was recall `0.8000`,
precision `0.0656`, FP/frame `14.2500`, hard violation `0.6667`. The V2
volume/precision failure is more pronounced. All 40 rows and all candidate
rows were retained.

## Internal TrackEval result

The final selected strategy was evaluated on internal validation only, with
86 V1 query sequences and 537 V2 query sequences. Percentages below are the
local TrackEval normalized metrics multiplied by 100.

| dataset | L87-A HOTA | L89 HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Refer-KITTI V1 | 28.5752 | 2.2906 | 1.0015 | 5.3123 | 1.1646 | 6.6477 | 1.9903 | 59 |
| Refer-KITTI V2 | 22.1300 | 0.5894 | 0.2297 | 1.5550 | 0.2385 | 5.8315 | 0.4700 | 41 |

This is formal internal TrackEval evidence for the registered validation
scope, but not screening, official-test, or final public benchmark evidence.
The final full-video run scored 76,266 candidate rows and emitted 17,591
rows under the frozen rule. Its post-hoc target-row recall was only `0.0131`
on V1 and `0.0026` on V2, while inactive acceptance stayed `1.0`; this
explains why the fixed-frame ranking movement did not translate to sequence
performance.

## Attribution and limitations

The change tested by L89 was the QSC-D candidate-set/language correspondence
decoder. It used the frozen L84/L87-A Z1 observation representation and
unchanged temporal/prior/presence/NULL loss machinery. The evidence indicates
some local hard-negative ranking signal, but not a calibrated, persistent
frame-level membership signal. The first actionable bottleneck is therefore
query-to-candidate emission/no-match calibration and sequence realization in
this design, not an implementation smoke failure. This report does not claim
that the frozen representation has zero information, and does not claim that
L89 solved correspondence.

There is no verified token/span-to-region or static/motion alignment
supervision; both remain `UNALIGNED`. The local GroundingDINO implementation
has no verifiable Git HEAD. External literature was recorded as structural
context only; no external checkpoint, label, or code was imported.

## Boundary confirmation

All L89 machine outputs carry the relevant negative flags:

```text
screening_gt_used=false
official_test_labels_read=false
ordinary_mot_ovmot_touched=false
candidate_bank_changed=false
candidate_deletion=false
candidate_truncation=false
groundingdino_lora_used=false
groundingdino_trainable=false
bert_trainable=false
token_span_region_alignment=UNALIGNED
static_motion_alignment=UNALIGNED
```

The fixed manifest SHA remains
`06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
Ordinary MOT/OVMOT/TAO, UIDM, old banks/checkpoints, and TrackEval source
were not modified.

## Next action

Stop L89 and request supervisor review of this complete evidence package.
Any subsequent experiment must be separately authorized as one new
single-factor RMOT design; do not extend L89, retune its threshold, add NULL
filtering, read screening labels, or change ordinary MOT/OVMOT.
