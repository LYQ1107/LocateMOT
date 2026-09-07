# L89 preregistered plan

Date: 2026-09-07  
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`  
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`  
L89 worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`  
Base SHA: `ece9fc5549a9210424eb127fb10430c3fed7b7ba`

## Registered question

L88C corrected internal validation remained below L87-A on both domains and
the fixed semantic result failed recall/multi-positive preservation. L89 tests
one new factor only: replace the independent scalar candidate semantic head
with a query-conditioned candidate-set decoder that performs candidate
self-attention followed by cross-attention to pure frozen language tokens,
then feeds the resulting state into the unchanged L87-A/L86 temporal, prior,
presence and NULL contracts. The hypothesis is that set-aware language
correspondence can preserve complete multi-target bags without changing the
frozen candidate bank or tracker.

This is not a LoRA, detector, proposal, threshold, NULL-only, geometry,
cardinality, or tracker experiment. Token/span-to-region and static/motion
alignment remain `UNALIGNED`.

## Frozen inputs and boundaries

- Frozen L84/L87-A Z1 cache: `outputs/l85/features/fit_dev_eval_full_attempt2`.
- Frozen L69 candidate bank and all row offsets/boxes/track provenance.
- Frozen L48 pure text cache only as lookup/sentence source; L89 will build a
  new pure GroundingDINO/BERT token cache without image fusion or labels.
- L49 V1/V2 fit units only for optimization; fixed 16 calibration and 24
  validation are read only after all model/strategy choices are frozen.
- Fixed manifest SHA256:
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- No screening/official-test labels, external training data, raw/dense cache,
  ordinary MOT/OVMOT/TAO, UIDM, tracker or shared source changes.

## New files and outputs

Only new L89-prefixed source files are permitted: `locatemot/models/l89_*`,
`locatemot/rmot/l89_*`, and `tools/l89_*`. Reports go under `reports/l89/`
and machine outputs under `outputs/l89/`. The boundary guard must prove that
all pre-L89 scientific source files, especially `locatemot/tracking/**`, are
unchanged.

The QSC-D configuration is fixed at semantic dimension/hidden `256`, 2 set
layers, 8 heads, FFN `1024`, dropout `0.10`, history length `8`; no depth,
head, FFN, rank, geometry or loss sweep is permitted. The L89 model uses a
fresh initialization and the unchanged `l87a_loss` implementation.

## Execution gates

1. Implement source boundary guard, pure-language cache, QSC-D, explicit L89
   model, trainer and wrappers. Run only targeted compile/import, the guard,
   and one real fit-group forward/backward smoke. The smoke must prove finite
   output/loss, set-decoder and membership gradients, unchanged candidate
   order, causal history, and the exact L87-A loss key contract.
2. Build the label-free language cache for only L89 fit/dev/fixed/internal
   V1/V2 expressions; cache entries must carry no labels or candidate data.
3. Commit and freeze scientific code before the authorized 40-epoch training.
   Use seed `20260829`, AdamW `2e-4`, weight decay `1e-2`, BF16 autocast with
   FP32 master parameters, gradient clipping `1.0`, curriculum S 1--8,
   T 9--20, J 21--40, and at most four GPUs. Save every even epoch.
4. Score all 20 checkpoints on the legal video-disjoint dev split, shortlist
   at most five under the fixed B/R/P rules, run full-video dev only for those,
   and freeze the final checkpoint/rule before fixed validation.
5. Run the fixed 16-calibration/24-validation semantic diagnostic, then the
   required internal full-video V1 (86 sequences) and V2 (537 sequences)
   TrackEval runs regardless of the semantic gate. Do not read screening or
   official-test labels.

## Fixed semantic gate

Legacy gate thresholds are fixed: recall `>= .7233333`, precision
`>= .0830188679`, FP/frame `<= 11.125`, predictions/positive `<= 4.069`, row
hard violation `<= .8666667`, row multi-positive recall `>= .7894444`, and
inactive false acceptance `< 1.0`. Target-bag metrics are primary diagnostics
and are never mixed with legacy row metrics. A semantic failure does not block
the already-authorized internal V1/V2 TrackEval, but does block any screening,
official-test, or later architecture.

## Pre-registered interpretation

L89 is a learned sequence improvement only if both V1 HOTA `> 28.5752` and V2
HOTA `> 22.1300`. It is a material improvement only if both reach at least
`31.5752` and `25.1300`. These are internal validation comparisons, not the
ultimate public-test target (V1 `45`, V2 `40`, Dance `42+`). After final
internal TrackEval, stop and write a supervisor approval request rather than
adding another architecture or training schedule.

All machine outputs carry:
`ordinary_mot_ovmot_touched=false`, `production_mot_ovmot_changed=false`,
`tracker_code_changed=false`, `uidm_code_changed=false`,
`candidate_bank_changed=false`, `candidate_deletion=false`,
`candidate_truncation=false`, `groundingdino_lora_used=false`,
`groundingdino_trainable=false`, `bert_trainable=false`,
`screening_gt_used=false`, `official_test_labels_read=false`, and
`token_span_region_alignment=UNALIGNED`.
