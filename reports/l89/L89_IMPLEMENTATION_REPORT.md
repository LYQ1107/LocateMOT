# L89 implementation report

Date: 2026-09-07
Project root: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`
L89 worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`
Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`
Base commit: `ece9fc5549a9210424eb127fb10430c3fed7b7ba`

## Question and isolated change

L89 tested one new factor after L88C: a query-conditioned candidate-set
decoder (QSC-D) over the frozen L84/L87-A Z1 candidate representation. Each
frame's complete L69 candidate set first receives candidate self-attention,
then cross-attention to pure GroundingDINO/BERT language tokens. The resulting
state is passed through the unchanged L86/L87-A temporal, prior, presence and
NULL machinery. The candidate bank, detector, tracker, UIDM, geometry input,
and ordinary MOT/OVMOT paths were not changed.

Registered QSC-D configuration:

| field | value |
|---|---:|
| semantic/hidden dimension | 256 |
| candidate-set layers | 2 |
| attention heads | 8 |
| FFN dimension | 1024 |
| set dropout | 0.10 |
| causal history length | 8 |
| observation dimension | 1432 |
| trainable parameters | 4,169,829 |

The model signature is explicitly candidate-set based and preserves every row:
`z1 [Q,N,256]`, language tokens `[Q,L,256]` with a true mask, current
observations `[N,1432]`, and causal histories `[N,8,1432]`. No source/pool/
group/query/track identifier is a semantic input. `track_id` is used only by
the frozen data view to assemble causal observations. Token/span-to-region and
static/motion alignment remain `UNALIGNED`.

## Interface and source boundary

The pure language cache contains 3,418 expression entries, 3,418 compact
per-expression tensors, dimension 256, and no labels or candidate fields. The
first cache attempt and the final retry are both preserved; retry1 is the
authoritative complete cache. The local GroundingDINO implementation has no
verifiable upstream Git HEAD. Its harmless load warning about
`language_model...position_ids` is recorded in the cache provenance; it was
not silently treated as a perfect key match.

The contract smoke passed after two preserved implementation attempts:

- group `refer_kitti_v1|0001|5`, complete candidate set `N=53`;
- finite loss `4.35552549` and total gradient norm `52.7609`;
- 90 nonzero parameter gradients, positive-row gradient max `0.69057`,
  negative-row gradient max `0.34210`;
- strict reload maximum absolute difference `0.0`.

All new source is L89-prefixed in the isolated worktree. The boundary guard,
`git diff --check`, and Python compilation are part of the final handoff. No
pre-L89 scientific source, old bank, checkpoint, fixed manifest, tracker, or
ordinary MOT/OVMOT entrypoint was overwritten.

## Preserved failure attempts

The following attempts remain intact and are not semantic results: the initial
language-cache package-path failure, the first contract-smoke package-path
failure, and the missing `candidate_count` contract attempt. The fixed semantic
replay also preserves its wrong-cache-path, scalar-`candidate_prior`, and
command-argument attempts as `INCOMPLETE.md`; only retry4 is authoritative.

## Evidence status

The implementation and fit evidence is valid, but it does not establish
RMOT-level success. The complete evidence chain is recorded in the companion
training, selection, fixed-semantic, and TrackEval reports. Screening,
official-test, and production MOT/OVMOT evaluation were not run.
