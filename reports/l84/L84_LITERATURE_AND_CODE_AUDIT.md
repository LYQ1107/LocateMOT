# L84/L85 literature and code audit

Date: 2026-09-02.  This is a source/provenance audit, not a claim of an
official reproduction or a benchmark result.

## Directly inspected local implementation

The local MMDetection checkout at
`/data1/LWR/vranlee/LLM/mmdetection-3.3.0` has no verifiable Git HEAD in this
environment.  Its relevant files were read directly:

* `mmdet/models/detectors/grounding_dino.py` and `dino.py` for text prompt
  construction, `pre_decoder`, query embeddings, reference points, and the
  regression branches;
* `mmdet/models/layers/transformer/grounding_dino_layers.py` for
  `GroundingDinoTransformerEncoder`, `SingleScaleBiAttentionBlock` fusion,
  decoder self-attention, text cross-attention, deformable image attention,
  and iterative refinement.

The public reference is [MMDetection](https://github.com/open-mmlab/mmdetection),
tag `v3.3.0`, commit
[`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`](https://github.com/open-mmlab/mmdetection/tree/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a).
L84/L85 reuse only the local audited runtime interface and read-only frozen
weights; no third-party source is copied into production and no external
checkpoint is imported.

## RMOT/trajectory references

* [ReferDINO](https://github.com/iSEE-Laboratory/ReferDINO), commit
  [`3cfc01f57dff97f7d801b1bd54c251e0f34fcef8`](https://github.com/iSEE-Laboratory/ReferDINO/tree/3cfc01f57dff97f7d801b1bd54c251e0f34fcef8).
  The repository and temporal module sources are a structural reference for
  object-consistent temporal enhancement.  L84/L85 do not reuse its RVOS mask
  branch or weights.
* [DKGTrack](https://github.com/acyddl/DKGTrack), commit
  [`197f354443bd1e7b490d204456a7654b7d1e4ccd`](https://github.com/acyddl/DKGTrack/tree/197f354443bd1e7b490d204456a7654b7d1e4ccd).
  Its dataset/training/TrackEval organization is a reference only; the local
  frozen bank and local supervision are not an official DKGTrack reproduction.
* [FlexHook](https://github.com/buptLwz/FlexHook), commit
  [`bd1acc38634b28525d54dc6e0fcb38335f0029f9`](https://github.com/buptLwz/FlexHook/tree/bd1acc38634b28525d54dc6e0fcb38335f0029f9).
  The four-frame/multi-expression batching idea is inspiration; no FlexHook
  weights or tracker output are used.
* [TempRMOT](https://github.com/zyn213/TempRMOT), commit
  [`6a65640d849fdee4a32bb055945ee34c3b0edeb1`](https://github.com/zyn213/TempRMOT/tree/6a65640d849fdee4a32bb055945ee34c3b0edeb1).
  Its public TrackEval layout is a protocol reference.  Published numbers
  will be reported only as literature controls when protocol equivalence is
  documented; no local checkpoint is assumed.
* [STORM paper](https://arxiv.org/abs/2604.10527) and [STORM repository](https://github.com/amazon-science/storm-referring-multi-object-grounding)
  are task-composition/data-design references.  No STORM training code or
  weights are reused unless separately verified.
* [COAL paper](https://arxiv.org/abs/2605.14795) is a paper-level reference
  for semantic injection/counterfactual ideas; no code or weights are claimed.

The local L83/L84 representation and target-bag loss are project code, not
external code.  `token_span_region_alignment=UNALIGNED` and
`static_motion_alignment=UNALIGNED`: the available labels supervise whole
expressions/targets, not verified token-to-region or motion-language links.

## Fairness and missing resources

The L84 probe uses a frozen L69 bank, local GroundingDINO checkpoint, local
BERT snapshot, and identical target-bag metric/loss across representations.
Public repositories may use different detectors, data splits, training
labels, trackers, or hidden protocol settings, so their results are not
directly comparable without the L85 TrackEval protocol audit.  Missing or
unverified items include official third-party checkpoints and a verified Git
revision for the local MMDetection checkout.  No such item is fabricated.
