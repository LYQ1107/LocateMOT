# L88 Literature and Code Audit

Date: 2026-09-04.  This is a structural provenance record for the RMOT-only
L88 branch; no outside checkpoint, dataset, or supervision is imported.

## Sources checked

| Source | Primary URL / revision | What was checked | L88 use |
|---|---|---|---|
| MMDetection GroundingDINO | [MMDetection v3.3.0](https://github.com/open-mmlab/mmdetection/tree/v3.3.0), local reference tree `/data1/LWR/vranlee/LLM/mmdetection-3.3.0`, recorded reference `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` | `grounding_dino_layers.py`, `vlfuse_helper.py`, and `grounding_dino.py`; encoder fusion, decoder MHA/deformable attention, and tensor contracts | **Code actually reused through the local runtime contract only**. The L88 adapter does not alter the checkout. |
| FlexHook | [paper/project repository](https://github.com/buptLwz/FlexHook), registered revision `bd1acc...` | Query-conditioned open-vocabulary tracking motivation and hook/fusion design | Structural inspiration only; no code, weights, or data imported. |
| STORM | [Amazon Science repository](https://github.com/amazon-science/storm-referring-multi-object-grounding) | Referring multi-object grounding and temporal/object-level evaluation framing | Structural inspiration only; external training and checkpoint are unavailable and not used. |
| COAL | [arXiv search record](https://arxiv.org/abs/2605.14795) | Recent object-language grounding direction | Paper context only; no verified local implementation or weights are used. |
| DKGTrack | [official repository](https://github.com/acyddl/DKGTrack), registered revision `197f354...` | Open-vocabulary trajectory/knowledge association motivation | Structural inspiration only; no code/data/checkpoint imported. |
| PropVG | [official repository](https://github.com/Dmmm1997/PropVG) | Proposal/visual grounding framing | Structural inspiration only; the frozen L69 proposal bank remains the sole candidate source. |
| ReferDINO | [official repository](https://github.com/iSEE-Laboratory/ReferDINO), registered revision `3cfc...` | GroundingDINO-style referring expression interface | Context only; its weights/data are not available in this branch and are not used. |
| VMRMOT | [paper/project URL](https://arxiv.org/abs/2511.17681) | Vision-motion-reference alignment motivation | Paper inspiration only. No verified motion-language annotations or official usable checkpoint is present. |

The abbreviated revisions above are the revisions recorded by the preceding
supervised task provenance; this work does not claim they are local checkouts or
official reproductions. Where an exact full revision is not present in the
workspace, it is intentionally marked unavailable rather than invented.

## Local model and fairness boundary

The only detector used by L88 is the already verified local GroundingDINO
configuration and checkpoint recorded in `L88_PREREGISTERED_PLAN.md`. The local
MMDetection checkout has no independently verified Git HEAD in this task, so
the source-tree reference above is provenance, not a claim of a clean official
reproduction. BERT is frozen and loaded from the local snapshot. The L69
budget-40 bank, UIDM/history fields, and L86/L87 sidecar are frozen inputs.

L88 actually reuses the local project’s audited data/loss/runtime interfaces,
but adds a new parametrized LoRA injector and an L88 runtime wrapper. It does
not reuse an external implementation or merge adapter weights into the base
detector. No raw text-span-to-region or motion-language annotation is
verified; both remain `UNALIGNED`. Comparison to public RMOT numbers is only
descriptive because L88 uses a frozen candidate bank and internal training/dev
protocol rather than a full official end-to-end reproduction.
