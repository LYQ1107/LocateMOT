# Stage L8 — Literature and Code Audit

Date: 2026-08-14

All items below were independently verified by opening the linked page or
cloning/reading the linked repository. No paper/GitHub is cited from chat
memory alone.

## 1. Verified references

### RMOT — "Referring Multi-Object Tracking" (Wu et al., CVPR 2023)

- Paper: arXiv:2303.03366
- Project page: https://referringmot.github.io/
- Official code: https://github.com/wudongming97/RMOT
  - Local clone: `LocateMOT_reference_repos/rmot_official`
  - Commit: `d4fedb35538e79a743ff78ff946abc6c84453cab`
  - License: MIT
- Relevant mechanism: RMOT task definition; Refer-KITTI benchmark;
  TransRMOT (language-conditioned DETR tracker); TrackEval RMOT runner.
- What we borrow conceptually: evaluation protocol (seqmap `video+expr`,
  `gt.txt` / `predict.txt` layout, HOTA threshold 0.5). We do not copy
  TransRMOT model code.

### iKUN — "Speak to Trackers without Retraining" (Du et al., CVPR 2024)

- Paper: arXiv:2312.16245
- Official code: https://github.com/dyhBUPT/iKUN
  - Local clone: `LocateMOT_reference_repos/iKUN`
  - Commit: `4db56bfaec703590e0fdfd1684d9769467a67e05`
  - License: MIT
- Relevant mechanism: tracking-to-referring decoupling — first track all
  candidates with an off-the-shelf tracker, then score each trajectory
  against the expression; KUM (text-guided visual fusion: cascade
  attention / cross-correlation / text-first modulation); Refer-Dance
  dataset construction; published Refer-Dance results.
- What we borrow conceptually: "language decides WHAT, tracker decides
  HOW" and the two-stream CLIP similarity scoring for RMOT queries.
- What we do NOT copy: iKUN's KUM/NeuralSORT code; in our pipeline the
  "tracker" is the shared learned UIDM, and the referring signal enters the
  same observation token space (not a post-hoc insertable module).

### TempRMOT — "Bootstrapping Referring Multi-Object Tracking" (arXiv 2406.05039)

- Official code: https://github.com/zyn213/TempRMOT
  - Local clone: `LocateMOT_reference_repos/temp_rmot`
  - Commit: `6a65640d849fdee4a32bb055945ee34c3b0edeb1`
  - License: not detected in repo root (recorded; will confirm before any reuse)
- Relevance: modern end-to-end RMOT (bootstrapping labels from a frozen
  tracker); confirms RMOT remains an active, officially implemented area.
- Not copied; only audited for novelty/protocol context.

### MOTIP / MOTIP-2 — "Multiple Object Tracking as ID Prediction" (CVPR 2025)

- Paper: arXiv:2403.16848
- Official code: https://github.com/MCG-NJU/MOTIP
  - Local clone: `LocateMOT_reference_repos/motip`
  - Commit: `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
  - License: Apache-2.0
- Relevance: sequence-local identity prediction with explicit NEW/NO_MATCH;
  already audited in Stage L6 (same commit) and used as scientific
  inspiration for UIDM's causal identity transitions. No code copied.

## 2. What the audit found about the "unified MOT" claim

- We did not find a publicly released, verifiable system that evaluates one
  shared checkpoint on ordinary MOT + OVMOT + RMOT with the same learned
  identity-dynamics core and lifecycle. L7's TAO OVMOT result and L6's
  multi-domain ordinary MOT result are our own prior stages.
- The RMOT literature (TransRMOT, iKUN, TempRMOT) is task-specific
  (language-conditioned trackers or insertable referring modules), not a
  unified three-formulation identity core.
- Therefore the L8 claim remains: *one learned identity-dynamics process
  shared across closed-set category, open-vocabulary category, and
  referring-expression target specification, with a unified observation
  token*. This is a formulation/architecture claim, not a claim of being
  the first MOT system.

## 3. Design choices supported by the audit

1. CLIP ViT-B/32 as the frozen open-vocabulary encoder for both candidate
   crops and referring text (same semantic space as L7 OVMOT; avoids a
   third task-specific semantic space). iKUN's two-stream CLIP similarity
   supports this choice for RMOT.
2. Identity evidence remains the LocateAnything PBD token (L6 positive
   signal; L7 showed replacing it with CLIP regresses ordinary MOT).
3. A learned gated fusion (identity-residual) produces the unified
   observation token consumed by the same UIDM, so RMOT is not a
   post-hoc module: language enters the shared observation space, while a
   separate lightweight relevance head performs the WHAT decision.
4. Evaluation strictly follows the official RMOT TrackEval runner with
   the documented local patch (hardcoded image path only).

