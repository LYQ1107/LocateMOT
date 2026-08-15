# Stage L9 — ICLR Novelty Audit

Date: 2026-08-15 (to be finalised with the final results)

## Claim under audit

One trained identity-dynamics core (UIDM: persistent memory, set-level
competition, Existing/NEW/NO-MATCH, lifecycle, reactivation) serves
**one shared checkpoint** across:

1. closed-set ordinary MOT (DanceTrack/BDD/MOT17/MOT20),
2. open-vocabulary MOT (TAO val, official TETA),
3. referring-expression MOT (Refer-Dance, official RMOT TrackEval),

with target specification (category text / open category / referring
expression) entering through a unified observation space (PBD box-end
identity token + CLIP crop + spec token) and, in the L9 main variant, a
learned per-candidate gate that modulates the semantic residual.

## Verified neighbours (all audited in `reports/l9_literature_and_code_audit.md`)

| Method | Unified identity core | Shared ckpt | MOT | OVMOT | RMOT |
|---|---|---|---|---|---|
| OVTR (ICLR 2025) | no (DETR queries) | no (OVMOT-specific) | — | yes | — |
| TRACT (ICCV 2025) | no (MASA association) | no (OVMOT-specific) | — | yes | — |
| AED (TIP 2025) | no (similarity decoder) | yes (CV+OV) | yes | yes | — |
| QTrack (2026) | no (3B VLM, RMOT) | no (RMOT-specific) | (probe) | — | yes |
| MOTIP (CVPR 2025) | identity-as-ID-prediction | yes (MOT) | yes | — | — |
| iKUN (CVPR 2024) | no (post-hoc language) | no | yes | — | yes |
| TransRMOT (CVPR 2023) | no (language DETR) | no | — | — | yes |

## Audit conclusion (interim)

We did **not** identify a published, verifiable system that:

- trains **one identity-dynamics process** with persistent memory,
  lifecycle and set-level competition, and
- evaluates **one shared checkpoint** on closed-set MOT, OVMOT and RMOT
  simultaneously, and
- represents all three WHAT specifications in one observation space.

Therefore the novelty claim is phrased as *"we did not identify ..."*,
not "first".  AED is the closest for CV+OV unification (no language, no
identity dynamics); QTrack is the closest for language-driven tracking
(no shared identity core across tasks); MOTIP is the closest for identity
prediction (ordinary MOT only).

## Finalisation checklist

- [ ] Confirm no new 2026 release appeared during the stage (final
      re-search before submission).
- [ ] Compare final L9 numbers against OVTR/OVTrack (TAO) and
      iKUN/TransRMOT (RMOT) with detector/protocol caveats in the paper.
- [ ] State the exact scientific contribution: specification-agnostic
      identity dynamics + unified observation + (gate) conditioning.

