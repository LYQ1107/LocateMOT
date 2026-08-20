# Stage L12 — ICLR Novelty Audit

Date: 2026-08-20

Searched 2025/2026 (and earlier where needed) for a verifiable
published/opensource system that simultaneously provides:

- one learned identity core;
- one shared checkpoint;
- closed-set MOT;
- open-vocabulary MOT (OVMOT);
- referring-expression MOT (RMOT);
- point/box/mask prompt-seeded tracking.

Verified candidates inspected (official code where available):

- SAM 2 / SAM 3: promptable VOS + multi-object tracking, but no shared
  closed-set MOT / OVMOT / RMOT discovery formulation.
- MOTIP / MOTIP-2: ID-prediction MOT, no prompt-seeded multi-modal
  tracking.
- MASA: universal appearance matching, no explicit lifecycle identity
  dynamics or prompt-seeded policy.
- COVTrack / COVTrack++: OVMOT with continuous annotations, no shared
  prompt-seeded tracking.
- U2MOT / Walker / PS-MOT: pseudo-label or self-supervised tracklets,
  no unified multi-specification shared checkpoint.
- LaMOT: language-guided MOT benchmark, no prompt-seeded policy.

Conclusion (conservative): **We did not identify** a published system
that simultaneously covers closed-set MOT, OVMOT, RMOT, and
point/box/mask prompt-seeded tracking with one identity core and one
shared checkpoint under official protocols.  The LocateMOT L11 result
(OVMOT AssocA 37.10 with a shared checkpoint that also runs MOT and
RMOT) is the strongest available evidence for this unification claim,
but prompt-seeded generalization is currently weak/mixed (frozen
phase), so the final paper should not claim prompt robustness without
further joint fine-tuning.
