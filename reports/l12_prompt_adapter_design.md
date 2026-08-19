# Stage L12 — Prompt-to-Identity Adapter Design

Date: 2026-08-19

## Principle

Do not create a PromptTracker.  The specification space is extended with
a `seeded` policy; UIDM sees only a unified seed identity token
(point/box/mask -> promptable localizer -> region crop -> PBD identity
token).  The localization/mask head is decoupled from identity.

## Mapping

```
point -> SAM2/3 point prompt -> region mask -> crop -> PBD token
box   -> SAM2/3 box prompt   -> region mask -> crop -> PBD token
mask  -> SAM2/3 mask prompt  -> region mask -> crop -> PBD token
```

All three share:

- region crop extraction (same crop/CLIP/PBD pipeline as MOT/OVMOT/RMOT);
- one trainable prompt adapter that converts (prompt embedding, region
  features) into the unified seed identity token;
- the same UIDM transition decoder (Existing / NO_MATCH / reactivation;
  NEW disabled or restricted for seeded identities).

## Transition policy for seeded identities

- Seeded identities may only: persist (Existing), be absent
  (NO_MATCH), or reactivate.
- NEW is not a legal transition for a seeded identity unless the spec
  also requests semantic discovery (not used in the controlled prompt
  evaluation).
- Multiple simultaneous seeds are independent slots; competition is
  resolved by the same pair/col decoding as MOT.

## Training plan

1. Frozen shared UIDM + trainable prompt adapter on DAVIS-style
   prompt-seeded data; measure positive signal for point/box/mask.
2. If positive, joint fine-tune MOT + OVMOT + RMOT + prompt tracking
   into one shared checkpoint.
3. Final evaluation uses ONE checkpoint for all formulations
   (task-specific checkpoints only as upper-bound analysis).

## Reference code inspected (no code copied)

- SAM2 official repo (Apache-2.0): promptable video predictor,
  per-object memory bank.
- SAM-PT: point propagation -> mask decoding (WACV 2025).
- DEVA: decoupled mask propagation + recognition (ECCV 2024).
- PS-MOT (MIT): SAM-based point->box pseudo labels with temporal
  feedback (also relevant to L11).
