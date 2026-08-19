# Stage L12 — Prompt-Seeded Benchmark Protocol (Controlled)

Date: 2026-08-19

## Benchmark choice

Primary: **DAVIS 2017 val** (locally present at
`/data1/LWR/vranlee/SERVER_ONLY/avis/DAVIS/DAVIS`, 819 MB).

- Multi-object (2017 protocol), 30 val videos, first-frame mask
  supervision, official J&F evaluator available via
  `davisvideochallenge/davis-2017`.
- No official point/box prompt protocol for DAVIS; the official
  protocol is first-frame mask.

## Controlled prompt-type evaluation (explicitly NOT official)

From the official first-frame GT mask of each object we derive three
controlled prompt types:

1. **mask**: the GT mask itself (official protocol).
2. **box**: tight bounding box of the GT mask (deterministic).
3. **point**: one deterministic interior point (e.g., the top-left
   interior pixel of the mask's bounding box, or the mask centroid if
   it lies inside the mask).

These are called **controlled prompt-type evaluations**.  They compare
the same identity-dynamics core under different prompt modalities; they
are NOT presented as official point/box benchmark results.

## Evaluation design (identity-dynamics focus)

The scientific question is not "best VOS mask quality" but:

> Does one shared learned identity-dynamics core (UIDM) persist seeded
> identities across prompt modalities (point / box / mask)?

For each DAVIS val video:

- Seed every first-frame object's identity token from the prompt
  (point/box/mask -> localizer -> region crop -> PBD identity token).
- Run the shared UIDM over the video with its normal candidate stream
  (DLA detections or SAM2 regions on subsequent frames).
- Measure identity persistence:
  - fraction of seeded identities still matched (IoU >= 0.5) at later
    frames;
  - identity switches (a seeded id jumping to a different object);
  - fragmentation (same object receiving a new id after NO_MATCH);
  - NEW-birth rate (should be ~0 for seeded-only policy);
  - multi-object competition (two simultaneous seeds must not merge).

Same-video same-target comparisons of point vs box vs mask seeding
answer whether identity dynamics is robust to prompt modality.

## Localizer policy

- Prefer SAM3 if its checkpoint becomes locally available/authorized.
- Fallback: SAM2 (present locally) maps point/box/mask -> mask/region.
- The localizer is used ONLY to obtain the seed region; UIDM remains
  the identity core (no VOS-specific tracker).

## Metrics

- Identity persistence (IoU-matched continuation rate)
- Identity switch rate
- Fragmentation rate
- NEW rate under seeded policy
- Per-prompt-type robustness table (point/box/mask)

Official VOS metrics (J&F) are reported only for the mask-seeded
condition and only if a mask head is available; otherwise the report
explicitly states that mask output is out of scope and identity metrics
are the primary evidence.
