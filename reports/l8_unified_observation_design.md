# Stage L8 — Unified Observation / Specification Design

## 1. Problem

L7 established that identity-discriminative PBD evidence and
open-vocabulary CLIP evidence cannot be trivially exchanged: replacing the
PBD appearance token with CLIP regresses ordinary MOT Macro AssA from
0.4922 to 0.4290, while CLIP is necessary for OVMOT/RMOT semantics.

L8 asks: can one shared identity-dynamics core consume a *unified
observation* across closed-set MOT, OVMOT, and RMOT without regressing
identity?

## 2. Unified Specification interface

All three tasks emit a specification embedding in one 512-d CLIP text
space:

- ordinary MOT: category text (e.g., "person", "person, car, truck, ...");
- OVMOT: "all objects" (automatic discovery; categories assigned
  post-hoc, same as L7 protocol);
- RMOT: the referring expression sentence (e.g., "dancing person dressed
  all in white").

The UIDM never sees a dataset id / task id / one-hot router.

## 3. Candidate observation streams

Per candidate we use:

1. PBD box-end token (LocateAnything-3B, frozen, 2048-d) — identity
   evidence (same token the L6 core was trained on);
2. CLIP ViT-B/32 crop embedding (frozen, 512-d) — open-vocabulary visual
   semantics;
3. specification embedding (frozen CLIP text, 512-d) — WHAT condition;
4. existing evidence features consumed by UIDM (geometry / motion / IoU /
   generation score / track memory), unchanged.

## 4. Adapter (final design: v2, identity-pure)

`UnifiedObservationAdapter` (clean reimplementation):

```
clip_crop --clip_proj--> c
spec      --spec_proj--> s
sem = c + sigmoid(gate(c)) * s            (per-dim gated semantic residue)
relevance_logit = MLP(sem)                (language→target selection)
```

The **UIDM core consumes PBD only** (`sem_in_core=False`). The semantic
residue is used by the relevance head (WHAT), while identity persistence,
set-level competition, Existing/NEW/NO-MATCH, lifecycle and reactivation
remain in the shared UIDM core (HOW). During training:

- tracking losses (row/col/switch/lifecycle/motion) train the core;
- relevance BCE trains the adapter;
- `pbd_dropout=0.15` randomly zeroes the PBD stream so the same core can
  operate when identity evidence is unavailable (TAO OVMOT candidates have
  no cached PBD).

## 5. Variant studied: semantic residue inside the core (`sem_in_core`)

An earlier variant adds the gated semantic residue directly to UIDM
candidate tokens (candidate token = PBD token + sem). An early evaluation
appeared to show severe ordinary-MOT regression; this turned out to be an
artifact: the evaluation accidentally fed the PBD **coord-mean** token
while the core was trained on the PBD **box-end** token. After fixing the
feature key, the sem-in-core variant (L8-B1) gives Macro AssA 0.5087 and
the best RMOT result (HOTA 37.88) in this stage.

The two variants therefore stand as complementary evidence:

- L8-B1 (`sem_in_core=True`): specification enters the identity token
  stream; slightly better RMOT/ordinary numbers in our runs.
- L8-B2 (identity-pure, main): specification stays in the relevance head;
  simpler mechanism, cleaner WHAT/HOW decoupling story, and OVMOT official
  numbers obtained with the same checkpoint.

Both share the same UIDM core class, same adapter architecture, same
training data and budget; the only difference is whether `sem` is added to
the candidate token inside the core.

## 6. Why this is still "unified"

- One checkpoint: the same `uidm.*` weights and the same `adapter.*`
  weights are used for MOT, OVMOT, and RMOT;
- one shared identity-dynamics core: no task-specific tracker, head, or
  dataset router;
- specification changes only the spec embedding; selection (RMOT) and
  classification (OVMOT) consume the same unified semantic stream;
- the L7 finding is respected: WHAT and HOW are decoupled by design, but
  both are learned jointly (tracking loss + relevance loss in one
  objective).

## 7. Parameters / training

- UIDM core: L6 `uidm_full` (large, d_model=384, 6 layers), 14.37M params;
- adapter: ~0.5M trainable (clip/spec projections + gate + relevance);
- final v2 training: 2,500 DDP steps, batch 4/GPU × 4 GPUs, core LR 4e-5,
  adapter LR 1e-4, joint MOT+RMOT balanced sampling, PBD-dropout 0.15,
  seed 20260806;
- checkpoint: `outputs/l8/checkpoints/uidm_l8_v2/latest.pt`.
