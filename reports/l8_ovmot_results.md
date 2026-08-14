# Stage L8 — OVMOT Results (TAO val, official TETA)

Protocol: identical to L7's official TAO OVMOT evaluation (TETA50,
Base/Novel/All, LVIS v1 classes, official `run_ovmot.py`). Candidates are
the official Detic public detections used by L7. The tracker is the L8 v2
shared checkpoint; TAO candidates have no cached PBD identity tokens, so
the model operates in the semantic-only (PBD-zero) regime trained via
PBD-dropout. Classification uses frozen CLIP ViT-B/32 text embeddings over
LVIS categories (same as L7's CLIP classification row).

## Table B — TAO OVMOT

| Method | Split | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|---|
| L7 CLIP-only probe, Detic cls (reference) | All | 31.48 | — | 29.51 | 0.14 |
| L7 CLIP-only probe, CLIP cls (reference) | All | 33.94 | — | 29.51 | 7.51 |
| L8-B1 sem-in-core (PBD zero) | All | 34.07 | 65.06 | 29.64 | 7.52 |
| L8 v2 shared (PBD zero) | Base | 34.33 | 65.14 | 30.45 | 7.40 |
| L8 v2 shared (PBD zero) | Novel | 34.36 | 64.41 | 30.40 | 8.27 |
| L8 v2 shared (PBD zero) | All | **34.33** | 65.05 | **30.44** | 7.51 |

Notes:

- Same official protocol/evaluator as L7 (TETA50, Base=non-r, Novel=r).
- L8 runs in the semantic-only regime (TAO candidates have no cached PBD),
  enabled by PBD-dropout training of the same shared core.
- AssocA 30.44 > L7 probe 29.51; Base≈Novel gap 0.05pp; ClsA 7.51 ≈ L7's
  CLIP classification 7.51.
- L8-B1 (sem-in-core) is slightly lower (TETA 34.07, AssocA 29.64,
  Base/Novel gap 1.9pp), still above the L7 probe.
- Raw predictions: `outputs/l8/trackeval/ovmot_v2e/trackers/UIDM/data/pred.json`
