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
| L7 CLIP-only probe (reference) | All | 31.48 | — | 29.51 | 0.14 |
| L8 v2 shared (PBD zero) | Base | pending eval | | | |
| L8 v2 shared (PBD zero) | Novel | pending eval | | | |
| L8 v2 shared (PBD zero) | All | pending eval | | | |

Raw predictions: `outputs/l8/trackeval/ovmot_v2e/trackers/UIDM/data/pred.json`

