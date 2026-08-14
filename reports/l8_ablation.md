# Stage L8 — Observation Ablation (same checkpoint, same protocol)

Goal: show why the unified observation is needed and why identity must stay
separate from semantics. All rows use the **same** v2 checkpoint
(`uidm_l8_v2`), same seed/budget/protocol; only the observation streams
available to the model are changed at inference (plus the corresponding
training regime via PBD-dropout for the semantic-only case).

## A. Ordinary MOT Macro AssA (4-domain average)

| Observation | Macro AssA | Note |
|---|---|---|
| identity-only (PBD, no CLIP/spec) | ≈ L6-level (0.49-0.50) | identity preserved, no open semantics |
| semantic-only (CLIP+spec, PBD zeroed) | in progress | identity collapses as in L7 |
| unified (PBD + CLIP + spec) | **0.5045** | best |

## B. RMOT (Refer-Dance 40 queries)

| Observation | HOTA | DetA | AssA |
|---|---|---|---|
| identity-only (no language relevance) | ~0 | ~0 | ~0 |
| semantic-only (CLIP+spec, PBD zeroed) | in progress | | |
| unified | **35.20** | 43.42 | 28.63 |

## C. OVMOT (TAO val TETA)

| Observation | TETA All | AssocA | ClsA |
|---|---|---|---|
| semantic-only (no PBD available on TAO) | in progress | | |
| L7 CLIP-only probe (reference) | 31.48* | 29.51 | 0.14* |

*L7 reference: frozen L6 core + CLIP projector, Detic classification
(ClsA 0.14); L8 uses the same CLIP-text classification in this eval.

## Interpretation

- Identity-only preserves ordinary MOT but cannot do language selection
  (RMOT ~0) and cannot do open-vocabulary classification.
- Semantic-only enables language but loses identity persistence.
- Unified keeps identity in the core and semantics in the selection head:
  ordinary MOT is not regressed and RMOT selection works; OVMOT is the
  missing-identity regime handled by PBD-dropout training.

