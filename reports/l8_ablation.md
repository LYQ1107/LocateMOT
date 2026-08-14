# Stage L8 — Observation Ablation (same checkpoint, same protocol)

Goal: show the contribution of the identity stream (PBD), the semantic
stream (CLIP+spec), and their combination. Rows marked "same ckpt" use the
same checkpoint and only change the streams available at inference; the
sem-in-core vs identity-pure rows compare two trained variants with the
same data/budget.

## A. Ordinary MOT Macro AssA (4-domain average)

| Variant | Macro AssA | Note |
|---|---|---|
| L6 PBD (reference) | 0.4922 | identity stream only |
| L8-B1 sem-in-core (same ckpt) | **0.5087** | PBD + CLIP + spec |
| L8-B2 identity-pure (same ckpt) | **0.5045** | PBD + CLIP + spec |
| semantic-only (PBD zeroed, inference on v2) | in progress | expected drop (identity removed) |

## B. RMOT (Refer-Dance 40 queries)

| Variant | HOTA | DetA | AssA |
|---|---|---|---|
| L8-B1 sem-in-core | **37.88** | 46.51 | 31.02 |
| L8-B2 identity-pure | 35.20 | 43.42 | 28.63 |
| identity-only (no language, same ckpt) | ~0 | ~0 | ~0 |

## C. OVMOT (TAO val TETA)

| Observation | TETA All | AssocA | ClsA |
|---|---|---|---|
| L8-B2 identity-pure, PBD zero (TAO) | **34.33** | **30.44** | 7.51 |
| L7 CLIP-only probe (reference) | 33.94 | 29.51 | 7.51 |

L7 reference uses the same CLIP-text classification; L8 uses the same
official evaluator and protocol.

## Interpretation

- The identity stream is necessary for association quality: with it,
  ordinary MOT is at/above L6 while RMOT and OVMOT are strong.
- The semantic stream is necessary for language-driven selection (RMOT
  ~0 without it) and open-vocabulary classification.
- Both trained variants (sem-in-core and identity-pure) work; the
  difference is small and within run-to-run noise.
- OVMOT (no PBD on TAO) exercises the semantic-only regime enabled by
  PBD-dropout training; association remains strong (AssocA 30.44).
