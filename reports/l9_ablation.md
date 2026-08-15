# Stage L9 — Ablation (4 groups)

Status: L9 main row complete; eval-time mode ablations pending GPU time
(after the TAO val PBD cache frees GPUs).

## Protocol

Same official evaluators (TrackEval four-domain; TAO TETA; Refer-Dance
RMOT).  Identity-only / semantic-only rows use the **same L9 v5
checkpoint** with the eval-time `--ablation` mode switch (as in L8), so
the comparison isolates the observation stream rather than retraining.

| Group | Ordinary Macro AssA | TAO AssocA | RMOT AssA |
|---|---|---|---|
| identity-only | TBD | TBD | TBD |
| semantic-only | TBD | TBD | TBD |
| strict decoupled (L8-B2) | 0.5045 | 30.44 (PBD-zero) | 28.63 |
| spec-conditioned (L9 v5) | **0.5090** | TBD (full PBD) | 30.30 |

Reference (L8 report, same protocol): identity-only Macro AssA 0.4014;
semantic-only ~0.4014.

