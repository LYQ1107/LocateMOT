# GroundAnything G2 final research report

Status: **in progress — clean full-coverage training running**.

The current evidence establishes an auditable native training path, not a final detector or tracking result. The official source revision, parameter-freeze policy, DLM loss, ten-source coverage audit, legal split boundary, and two-card resource gate are complete. Formal training and independent held-out comparison remain pending; this report will be updated with checkpoint hashes, loss curves, detector controls, and MOT/RMOT/OVMOT evaluation receipts after those stages finish.

The 1,200-step historical G2 receipt is complete but protocol-contaminated by
3,317 MOT20-05 rows. The clean V2 course starts from G0 rather than inheriting
that adapter and uses a 583,251-row manifest with exact-once coverage, global
batch 32, strict per-rank recovery, and the official mixed objective. J1 remains
an active 1,200-step OneJev course; its observed step-600 state is not full-data
completion. G2/G1/J1 quality comparisons remain blocked until the clean terminal
receipt and fixed independent evaluation scopes exist.
