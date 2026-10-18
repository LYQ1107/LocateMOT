# GroundAnything G2 final research report

Status: **in progress — clean full-coverage training running**.

The current evidence establishes an auditable native training path, not a final detector or tracking result. The official source revision, parameter-freeze policy, DLM loss, ten-source coverage audit, legal split boundary, and two-card resource gate are complete. Formal training and independent held-out comparison remain pending; this report will be updated with checkpoint hashes, loss curves, detector controls, and MOT/RMOT/OVMOT evaluation receipts after those stages finish.

The 1,200-step historical G2 receipt is complete but protocol-contaminated by
3,317 MOT20-05 rows. The clean V2 course starts from G0 rather than inheriting
that adapter and uses a 583,251-row manifest with exact-once coverage, global
batch 32, strict per-rank recovery, and the official mixed objective. J1 remains
an active 1,200-step OneJev course; its observed step-650 state is not full-data
completion. G2/G1/J1 quality comparisons remain blocked until the clean terminal
receipt and fixed independent evaluation scopes exist.

The post-launch clean contract audit independently confirms unique row IDs,
zero MOT20-05 rows, zero RMOT placeholder rows, and no cross-source media-path
intersection. The raw MOT20 directory is still scanned for validity statistics,
so the audit's raw inventory includes the excluded MOT20-05 directory; this does
not alter the clean manifest. G1 GPU5 adaptation is registered behind the clean
G2 terminal receipt and has not been counted as complete.

The deployable clean G2 export is also registered as a dependent CPU-only handoff. It waits for both the exact G2 terminal receipt and the G1 GPU5 queue receipt, then merges the adapter into a fresh G0 copy while preserving the source bundle. Until that handoff writes G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED, there is no promoted G2 detector and no final comparison result.
