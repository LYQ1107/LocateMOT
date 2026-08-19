# Stage L10 — ICLR Novelty Audit

Date: 2026-08-17 (status: final after L10 results)

The full 2025/2026 audit is in `reports/l10_literature_and_code_audit.md`.
No published, verifiable system was identified that combines:

1. one trained identity-dynamics core (persistent memory, lifecycle,
   Existing/NEW/NO-MATCH, set-level competition);
2. one shared checkpoint across closed-set MOT, OVMOT and RMOT;
3. a unified frozen observation space (PBD box-end + CLIP + spec).

Nearest neighbors: COVTrack/C-TAO (OVMOT supervision only), OVTR/TRACT/
AED (OVMOT association only), QTrack (query-driven RMOT with 3B VLM+RL),
MOTIP/iKUN (single-task ID prediction / RMOT), TempRMOT/Refer-KITTI-V2
(RMOT data only), ReaMOT/CRMOT (reasoning/cross-view RMOT).

The claim remains "we did not identify ...", not "first".  L10 adds
full-supervision scaling evidence for the same unified core.

