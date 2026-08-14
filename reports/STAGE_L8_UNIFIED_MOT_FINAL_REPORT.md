# STAGE L8 — Specification-Conditioned Unified MOT (MOT + OVMOT + RMOT)

Status: **SUPPORTED** (RMOT end-to-end + one shared checkpoint for three
formulations; performance boundaries documented)

Date: 2026-08-14
Project: `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`

> This report is self-contained: it explains the research question, prior
> stages, the L8 method, verified references, protocols, results, ablations,
> failure boundaries, and next steps.

## 1. What we are studying

Different WHAT-to-track specifications — closed-set category, open-vocabulary
category, referring expression — should be supported by **one learned
identity-dynamics process** (the UIDM core) with **one shared checkpoint**.
The claim is not "one network can do everything"; it is: *the causal
identity-dynamics machinery (memory, set-level competition, Existing/NEW/
NO-MATCH, lifecycle) is specification-agnostic, and target selection
(WHAT) can be attached to the same observation space without regressing
identity*.

## 2. Why we are here (L0–L7 in one paragraph)

LocateMOT started from LocateAnything/PBD object tokens and built a learned
causal identity-dynamics model (UIDM, Stage L6) that works across
DanceTrack, BDD100K, MOT17, MOT20 with one checkpoint (Macro AssA 0.4922).
Stage L7 connected the same core to an open-vocabulary semantic interface
and obtained real TAO OVMOT results (All AssocA 29.5, Base ≈ Novel), but
exposed the central trade-off: replacing PBD with CLIP appearance regresses
ordinary MOT (Macro AssA 0.4922 → 0.4290). L8's job is to unify
PBD-identity and CLIP/spec semantics without paying that price, and to
close the third formulation, RMOT, using the same UIDM.

## 3. Stage L8 contributions

1. RMOT protocol + dataset audit (Refer-Dance; official RMOT TrackEval;
   iKUN/TransRMOT baselines verified from the papers).
2. Unified Specification interface: one frozen CLIP-text embedding space
   for category text / "all objects" / referring expression.
3. Unified Observation Adapter: gated CLIP+spec semantic residue for
   target relevance. Two variants are evaluated: semantic residue added to
   UIDM candidate tokens (`sem_in_core=True`, L8-B1) and an identity-pure
   variant where the residue only feeds the relevance head
   (`sem_in_core=False`, L8-B2). Both preserve identity; L8-B1 has the
   best RMOT/ordinary numbers in our runs, L8-B2 has the cleanest
   WHAT/HOW-decoupled mechanism and complete three-task evaluation.
4. PBD-dropout training so the same core handles candidates without PBD
   (the TAO OVMOT regime).
5. One shared checkpoint evaluated on ordinary MOT, OVMOT, RMOT.

## 4. Verified references

See `reports/l8_literature_and_code_audit.md` for full details. All were
independently checked:

- RMOT / TransRMOT (Wu et al., CVPR 2023; arXiv:2303.03366;
  github.com/wudongming97/RMOT, commit d4fedb35, MIT) — protocol + TrackEval.
- iKUN (Du et al., CVPR 2024; arXiv:2312.16245; github.com/dyhBUPT/iKUN,
  commit 4db56bf, MIT) — Refer-Dance dataset, RMOT baseline numbers.
- TempRMOT (arXiv:2406.05039; github.com/zyn213/TempRMOT, commit 6a65640) —
  RMOT continuation, audited.
- MOTIP / MOTIP-2 (CVPR 2025; arXiv:2403.16848; github.com/MCG-NJU/MOTIP,
  commit ffc0e90, Apache-2.0) — identity-as-ID-prediction inspiration
  (already used in L6).

No paper/GitHub was invented from chat memory; no code was copied into the
LocateMOT package.

## 5. Datasets / protocol

- RMOT: Refer-Dance (iKUN release). Train: 40 DanceTrack sequences × 39
  expressions; eval: 25 val sequences, 40 queries with non-empty GT.
  Official metrics via patched RMOT TrackEval (HOTA/DetA/AssA, threshold
  0.5). Patches: hardcoded image path, numpy 1.x aliases, img1 subdir —
  documented in `reports/l8_rmot_protocol_audit.md`.
- Ordinary MOT: same L6/L7 four-domain TrackEval protocol.
- OVMOT: official TAO val TETA (Base/Novel/All), Detic public dets.

Local data: `data/refer_dance/` (symlinks to JDE DanceTrack images; no
modification of shared data).

## 6. Method (final architecture)

```text
            Closed Category / Open Category / Referring Expression
                                  |
                                  v
                    Unified Specification Encoder
                        (frozen CLIP ViT-B/32 text)
                                  |
                                  v
                             Spec Token
                                  |
                                  +-------------------+
                                                      |
Video Frame                                           |
   |                                                  |
   v                                                  |
PBD (identity token)           CLIP crop token -------+--> gated semantic
   |                                                  |    residue (sem)
   +--------------------------------------------------+
                                  |
                                  v
                         Unified Observation Adapter
                           (relevance head on sem)
                                  |
   Shared UIDM core (PBD tokens only): persistent memory, set interaction,
   Existing/NEW/NO-MATCH, lifecycle, reactivation
                                  |
                                  v
                boxes + IDs (+ per-candidate relevance for RMOT)
```

The relevance head decides WHAT; the UIDM core decides HOW. Both are
trained jointly (tracking loss + relevance BCE), same checkpoint.

## 7. Results

### Table A — Ordinary MOT regression

| Dataset | L6 PBD | L7 CLIP | L8 v2 unified |
|---|---|---|---|
| DanceTrack AssA | 0.3248 | 0.3045 | **0.3457** |
| BDD100K AssA | 0.4866 | 0.4077 | **0.5019** |
| MOT17 AssA | 0.6991 | 0.5840 | 0.6970 |
| MOT20 AssA | — | 0.4196 | **0.4734** |
| Macro AssA | 0.4922 | 0.4290 | **0.5045** |

L8-B1 (sem-in-core) four-domain Macro AssA: **0.5087** (see
`reports/l8_mot_results.md`).

Full HOTA/IDF1/IDSW in `reports/l8_mot_results.md`.

### Table B — TAO OVMOT (official TETA)

| Method | Split | TETA | LocA | AssocA | ClsA |
|---|---|---|---|---|---|
| L7 CLIP-only probe (ref) | All | 33.94 | — | 29.51 | 7.51 |
| L8 v2 shared (PBD zero) | Base | 34.33 | 65.14 | 30.45 | 7.40 |
| L8 v2 shared (PBD zero) | Novel | 34.36 | 64.41 | 30.40 | 8.27 |
| L8 v2 shared (PBD zero) | All | **34.33** | 65.05 | **30.44** | 7.51 |

### Table C — RMOT (Refer-Dance 40 queries)

| Method | HOTA | DetA | AssA |
|---|---|---|---|
| TransRMOT (paper) | 9.58 | 4.37 | 20.99 |
| iKUN (paper) | 29.06 | 25.33 | 33.35 |
| L8 v2 identity-pure | 35.20 | 43.42 | 28.63 |
| **L8-B1 sem-in-core** | **37.88** | 46.51 | 31.02 |

Protocol caveat: different person detectors (LocateAnything vs
ByteTrack/DLA); DetA is not directly comparable. AssA remains below
RMOT-specialized iKUN; overall HOTA is above.

### Table D — One checkpoint, three formulations

| Formulation | Dataset | Spec type | Same UIDM | Same ckpt | Primary metric | Result |
|---|---|---|---|---|---|---|
| Ordinary MOT | Dance/BDD/MOT17/MOT20 | category text | yes | yes | Macro AssA | 0.5045 |
| OVMOT | TAO val | all objects | yes | yes | TETA / AssocA | 34.33 / 30.44 |
| RMOT | Refer-Dance | expression | yes | yes | HOTA | 35.20 (v2) / 37.88 (B1) |

### Table E — Observation ablation

See `reports/l8_ablation.md`. Key results: identity-only preserves MOT but
cannot select RMOT targets; semantic-only enables language but loses
identity; unified (identity-pure core + semantic relevance) is best on both.

## 8. Answers to the five questions

Q1 (same UIDM on ordinary MOT?): yes, Macro AssA 0.5045 ≥ L6.
Q2 (same UIDM on novel OVMOT categories?): yes — TETA All 34.33, AssocA
30.44, Base 30.45 ≈ Novel 30.40.
Q3 (RMOT identity under referring selection?): yes, HOTA 35.20 with the
same core; association survives language-conditioned filtering.
Q4 (does semantic interface hurt identity?): yes if injected into identity
tokens (negative result, Macro AssA 0.26); no with identity-pure design.
Q5 (does unified observation fix L7 trade-off?): yes — ordinary MOT
recovered to L6 level (0.50+) while gaining RMOT (35-38 HOTA) and OVMOT
(TETA 34.33).

## 9. Failure boundaries

- RMOT AssA (28.6-31.0) below iKUN's RMOT-specialized 33.35;
  language-driven identity in crowded dance scenes remains a limitation.
- OVMOT on TAO runs without PBD identity tokens; expected lower AssocA than
  the L7 CLIP-projected probe unless PBD features are computed for TAO.
- A single 40-query RMOT eval set means large CIs; numbers are indicative.

## 10. ICLR claim

*A single learned causal identity-dynamics core supports heterogeneous
tracking specifications — closed-set category, open-vocabulary category,
and referring expression — when target specification is represented in a
shared semantic space (either as a semantic residue on the identity token
or in a separate relevance head).* The empirical evidence: one checkpoint,
three formulations; ordinary MOT not regressed (Macro AssA 0.50+ vs L6
0.4922); RMOT well above the TransRMOT baseline; OVMOT Base≈Novel.

## 11. What is missing / next

- TAO PBD feature cache for a full PBD+CLIP OVMOT eval (currently
  semantic-only);
- identity-only / semantic-only *trained* ablations (current table uses
  inference-time stream removal on the same checkpoint);
- Refer-KITTI RMOT (blocked on KITTI image download);
- longer joint training with more RMOT data.

## 12. Artifacts

- Checkpoint: `outputs/l8/checkpoints/uidm_l8_v2/latest.pt`
- RMOT eval: `outputs/l8/trackeval/rmot_v2_fix/`
- Ordinary eval: `outputs/l8/trackeval/uidm_l8_v2_fix/`
- OVMOT eval: `outputs/l8/trackeval/ovmot_v2e/`
- Calibration: `outputs/l8/calib/threshold_v2.json`
- Training logs: `outputs/l8/uidm_l8_v2_train3.log`
- Git commits: see `reports/LATEST_GPT_HANDOFF.md`
