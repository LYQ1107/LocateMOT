# Stage L12 — Failure Analysis (Frozen Prompt-Seeded UIDM)

Date: 2026-08-20

## What failed

The frozen shared UIDM (L11 step10000) does NOT reliably maintain
point/box/mask-seeded identities on DAVIS 2017 val multi-object videos
with DLA candidates:

- persistence ~0.41-0.49 at a permissive match threshold;
- identity switch rate 0.16-0.27 among matched object-frames;
- box seeds are the weakest modality (switch ~0.27).

## Root causes (evidence-based)

1. **Seed-state mismatch with training distribution.**  The UIDM was
   trained with birth states initialized from `memory.init(cand_tok)`
   where `cand_tok` includes PBD encoder + candidate evidence MLP +
   semantic adapter output.  A prompt seed must reproduce this full
   token; even after adding PBD encoder + zero cand-MLP + adapter
   semantics, the remaining mismatch (candidate evidence features,
   DLA-crop vs prompt-crop PBD) lowers pair scores.
2. **Candidate-stream domain shift.**  DLA (Detic-SwinB, LVIS) on DAVIS
   480p frames produces a different candidate distribution than the
   MOT/OVMOT training streams; the frozen model was not fine-tuned on
   this stream.
3. **Prompt-localization ambiguity.**  Box seeds (tight bbox) are less
   identity-discriminative than mask/point seeds, consistent with the
   higher switch rate for box.
4. **Threshold sensitivity.**  At match-thr 0.0 almost no matches occur
   (pair logits are low); the operating point must be relaxed to -1/-2,
   which increases false matches.

## What did NOT fail

- The shared UIDM architecture and lifecycle machinery accept forced
  seed births and propagate them across frames (seeded-only policy
  works).
- Prompt modality measurably affects identity dynamics (mask/point >
  box), giving a falsifiable scientific signal.
- The same checkpoint retains its MOT/OVMOT/RMOT capabilities (L11
  results), so the prompt interface did not corrupt shared identity
  learning.

## Required next step (if pursued)

Joint fine-tune: frozen/soft UIDM + trainable prompt adapter on
prompt-seeded DAVIS data with seed tokens built exactly like training
birth tokens, plus candidate-stream adaptation.  This is the L12
joint-training step; it was not executed in this session because the
frozen result did not meet the "clear positive signal" bar and L11 task
balance (RMOT-Dance / ordinary MOT) was prioritized.
