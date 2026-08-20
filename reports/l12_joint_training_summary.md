# Stage L12 — Joint Training Summary

Date: 2026-08-20

Status: **NOT RUN in this session.**

The Stage L12 prompt says: "Frozen shared UIDM + trainable prompt
adapter; if point/box/mask all show clear positive signal, continue to
joint fine-tuning."

The frozen-phase controlled evaluation
(`l12_prompt_robustness_summary.md`) showed:

- prompt-modality sensitivity exists (mask/point > box);
- absolute persistence is only ~0.41-0.49 with ~16-27% switches;

which is a mixed/weak signal, not a clear positive.  Per the protocol,
joint fine-tuning was therefore not launched.  The infrastructure for it
is in place:

- DAVIS pkls with DLA candidates + CLIP + PBD + GT masks;
- per-object seed PBD + CLIP for mask/box/point;
- seeded-only tracker policy with forced birth states;
- `tools/eval_l12_davis.py` for identity metrics.

If L11 task balance is restored and compute remains, the joint
fine-tune would: build prompt-seeded training batches (seed token =
full birth token as in training), train the adapter (+ UIDM) on DAVIS
train/val splits, and re-run the same controlled protocol.
