# Stage L12 — Point-Prompt Seeded Identity Results

Date: 2026-08-20

Controlled prompt-type evaluation on DAVIS 2017 val (10 multi-object
videos: dogs-jump, gold-fish, india, lab-coat, loading, pigs, shooting,
soapbox, paragliding-launch, kite-surf; 34 object seeds), frozen shared
UIDM (step10000) with seeded-only policy (NEW disabled).

Point seeds: one deterministic interior pixel of the first-frame GT
mask -> square crop -> LocateAnything-3B PBD token + CLIP.

## Match-threshold sweep (greedy seeded decode)

| match-thr | frames | matched | persistence | switch |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2,171 | 267 | 0.123 | 0.251 |
| -1.0 | 2,171 | 908 | 0.418 | 0.160 |
| -2.0 | 2,171 | 1,004 | 0.462 | 0.217 |

Per-video detail: `results/l12/davis_point*.json`.

Reading: the frozen shared UIDM can carry point-seeded identities
through ~46% of object-frames at match-thr -2, with ~22% identity
switches among matched frames.  Point seeds are comparable to mask
seeds and better than box seeds on switch rate.
