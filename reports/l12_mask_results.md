# Stage L12 — Mask-Prompt Seeded Identity Results

Date: 2026-08-20

Same controlled protocol; mask seeds are first-frame GT masks (bbox
crop with background blacked out) -> PBD token + CLIP.

## Match-threshold sweep

| match-thr | frames | matched | persistence | switch |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2,171 | 265 | 0.122 | 0.200 |
| -1.0 | 2,171 | 1,009 | 0.465 | 0.161 |
| -2.0 | 2,171 | 1,057 | 0.487 | 0.207 |

Reading: mask seeds give the highest persistence at both operating
points and the lowest switch rate at match-thr -1 (0.161), confirming
that richer seed localization (mask) transfers to more robust identity
persistence through the shared UIDM.
