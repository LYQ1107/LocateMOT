# Stage L12 — Box-Prompt Seeded Identity Results

Date: 2026-08-20

Same controlled protocol as `l12_point_results.md`; box seeds are the
tight first-frame GT-mask bbox -> crop -> PBD token + CLIP.

## Match-threshold sweep

| match-thr | frames | matched | persistence | switch |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2,093 | 239 | 0.114 | 0.205 |
| -1.0 | 2,093 | 858 | 0.410 | 0.270 |
| -2.0 | 2,093 | 977 | 0.467 | 0.268 |

Reading: box-seeded identities persist similarly to mask/point at
threshold -2 (0.467) but have the HIGHEST identity-switch rate (~27%),
i.e. the box-only seed is the least robust prompt modality for identity
dynamics in this controlled comparison.
