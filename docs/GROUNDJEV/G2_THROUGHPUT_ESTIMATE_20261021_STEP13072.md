# G2 Clean throughput estimate — step 13072

- Observed: `2026-10-21T06:39:56.369618+08:00`
- Event step: `13072/18227` (event time `2026-10-21T06:39:45+08:00`)

| Window | Steps/min | Remaining time |
|---:|---:|---:|
| 100 events | 5.170 | 16.62 h |
| 500 events | 5.203 | 16.51 h |
| 1000 events | 5.200 | 16.52 h |

The estimate uses actual event timestamps under the protected two-rank run on GPUs 3/5. It is an operational ETA only; it is not a benchmark or scientific performance claim. Downstream jobs remain gated on the terminal G2 receipt.
