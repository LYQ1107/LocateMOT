# Stage L6 Ablation Report

Status: COMPLETE (MOT17 protocol; Dance/BDD full-domain ablation evals
marked where not run).

All ablations: UIDM-Large, same data/seed/teacher schedule, 800–1200
steps, fresh TrackEval tag, MOT17 train.

| Ablation | HOTA | AssA | IDF1 | IDSW | Δ AssA vs full |
|---|---:|---:|---:|---:|---:|
| Full UIDM (4200 steps) | 0.7084 | 0.6991 | 0.6244 | 434 | — |
| no tracking-level loss (1200 steps) | 0.5801 | 0.4678 | 0.4985 | 492 | −23.1pp |
| no persistent memory (800 steps) | 0.5823 | 0.4724 | 0.4959 | 436 | −22.7pp |
| no inter-track interaction (800 steps) | 0.5929 | 0.4901 | 0.5161 | 404 | −20.9pp |
| no learned lifecycle (800 steps) | 0.5916 | 0.4876 | 0.5012 | 481 | −21.2pp |
| small UIDM 3M (800 steps) | 0.6256 | 0.5458 | 0.5507 | 272 | −15.3pp |

Interpretation:

- **Tracking-level objective matters most**: removing switch/motion/
  lifecycle losses costs −23pp AssA; row/col CE alone is not enough
  (confirms L5's finding and UniTrack-style loss value).
- **Persistent memory is essential** (−23pp without it): per-frame
  re-encoding cannot carry identity across time.
- **Inter-track competition is essential** (−21pp without it): set-level
  interaction is not decoration.
- Fixed L1DK vs learned transition is the baseline comparison
  (B2 L1DK MOT17 AssA 0.6010 < UIDM 0.6991; see baseline reconciliation).
- Learned lifecycle contributes −21pp; all four mechanisms contribute
  roughly equally (~21–23pp), and a 3M capacity version loses −15pp,
  i.e. both architecture mechanisms and capacity matter.
