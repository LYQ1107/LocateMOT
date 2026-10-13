# Experiment registry

| Experiment | Status | Checkpoint / scope |
|---|---|---|
| B0 old U2 | perception results preserved; no trained tracker | epoch-1 1d9077ae… |
| B1 GroundAnything + Simple | pending | same legal candidates as Full |
| B2 GroundAnything + U2 Core | pending; no fabricated trained checkpoint | reused architecture explicitly labeled |
| B3 GroundAnything + pretrained OneJev Full | pending real model load | OmniJev/OneJev-0.8B |
| B4 tracking-SFT OneJev | pending | 100–200-step check, then finite first run |
| B5 distilled Compact | pending teacher/GT data | matched-capacity controls required |

Every run records date/commit, devices, split, checkpoints, steps/loss,
official metrics, full FPS, peak memory, failures and next action. Empty
metrics are pending work, never inferred performance.
