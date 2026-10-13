# Experiment registry

| Experiment | Status | Checkpoint / scope |
|---|---|---|
| B0 old U2 | perception results preserved; no trained tracker | epoch-1 1d9077ae… |
| B1 GroundAnything + Simple | four-frame official result; full video pending | same legal candidates as Full |
| B2 GroundAnything + U2 Core | pending; no fabricated trained checkpoint | reused architecture explicitly labeled |
| B3 GroundAnything + pretrained OneJev Full | four-frame native result; full DanceTrack video running | OmniJev/OneJev-0.8B |
| B4 tracking-SFT OneJev | official100-step full-SFT check running | teacher phase; formal rollout/before-after pending |
| B5 distilled Compact | pending teacher/GT data | matched-capacity controls required |

Every run records date/commit, devices, split, checkpoints, steps/loss,
official metrics, full FPS, peak memory, failures and next action. Empty
metrics are pending work, never inferred performance.
