# Experiment registry

| Experiment | Status | Checkpoint / scope |
|---|---|---|
| B0 old U2 | perception results preserved; no trained tracker | epoch-1 1d9077ae… |
| B1 GroundAnything + Simple | four-frame official result; full video pending | same legal candidates as Full |
| B2 GroundAnything + U2 Core | pending; no fabricated trained checkpoint | reused architecture explicitly labeled |
| B3 GroundAnything + pretrained OneJev Full | four-frame native result; full DanceTrack video running | OmniJev/OneJev-0.8B |
| B4 tracking-SFT OneJev | 100-step valid reload; negative same-candidate metrics; own-history500-step running | GT-proposal curriculum; actual DLM rollout pending |
| B5 distilled Compact | pending teacher/GT data | matched-capacity controls required |

Every run records date/commit, devices, split, checkpoints, steps/loss,
official metrics, full FPS, peak memory, failures and next action. Empty
metrics are pending work, never inferred performance.

Evidence controls V1/V2/V3 and their negative results are recorded in
outputs/groundjev/metrics/mot17_first4_sft_evidence_controls.json. No zero
IDSW result from a collapsed predictor counts as good association.
