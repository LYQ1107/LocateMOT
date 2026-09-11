# R1 Aligned Track Conditioning Hook — formal training report

## Scope

R1 is an isolated RMOT sidecar experiment in worktree
`/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`. It does not modify the
canonical LocateMOT checkout, ordinary MOT/OVMOT/TAO, UIDM, the L69 bank, the
L89E anchor, or any production entrypoint. The experiment uses expression-level
fit supervision and is not a final ordinary-RMOT result.

## Registered configuration

- Seed: `20260829`.
- Domains: separate V1 and V2 heads, trained from the same architecture and
  hyperparameters.
- Formal schedule: six epochs, 2,250 optimizer steps, the exact R1 request
  manifest, and the registered deterministic fit sampling.
- Sidecar: `5,204,510` trainable parameters; hidden size 256; 8 attention
  heads; one stage mixer; two language/detail cross-attention blocks; one
  causal geometry GRU; two residual set-reasoning layers.
- Frozen anchor: L89E Stage-S checkpoint
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt`
  (SHA256
  `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`).
- Frozen Rule-B decision values: candidate threshold `1.0`, presence
  threshold `0.5`, null margin `0.0`.
- No presence residual was enabled: the frozen-anchor presence-only miss was
  `3/400 = 0.0075`, below the preregistered `0.05` activation threshold.

## Training contract result

Authoritative output:
`outputs/r1/train/formal_fit_attempt1/metrics_r1_formal.json`.

- `finite_steps=2250` and `nonzero_gradient_steps=2250`.
- All six epochs completed under world size 4.
- Strict reload passed for all eight saved V1/V2 checkpoints; each checkpoint
  is 63,115,618 bytes and has a distinct recorded SHA256.
- Anchor remained frozen; candidate deletion and truncation were false; no
  persistent raw/dense debug cache was created.
- Peak allocated/reserved CUDA memory was 1,529,634,816 / 2,730,491,904
  bytes. Wall time was 11,133.816 seconds.
- Token/span-to-region and static/motion alignment remain `UNALIGNED`.

This is valid formal fit and reload evidence. It does not establish expression
correspondence or ordinary RMOT performance.

## Preserved artifacts

The fit trace, sampling trace, eight checkpoints, reload audit, provenance and
status are retained in `outputs/r1/train/formal_fit_attempt1/`. Technical
replay/aggregate failures remain in their original attempt directories with
their `INCOMPLETE` evidence. No failed attempt was silently promoted.
