# G2 durable checkpoint protection — 2026-10-20

The latest durable G2 recovery state was loaded read-only from
`latest_training_state.pt`. It is step **10500**, carries the registered
clean manifest SHA, and contains optimizer, scheduler, sampler, and per-rank RNG
state. The latest adapter file is also present.

- State SHA256: `fc2c4c69266c750119fff80f6d7762236d75e01b9e296e9d51fe82c5eb4a3c9f`
- Adapter SHA256: `9c71a9e1ba692ea994af4335481803be2d020e6f5ff286f9d4349e1681b82fb2`
- World size: `2`
- Final receipt: absent; training remains live

This receipt protects recovery integrity only and makes no completion or metric
claim.
