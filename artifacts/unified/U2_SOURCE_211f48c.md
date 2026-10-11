# LocateMOT-U U2 source recovery and bounded perception preparation

Original local source commit: `211f48c707738f8c56ce5c90c81e835d8d5cb3f8`. Foundation remains `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.

This snapshot validates all original BDD validation GT/images, registers the complete 39,973-frame BDD audit, and prepares the fixed two-epoch U2-P replay. Full TAO/Dance/BDD audits and BDD train-image recovery remain active; no perception or track optimization has started. MOT17/MOT20 F0 gates still fail. This is not U2 completion.

The source tree preserves original local contents. The app publication commit differs from the original local commit. Restore original history after the existing prerequisite commit `014b8a5ec81bb83c5993b56da171089f8a440a11` is present:

```bash
git fetch artifacts/unified/u2_source_history_211f48c.bundle HEAD:refs/heads/codex/u2-original-history-20261012
```

Large datasets, checkpoints, download credentials and local helper directories are excluded.
