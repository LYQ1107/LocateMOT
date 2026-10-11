# U2 complete source data and automatic bounded perception run

Original local source commit: `0b6f1d4e20faed17bc105b67aef4505bcdc87ea2`. All seven required streams have complete source coverage. BDD original train/val image counts are 278,079/39,973. Complete DanceTrack recall is 0.996313536/0.997685967 at Top150/Top300. The real 20,000-row native eight-rank replay has no duplicate/omitted sample or removed box.

The live controller waits for complete original BDD/TAO audits, then owns at most two 1,250-step training epochs and full same-scope audits before selecting weights. Optimizer steps at this snapshot: zero. No learned track training or U3 readiness is claimed.

Restore original history after the original prerequisite commit `c94d590dd4256b28de17e8137e38164834b44eb6` (preserved by preceding bundles):

```bash
git fetch artifacts/unified/u2_source_history_0b6f1d4.bundle HEAD:refs/heads/codex/u2-original-history-latest
```
