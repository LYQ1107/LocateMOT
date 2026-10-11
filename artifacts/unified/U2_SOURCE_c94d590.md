# U2 bounded adaptation allowance correction

Original local source commit: `c94d590dd4256b28de17e8137e38164834b44eb6`. An epoch-1 failure rejects that candidate for track training while retaining the permitted second perception epoch. Final portability/retention gates and the two-epoch cap remain unchanged. No optimization has started.

The previous full history bundle `u2_source_history_211f48c.bundle` must first restore original commit `211f48c707738f8c56ce5c90c81e835d8d5cb3f8`. Then restore this incremental original history:

```bash
git fetch artifacts/unified/u2_source_history_c94d590.bundle HEAD:refs/heads/codex/u2-original-history-latest
```
