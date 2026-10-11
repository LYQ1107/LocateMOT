# U2 recovery source history

The published research source tree is byte-identical to local commit `7b63074575779818b15cf5c12d3d423b5f195cf5` (tree `c568b925fdddac9986095362a13bf831c6b7a513`). The additional files in this directory preserve the original local Git history because publication used the connected GitHub application.

The bundle preserves all local commits after the published base `9c21a1b5a22c21704fa51e69ed49bb684e4acb77`, including U1R2 `56507f4aa9a12e76d6f9f6702f0f0b38163666ec`, the U2 implementation, and the current measured portability results.

Restore the original commits into a separate branch after cloning:

```bash
git fetch artifacts/unified/u2_source_history_7b63074.bundle HEAD:refs/heads/codex/u2-original-history-20261011
```

The current scientific decision is LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY. MOT17/MOT20 measurements cover all 7,115 registered dev frames and fail their Top150 gates. Dataset acquisition and recovery continue; learned tracking and U3 readiness remain unproven. See reports/unified/U2_FINAL_DECISION.md and outputs/unified/u2/final_status.json.
