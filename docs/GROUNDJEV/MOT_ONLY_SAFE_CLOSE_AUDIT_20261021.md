# MOT-only long-training safe-close audit

- Checked: `2026-10-21T06:24:03.788039+08:00`
- Scope: all `lyq` Python/torch/train/eval/service processes on the host.
- G2 Clean: the only project optimizer, step `12991/18227`, still protected on physical GPU3/GPU5.
- MOT-only long training matches: **none**.
- Legacy MOT-only action: safely stopped; no restart issued.
- Remaining project processes are dependency queues, the GroundAnything service, and the G2 training/checkpoint workers.

This is a process audit, not a tracking metric or benchmark result.
