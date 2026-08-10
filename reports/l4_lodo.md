# Stage L4 — LODO（Leave-One-Domain-Out）

状态：**NOT_EXECUTED**。

原因：Pilot Gate 未通过（`L4_PILOT_GATE_FAIL`）。按任务书，早期失败
直接进入 failure analysis + final report，不机械执行后续 LODO。

若后续重设计 trajectory-level consistency 并通过 pilot，LODO 计划：

- Leave-BDD-Out：训练不含 BDD，测试 BDD multi-class/category；
- Leave-DanceTrack-Out：训练其他域，测试 DanceTrack dense same-class；
- held-out 域不参与 training / calibration / threshold 选择。
