# Stage L4 — Full Multi-Domain Multi-Spec Training

状态：**NOT_EXECUTED**。

原因：Pilot Gate 未通过（`L4_PILOT_GATE_FAIL`）。任务书规定早期失败
直接进入 failure analysis + final report，不机械执行正式 multi-domain
multi-spec training。

现有 A2/A5 训练本身已覆盖 BDD + DanceTrack + MOT17 + MOT20 的
paired views（15,851 pairs，20 epochs，1 GPU each），可视为
pilot-scale 的多域多 spec 训练；其结论见 `reports/l4_pilot.md`。
