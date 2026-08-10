# Latest GPT Handoff

日期：2026-08-11 00:35。项目：LocateMOT。

## Stage L4 结论：PILOT FAIL（ICLR NOT_READY）

完整 handoff：`reports/STAGE_L4_GPT_HANDOFF.md`

要点：

1. specification restriction 真实改变 identity
   （`L4_SPEC_RESTRICTION_SIGNAL_SUPPORTED`）：BDD 33–67%、
   DanceTrack 31–32%、TAO 14–24% drift；
2. A2/A5/A5p paired-consistency 训练均未降低 drift，ALL 官方
   TrackEval 保持 U0 水平（macro AssA 0.4013）；
3. 失败根因：身份漂移是时间现象，单帧 consistency 不足；
4. 下一步唯一建议：trajectory-level 一致性（clip 级可微 track
   propagation / path consistency），否则不宣称方法创新。
