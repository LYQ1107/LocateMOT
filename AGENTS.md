# AGENTS.md

## 通用原则

1. 我们不是一篇安全攻防论文，你有权力进行校验，但是禁止禁止禁止过度防御。
2. 禁止为反复出现的基本不可能发生的 case 写防御。
3. 需要 rubric 的地方不要过度机械化。
4. 每次任务开始先确认项目身份：当前项目根目录必须是
   `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT`。若收到的任务/附件指向
   其它项目（例如 OCD_OVMOT / TrackOCD），先提醒用户“这是 LocateMOT 任务吗”，
   避免把复制错的任务执行到本项目。

## 研究执行原则

1. 先阅读代码 -> 写简短研究计划 -> 开始实现和实验，不要长时间停留在文档规划。
2. 仅做保证数值结果可信所必需的 sanity checks；不要把主要精力用于单元测试
   覆盖率、hash、格式化、CI、异常输入防御等。
3. 维护 `research_log.md`，简洁记录：实验假设、失败现象、原因判断、修改内容、
   结果变化、是否保留。

## Long-running experiment policy

For long training or evaluation jobs, never perform agent-level polling.

Use exactly one blocking shell command to wait for completion. Internal shell
sleep and process checks are allowed, but do not repeatedly return to the model
to run ps, tail, wc, ls, date, or nvidia-smi.

After the blocking command returns:

1. Validate completion and output integrity.
2. If the task failed, identify the first actionable root cause.
3. Apply the smallest justified fix.
5. Run a targeted regression test.
6. Resume from the latest valid checkpoint or unfinished unit.

Never hide failures, skip samples, change metrics, change seeds, weaken
acceptance criteria, or silently alter the experimental protocol.
