# AGENTS.md

## 通用原则

1. 我们不是一篇安全攻防论文，你有权力进行校验，但是禁止禁止禁止过度防御。
2. 禁止为反复出现的基本不可能发生的 case 写防御。
3. 需要 rubric 的地方不要过度机械化。

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
