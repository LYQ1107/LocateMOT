# 官方代码修改记录

当前状态：尚未修改任何官方仓库文件。

## 原则

- `third_party/Eagle` 原则上只读；通过 wrapper、forward hook、subclass、adapter 或最小 monkey patch 提取 hidden states。
- 若必须修改官方文件：
  - 修改范围最小；
  - 保存独立 patch 到 `patches/`；
  - 记录原文件、修改行、原因；
  - 不把修改伪装成官方实现。

## 计划中的最小接入点（尚未实施）

1. PBD hidden-state hook：
   - 在 `Embodied/eaglevl/utils/locany/modeling_locateanything.py::generate` 的 `outputs = self.language_model(**prepare_inputs)` 处通过 forward hook / subclass 捕获 `outputs.hidden_states`（对应 MTP 6-token block）。
   - 优先不修改文件：注册 `forward_hook` 到语言模型，或用子类覆盖 `generate`。
2. Visual prompt worker：
   - 复用 `locateanything_worker.py` 的 `_crop_visual_prompt` / `_build_messages`（通过 import 调用，不复制实现，若复制则保留 NVIDIA 版权头并记录）。
3. 训练数据：
   - 不修改官方 `tools.py`；生成符合官方 JSONL 格式的两帧数据。

## 修改记录表

| 日期 | 文件 | 修改内容 | 原因 | patch 文件 |
| --- | --- | --- | --- | --- |
| （无） | - | - | - | - |
