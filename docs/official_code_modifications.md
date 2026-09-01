# Official Code Modifications（Stage L1-C）

所有修改均基于 NVlabs/Eagle commit 783f656d；修改目的仅为 A100 兼容与
LocateMOT 数据接入，不改变官方训练语义。

## 1. `Embodied/eaglevl/train/locany_finetune_magi_stream.py`

- 位置：LocateAnything 加载路径（约 line 1337）与 vision-only 路径
  （约 line 1384）。
- 修改：`config.vision_config._attn_implementation = 'flash_attention_2'`
  → `'eager'`。
- 原因：A100 无 magi/flash_attn；官方 sdpa_attention 的 mask 形状在
  transformers 4.57 下报 CUDA driver error（实测），eager 路径可运行。
- 影响：视觉编码使用官方 eager attention（内存/速度略低，数值等价）。

## 2. `Embodied/eaglevl/model/locany/modeling_locateanything.py`

- 位置：`LocateAnythingForConditionalGeneration.__init__`（约 line 95）。
- 修改：vision attn 同样 `flash_attention_2` → `'eager'`。
- 原因：模型类自身会覆盖训练脚本设置的 vision attn（实测），必须同步改。

## 3. A100 运行参数（不改代码，仅记录）

- `--attn_implementation sdpa`（文本 LLM 用 SDPA；官方默认 magi 仅
  Hopper/Blackwell）。
- `--max_seq_length 4096 --max_num_tokens_per_sample 4096
  --max_num_tokens 4096 --video_total_pixels 8192 --packing_buffer_size 1`
  （A100 40GB 内存限制；官方 TRAINING.md 说明 SDPA 仅支持 ~4K）。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 4. 依赖适配（非官方代码修改）

- `third_party/eagle_deps`：datasets/dotenv/filetype/bitstring/ebmlite/
  decord/hjson/msgpack/py-cpuinfo/protobuf 3.20.3/liger-kernel v0.3.1
  （官方要求 0.3.1；PyPI 镜像只有 0.8.1，API 不兼容，从 GitHub tag 安装）。
- `third_party/DeepSpeed`：官方要求 deepspeed==0.15.4；镜像无该版本，
  从 GitHub v0.15.4 clone 源码加入 PYTHONPATH（未编译 C++ ops）。

## 5. 验证结果（smoke）

- 5 步训练 loss=3.15（finite），trainable params=119,734,272（LLM LoRA
  rank 64 + MLP connector，base LLM/backbone 冻结）。
- checkpoint save→load→同一 prompt 生成正常（输出 PBD box tokens）。
