# Stage L1-C LoRA（Route B）状态

## 已就绪

- 官方审计：`docs/l1_c_locateanything_lora_audit.md`
  （NVlabs/Eagle commit 783f656d，LLM LoRA rank 64、backbone 0、
   freeze_llm/backbone=True、freeze_mlp=False、lr 2e-5、
   magi 仅 Hopper → A100 必须 sdpa + 4K 序列）。
- 训练数据：`outputs/l1_c/lora/*.jsonl`
  （dancetrack 1,643 / bdd100k 7,887 / tao_amodal 17,208 / mot17 864 /
  mot20 166；官方 JSONL 格式，绝对 image 路径）。
- Recipe：`configs/l1_c/lora_recipe.json`；smoke recipe：
  `configs/l1_c/lora_smoke_recipe.json`（各 400 行 × 3 数据集）。

## 环境依赖（已解决）

使用 locatemot env + 独立 target 依赖目录：

- `third_party/eagle_deps`（datasets/dotenv/filetype/bitstring/ebmlite/
  decord/hjson/msgpack/py-cpuinfo/protobuf 3.20.3/liger-kernel v0.3.1）
- `third_party/DeepSpeed`（GitHub v0.15.4 源码，PYTHONPATH 引入）
- 运行需 `PYTHONPATH=Embodied:eagle_deps:DeepSpeed`
- 官方代码两处 vision attn patch 见 `docs/official_code_modifications.md`

## 官方脚本（A100 修改版，smoke 已验证）

```bash
cd third_party/Eagle/Embodied
HF_TOKEN=dummy LAUNCHER=pytorch RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=N python -u eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path models/LocateAnything-3B \
  --max_steps 300 --output_dir outputs/l1_c/checkpoints/lora \
  --meta_path configs/l1_c/lora_recipe.json \
  --block_size 6 --attn_implementation sdpa --causal_attn False \
  --freeze_llm True --freeze_mlp False --freeze_backbone True \
  --use_llm_lora 64 --use_backbone_lora 0 \
  --max_seq_length 4096 --max_num_tokens_per_sample 4096 --max_num_tokens 4096 \
  --video_total_pixels 8192 --packing_buffer_size 1 \
  --save_strategy steps --save_steps 100 --learning_rate 2e-5
```

## smoke 验证项（已完成）

1. trainable params 只含 LLM LoRA（+ MLP connector，因为 freeze_mlp=False）。
   → 119.7M / 3.52B（3.4%）。
2. 5 步 loss=3.15 finite；无 NaN。
3. save → load → 相同 prompt 生成正常（PBD box tokens）。
4. visual prompt 推理当前权重不支持 → 采用 grounding SFT（sequential）。

## 下一步

- 正式 LoRA grounding 训练已完成（300 步，train_loss=1.27，
  checkpoints: outputs/l1_c/checkpoints/lora/checkpoint-300）。
- 阻塞：LoRA 合并模型的 instrumented PBD 特征提取
  （tools/cache_l1c_lora.py + ObjectTokenExtractor）单帧 10 分钟未完成，
  疑似 MTP 快速路径与 merged 模型/旧驱动组合下极慢或挂起；需调试
  `_generate_loop`（事件/hook 或 MTP 路径），或改用官方 worker 的
  batch inference 后接 PBD hidden 提取。
- 若提取无法在合理成本内打通：记录
  `AC_LORA_FIXED_BOX_FEATURE_EXTRACTION_UNSUPPORTED`，UAL 主表不填，
  阶段结论以 UAF 负结果 + LoRA grounding 可训练性为主。
