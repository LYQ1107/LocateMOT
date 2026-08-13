# Stage L7 Specification Encoder Design

## 抽象接口

```text
s = SpecEncoder(spec),  spec ∈ {ALL, "category text", "referring expression"}
relevance(j) = cos(app_embed(candidate_j), s)
```

- ALL：所有候选 relevant（保留全部，selection = identity dynamics 的
  NEW/NO-MATCH 机制本身），等价于无硬过滤。
- category text：spec = CLIP text embedding("a {category_name}")，
  relevance 门控或软加权进候选特征。
- referring text（RMOT）：同接口，text encoder 换成 language encoder
  （RoBERTa 风格，参照 TempRMOT 接口）；首版只做标准 RMOT，不做复杂推理。

## 首版（Stage L7 OVMOT）

- frozen CLIP ViT-B/32：
  - 文本侧：LVIS v1 1203 类别名 `"a {name}"` embedding（与官方 MASA
    预计算 `lvis_v1_clip_a+cname.npy` 核对：ViT-B/32 + "a {}" 模板
    mean cosine 0.9999，可直接复用其协议定义）；
  - 图像侧：candidate crop 224×224 归一化后 encode_image，512-d。
- 分类输出（TETA ClsA）首版直接使用官方 Detic 预测 label（frozen
  perception），tracker 只负责 association；后续可用
  argmax cosine(crop, class embeddings) 做 CLIP 分类对照。
- 外观 token = CLIP crop embedding（512-d）替换 PBD（2048-d）进入
  UIDM 前端投影器；其余身份动力学参数与 closed-set 完全相同。

## 不做

- 不训练大 VLM；不试五个 foundation models（只选 CLIP ViT-B/32 一个
  主方案，LocateAnything 经审计不具备文本对齐的 category 语义，只保留
  在 closed-set 外观证据）。
- 不做复杂 referring reasoning（ReaMOT 级）——留在 RMOT 成功后再评估。

