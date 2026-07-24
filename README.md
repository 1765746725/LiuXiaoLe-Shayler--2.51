# 柳小乐 (Shayler) 2.51

**一个 280.9M 的中文对话模型，身份为「柳小乐」——温柔、细腻、有点幽默感的 AI 助手。**

---

## 前言

这是一个个人练习的模型项目，主旨在于从零开始的模型设计和开发。目前模型已经训练完成，但仍然存在许多问题：

- **身份残留**：训练数据中仍有少量原始身份（如 MOSS）未被完全清洗干净
- **数据量削减**：数据经过大规模清洗后，比原始数据少了将近一半，导致模型对 `user` / `assistant` 的对话格式未能完全学会
- **格式问题**：少部分数据内容的格式尚未修正

小模型对数据质量极其敏感，容错率远比大模型低，需要足够高质量的数据才能发挥效果。为数不多的优势在于训练成本和数据量需求明显优于大模型。

目前柳小乐已在 **v2.51** 完成训练，实际测试效果可能没有预期理想，但对个人而言也不算差。在后续 **v3.0** 版本中，将完善以上问题，并引入全新内容（如多模态、底层模型框架重构等方向）。

详情见log日志文件

模型文件和训练数据都已拆分使用需要split_merge.py合并
> 本次模型使用的训练数据均来自 **MiniMind** 数据集。

---

## 模型概览

| 项目 | 说明 |
|------|------|
| 参数量 | 280.9M |
| 架构 | StdAttn + GQA (4KV) + QK-Norm + SwiGLU |
| 上下文长度 | 1024 tokens（约 1460 中文字符） |
| 词表 | MiniMind 8K Metaspace BPE |
| 训练精度 | BF16 |
| 训练平台 | AMD MI300X 192GB (ROCm) |

## 文件夹结构

```
LiuXiaoLe(shayler)-2.51/
├── code/                       # 训练 & 推理代码
│   ├── eval.py                 # 模型加载、推理脚本
│   ├── pretrain_v5.1.ipynb     # Base 预训练
│   ├── sft_v30_stage1.ipynb    # SFT Stage 1 — 纯对话训练
│   ├── sft_v31_stage2_identity.ipynb  # Stage 2 — 混合身份注入
│   ├── mix_stage2_data.ipynb   # Stage 2 数据混合（9:1）
│   ├── preprocess_pretrain_data.ipynb  # 预训练数据预处理
│   └── clean_base_pretrain_data.ipynb  # Base 数据清洗
│
├── model/                      # 三个阶段的 checkpoint
│   ├── step_015000.pt          # Base v5.1（预训练完成）
│   ├── sft_step_014000.pt      # SFT Stage 1（纯对话）
│   └── sft2_step_001450.pt     # Stage 2 最佳（身份注入后）★
│
├── tokenizer_minimind_8k/      # 分词器
│   ├── tokenizer.json
│   └── vocab_analysis.txt
│
├── data_stage1/                # Stage 1 训练数据
│   ├── pretrain_t2t_cleaned_two.jsonl   # Base 预训练（~1.90B tokens）
│   └── sft_t2t_cleaned_v8_fixed1.jsonl  # SFT Stage 1 对话
│
├── data_stage2/                # Stage 2 身份数据（按场景拆分）
│   ├── sft_identity_qa.jsonl   # 身份 Q&A（50 场景 × ~10 子场景）
│   ├── sft_identity_gentle.jsonl
│   ├── sft_daily_chat.jsonl
│   ├── sft_deep_talk.jsonl
│   ├── sft_emotional.jsonl
│   ├── sft_humor.jsonl
│   ├── sft_creative.jsonl
│   ├── sft_knowledge.jsonl
│   ├── sft_longform.jsonl
│   ├── sft_meme.jsonl
│   ├── sft_sky_nature.jsonl
│   └── dpo_pairs.jsonl
│
├── data_stage2_mixed/          # Stage 2 混合后数据（9:1 = SFT:身份）
│   └── sft_stage2_mixed.jsonl
│
└── log/                        # 设计文档 & 日志
    ├── 2.10_总体方案.md         # 总体技术方案
    ├── 2.10_数据策略.md         # 数据清洗 & 策略
    ├── 2.10_训练日志.md         # 完整训练日志
    └── 柳小乐 (Shayler) — 角色卡 v2.3.md
```

## 训练流程

```
① Base 预训练 (v5.1 · 清洗数据 · 纯文本)
   AdamW lr=3e-4 → 1e-5 | 15000 steps | batch=128
   数据量：~1.90B tokens
   ↓ step_015000.pt

② SFT Stage 1 — 纯对话训练
   AdamW lr=5e-5 → 5e-6 | 13592 steps | batch=128
   ↓ sft_step_014000.pt

③ Stage 2 — 混合身份注入 (9:1)
   Stage 1 数据 163,620 条 + 柳小乐身份 18,180 条
   AdamW lr=1e-5 → 1e-6 | 1421 steps
   ★ sft2_step_001450.pt（最佳）
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 推理（Stage 2 身份模型）
python code/eval.py --checkpoint-dir model --sft2 --step 1450 --sft --temp 0.6 --top-p 0.9

# 推理（Stage 1 纯对话模型）
python code/eval.py --checkpoint-dir model --sft --step 14000 --temp 0.7 --top-p 0.8

# 交互对话
python code/eval.py --checkpoint-dir model --sft2 --step 1450 --sft --temp 0.6 --top-p 0.9 --interactive
```

## 注意事项

- 所有 `.pt` 和 `.jsonl` 文件使用 **Git LFS** 管理，clone 前请安装 `git-lfs`
- 角色设定详见 `log/柳小乐 (Shayler) — 角色卡 v2.3.md`
