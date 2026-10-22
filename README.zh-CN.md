# tools-in-weights：把工具训进权重

**Schema-free tool calling：能不能把工具定义训进模型权重，让「上下文里没有任何 tool schema」的模型依然会调工具？**

这个仓库给出一个可复现的单卡（RTX 4090, 24 GB）实验：41 个车载工具、7993 条训练样本，
并且把「这样做换来了什么」算清楚。

**术语**：这个推理设置通常叫 **schema-free（schema-less）tool calling**。
它背后的问题是：工具 schema 能不能成为**参数化知识（parametric knowledge）**，
而不是**上下文知识（contextual knowledge）**——这是 RAG 文献里对「事实」的同一个二分。
仓库名 *tools in weights* 就是这个意思。

[English](README.md) · [博客版](docs/BLOG.md) · [完整报告](docs/REPORT.md) · [决策记录](docs/DECISIONS.md)

---

## 一句话结论

同一批数据、同一个模型，只改一件事：**工具清单在不在输入里**。

| | 输入里的工具清单 | 输出格式 |
|---|---|---|
| **P1** | 完整 JSON schema（约 5563 tok） | `<tool_call>{"name":...,"arguments":{...}}</tool_call>` |
| **P2** | **没有**（全文 84 tok） | 完全相同 |

| 模型 | 训练数据 | 评 P1（有清单） | 评 P2（无清单） |
|---|---|---:|---:|
| Qwen3-4B，不训练 | — | 46.52% | **0.00%** |
| Qwen3-4B + LoRA | P2（无清单） | 61.82% | **81.76%** |
| Qwen3-0.6B + LoRA | P2（无清单） | 48.22% | 75.73% |
| Qwen3-0.6B + LoRA | **P1（有清单）** | **77.59%** | **0.00%** |

*Call Acc，test（n=647），greedy，单 seed。*

**四条结论：**

1. **能训进去。** 无清单的工具调用从 0.00% 到 81.76%；只用了 0.8% 的可训练参数，单卡约 18 分钟。
2. **质量上并没有赢。** 同为 0.6B，训练和评测都给清单是 77.59%，都不给是 75.73%；
   差 1.85 个点，不显著（95% CI ±2.46，McNemar p=0.175），且在 dev 上方向相反。
3. **迁移是单向的。** 无清单训练能迁移到有清单（48.22%）；有清单训练拿掉清单直接 **0.00%**。
   格式没崩（97.8% 合法），是**词表崩**（工具名退回 `播放歌` / `get_temperature`）。
   清单在上下文里时，模型学到的是照抄，不是记忆。
4. **收益在成本，而前缀缓存只能补回一部分。** 每请求少 5563 token；冷启首字延迟 359.6 → 13.3 ms（27×）。
   开前缀缓存后单用户几乎打平（固定输出 32 token：379.7 vs 343.7 ms，1.10×），
   **但 100 并发下输出吞吐 1300.5 vs 5764.2 tok/s（4.4×）**；不开缓存则是 73.4 vs 3463.4 tok/s（47×）。
   KV 显存 0.485 vs 0.007 GiB（67×）。前缀缓存省的是 prefill，不是 decode 时读 KV 的带宽与显存。

---

## 仓库结构

```
.
├── docs/
│   ├── BLOG.md              # 博客版（约 30 分钟）
│   ├── REPORT.md            # 完整技术报告
│   └── DECISIONS.md         # 每个决策的「为什么」
├── results/                 # 归档的评测输出、成本测量、训练日志
├── assets/                  # 博客/报告用的图（SVG）
├── rules/                   # 人工 review 的数据规则（TSV/JSON）
├── src/                     # 清洗 / 渲染 / 训练 / 推理 / 评测 / 成本 / 吞吐测量
├── data/macslu/label/       # MAC-SLU 标注文本（Apache-2.0）
├── requirements.txt
└── LICENSE
```

`out/`、`models/`、`.venv/`、`logs/` 都是生成物，已 gitignore；`results/` 里的东西都可以用脚本重新生成。

---

## 快速开始

### 0. 环境

Python 3.12 / CUDA 13.0。4B LoRA 训练峰值显存约 14 GB。

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### 1. 数据

仓库已带 `data/macslu/label/`（6.2 MB，纯文本）。要从源头重新下载：

```bash
huggingface-cli download Gatsby1984/MAC_SLU --repo-type dataset --include "label/*"
```

上游数据集约 1.9 GB（含音频），本实验只用标注 JSONL。

### 2. 清洗 + 生成 schema

```bash
python3 -I src/clean/build.py
```

产出 `out/cleaned/{train,dev,test}.jsonl`、`out/schema/tools.json`、`out/manifest.json`
（每一步的输入/输出/计数/sha256）。**两次运行逐字节一致。**

### 3. 渲染 prompt

```bash
.venv/bin/python -I src/render.py --directive                               # dev/test：P1 + P2
.venv/bin/python -I src/render.py --splits train --paradigms P1 --directive  # train：P1（只给对照臂用）
```

### 4. 训练

```bash
# 主臂：用 P2（无清单）训练，4B LoRA —— 4090 上约 18 分钟
.venv/bin/python -I src/train.py --model models/Qwen3-4B --mode lora \
    --out models/Qwen3-4B-M2 --epochs 3 --batch 4 --grad-accum 8 --grad-ckpt

# 对照臂：用 P1（有清单）训练 —— 4090 上约 5 小时/epoch
.venv/bin/python -I src/train.py --model models/Qwen3-4B --mode lora \
    --data out/prompts/train_P1_d.jsonl --out models/Qwen3-4B-M2p --epochs 3 \
    --max-len 6144 --batch 1 --grad-accum 32 --grad-ckpt
```

每个 epoch 都会存成完整、可直接用 vLLM 加载的模型目录（LoRA 已 merge）。

### 5. 推理 + 评测

```bash
.venv/bin/python -I src/infer.py --model models/Qwen3-4B-M2/epoch3 --suffix _m2 \
    --inputs out/prompts/test_P1_d.jsonl out/prompts/test_P2_d.jsonl

python3 -I src/eval.py out/preds/test_P1_d_m2.jsonl out/preds/test_P2_d_m2.jsonl
```

### 6. 成本测量

```bash
.venv/bin/python -I src/cost.py --model models/Qwen3-4B --apc 0 --n 100 --out out/cost/4b_apc0.json
.venv/bin/python -I src/cost.py --model models/Qwen3-4B --apc 1 --n 100 --out out/cost/4b_apc1.json
```

---

## 实验为什么可信

- **单一变量**：P1 与 P2 只差 `<tools>` 块，金标、解析器、评分脚本完全相同。
- **留出协议**：dev（588 条）只用于选 epoch；表里所有数字都来自 test（647 条），最终模型只跑一次 test。
- **归因对照臂**：`M2′` 用 P1 训练。没有它，「M2/P2 高于 M1/P1」会把「训练」和「清单位置」混在一起。
- **噪声地板**：greedy 在 vLLM 下**不是**逐字节可复现——同输入两次运行 4.6% 输出不同，指标 ±0.5 pt；
  同配置重训差 2.2 pt。**小于 2 pt 的差异不做解释。**
- **确定性构建**：`build.py` 两次运行逐字节一致（manifest 里有 sha256）。

---

## 局限

- 41 个封闭工具、单轮、中文；不外推到开放工具集。
- 单 seed（42）；没有通用能力回归测试。
- 延迟是离线单卡测量，倍率可参考，绝对值不是生产数字。
- 4B 的对照臂（用 P1 训练）**未跑完**：P1 训练每条样本带 5673 token，4090 上约 5 小时/epoch。
  因此「打平」目前由 0.6B 支撑。

---

## 数据与许可

- 代码：**MIT**（见 `LICENSE`）。
- 数据：**MAC-SLU**（[Gatsby1984/MAC_SLU](https://huggingface.co/datasets/Gatsby1984/MAC_SLU)），
  **Apache-2.0**。本仓库只重新分发标注 JSONL（6.2 MB），不含音频。
- 模型权重未包含在仓库中。训练使用 [Qwen3](https://github.com/QwenLM/Qwen3)。

使用数据集请引用 MAC-SLU 原工作。

---

## 与既有工作的关系

「把工具变成可训练参数」不是新想法：工具名做成特殊 token（ToolkenGPT 一脉）、输出侧 function token、
function-calling SFT（Gorilla / ToolLLM 一脉）都探索过相关工作。
本仓库的增量是**受控的 3×2 对比**（训练用 P1/P2 × 评测用 P1/P2）与**把成本拆成 prefill / decode / KV 三项**。

每个设计决策的完整理由见 [docs/REPORT.md](docs/REPORT.md)；更轻松的叙述版本见 [docs/BLOG.md](docs/BLOG.md)。
