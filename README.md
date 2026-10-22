# tools-in-weights

**Schema-free tool calling: can tool definitions be trained into model weights, so that tool calling works with no tool schema in context?**

This repository is a reproducible, single-GPU (RTX 4090, 24 GB) experiment that answers that question
on 41 automotive-cabin tools, and measures what the answer costs.

**Terminology.** The inference setting is usually called **schema-free (schema-less) tool calling**.
The underlying question is whether a tool schema can become **parametric knowledge** instead of
**contextual knowledge** — the same distinction the RAG literature draws for facts.
Hence the repository name: *tools in weights*.

[中文说明](README.zh-CN.md) · [Blog post (中文)](docs/BLOG.md) · [Full report (中文)](docs/REPORT.md) · [Decision log (中文)](docs/DECISIONS.md)

---

## TL;DR

Two paradigms, one variable — whether the tool list is in the input:

| | Tool list in prompt | Output format |
|---|---|---|
| **P1** | full JSON schema (~5563 tok) | `<tool_call>{"name":...,"arguments":{...}}</tool_call>` |
| **P2** | **none** (84 tok total) | identical |

| Model | Trained on | Eval on P1 (schema in prompt) | Eval on P2 (no schema) |
|---|---|---:|---:|
| Qwen3-4B, untrained | — | 46.52% | **0.00%** |
| Qwen3-4B + LoRA | P2 (no schema) | 61.82% | **81.76%** |
| Qwen3-0.6B + LoRA | P2 (no schema) | 48.22% | 75.73% |
| Qwen3-0.6B + LoRA | **P1 (with schema)** | **77.59%** | **0.00%** |

*Call Acc on the held-out test split (n=647), greedy decoding, single seed.*

**Four findings:**

1. **It works.** Tool calling with no schema in context goes from 0.00% to 81.76% with 4B LoRA,
   using 0.8% trainable parameters and ~18 minutes on one 4090.
2. **It does not win on quality.** Trained and evaluated with the schema in the prompt,
   the same 0.6B model reaches 77.59%; trained and evaluated without it, 75.73%.
   The 1.85-point difference is not significant (95% CI ±2.46, McNemar p=0.175) and flips sign on dev.
3. **Transfer is one-way.** No-schema training transfers to the with-schema setting (48.22%);
   with-schema training collapses to **0.00%** when the schema is removed. The format is still valid
   (97.8%) — the tool *vocabulary* collapses (`播放歌`, `get_temperature`). Training with the schema
   in context teaches copying, not remembering.
4. **The gain is in cost, and prefix caching only recovers part of it.**
   Per request: −5563 input tokens; cold-start TTFT 359.6 ms → 13.3 ms (27×).
   With prefix caching, single-user latency is nearly parity (379.7 vs 343.7 ms for 32 tokens, 1.10×),
   **but output throughput at concurrency 100 is 1300.5 vs 5764.2 tok/s (4.4×)**; without caching it is
   73.4 vs 3463.4 tok/s (47×). KV cache: 0.485 vs 0.007 GiB (67×). Prefix caching removes prefill,
   not decode-time KV reads.

---

## Repository layout

```
.
├── docs/
│   ├── BLOG.md              # Chinese blog post (~30 min read)
│   ├── REPORT.md            # Full technical report (Chinese)
│   └── DECISIONS.md         # Design decisions and their rationale (Chinese)
├── results/                 # Archived eval outputs, cost measurements, training logs
├── assets/                  # Figures used in docs (SVG)
├── rules/                   # Human-reviewed data rules (TSV/JSON), diffable
├── src/
│   ├── clean/               # Data cleaning S1–S6 + parameter contract
│   ├── render.py            # P1 / P2 prompt renderer
│   ├── train.py             # M2 / M2' training (full fine-tune & LoRA)
│   ├── merge_lora.py        # Fix key names after LoRA merge
│   ├── infer.py             # vLLM batch inference
│   ├── parse.py             # Shared output parser
│   ├── eval.py              # Evaluation (lenient/strict Call Acc + slices)
│   ├── cost.py              # TTFT / KV / prefix-cache measurement
│   ├── throughput.py        # Fixed-output, concurrency-swept req/s and tok/s
│   └── plot_throughput.py   # Generates assets/throughput.svg
├── data/macslu/label/       # MAC-SLU annotation text (Apache-2.0)
├── requirements.txt
└── LICENSE
```

Generated artifacts (`out/`, `models/`, `.venv/`, `logs/`) are git-ignored; everything in `results/`
was produced by the scripts above and can be regenerated.

---

## Quickstart

### 0. Environment

Python 3.12, CUDA 13.0. ~14 GB VRAM is enough for 4B LoRA training.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### 1. Data

The label files used here are already in `data/macslu/label/` (6.2 MB, text only).
To re-download from the source:

```bash
huggingface-cli download Gatsby1984/MAC_SLU --repo-type dataset --include "label/*"
```

The upstream dataset is ~1.9 GB and includes audio; this experiment only uses the annotation JSONL.

### 2. Clean + build schema

```bash
python3 -I src/clean/build.py
```

Produces `out/cleaned/{train,dev,test}.jsonl`, `out/schema/tools.json`, and `out/manifest.json`
(inputs/outputs/counts/sha256 for every step). The build is byte-deterministic.

### 3. Render prompts

```bash
.venv/bin/python -I src/render.py --directive                          # dev/test: P1 + P2
.venv/bin/python -I src/render.py --splits train --paradigms P1 --directive  # train: P1 (control arm only)
```

### 4. Train

```bash
# main arm: train on P2 (no schema), 4B LoRA — ~18 min on a 4090
.venv/bin/python -I src/train.py --model models/Qwen3-4B --mode lora \
    --out models/Qwen3-4B-M2 --epochs 3 --batch 4 --grad-accum 8 --grad-ckpt

# control arm: train on P1 (schema in prompt) — ~5 h/epoch on a 4090
.venv/bin/python -I src/train.py --model models/Qwen3-4B --mode lora \
    --data out/prompts/train_P1_d.jsonl --out models/Qwen3-4B-M2p --epochs 3 \
    --max-len 6144 --batch 1 --grad-accum 32 --grad-ckpt
```

Each epoch is saved as a complete, vLLM-loadable model directory (LoRA merged).

### 5. Inference + evaluation

```bash
.venv/bin/python -I src/infer.py --model models/Qwen3-4B-M2/epoch3 --suffix _m2 \
    --inputs out/prompts/test_P1_d.jsonl out/prompts/test_P2_d.jsonl

python3 -I src/eval.py out/preds/test_P1_d_m2.jsonl out/preds/test_P2_d_m2.jsonl
```

### 6. Cost measurement

```bash
.venv/bin/python -I src/cost.py --model models/Qwen3-4B --apc 0 --n 100 --out out/cost/4b_apc0.json
.venv/bin/python -I src/cost.py --model models/Qwen3-4B --apc 1 --n 100 --out out/cost/4b_apc1.json
```

---

## How the experiment stays honest

- **One variable.** P1 and P2 differ only by the `<tools>` block. Same gold, same parser, same scorer.
- **Held-out protocol.** dev (n=588) is used only to pick the epoch; every number in the tables is test
  (n=647). The final model touches test once.
- **A control arm for attribution.** `M2′` trains on P1. Without it, “M2/P2 beats M1/P1” conflates
  *training* with *where the schema lives*.
- **A noise floor.** Greedy decoding is **not** bit-reproducible under vLLM: 4.6% of outputs differ
  between identical runs, ±0.5 pt on metrics. Re-training the same config changes dev Call Acc by 2.2 pt.
  Differences below ~2 pt are not interpreted.
- **Deterministic data build.** Two identical runs of `build.py` produce byte-identical outputs
  (verified via sha256 in the manifest).

---

## Limitations

- 41 closed tools, single turn, Chinese only. No extrapolation to open-ended tool sets.
- One seed (42). No general-capability regression test.
- Latency numbers are offline, single-GPU; ratios are informative, absolutes are not production numbers.
- The 4B control arm (`M2′`, trained on P1) was **not run to completion** — P1 training is ~5 h/epoch on a
  4090 because every sample carries the full schema. The parity claim therefore rests on 0.6B.

---

## Data and license

- Code: **MIT** (see `LICENSE`).
- Data: **MAC-SLU** ([Gatsby1984/MAC_SLU](https://huggingface.co/datasets/Gatsby1984/MAC_SLU)),
  licensed **Apache-2.0**. Only the annotation JSONL files are redistributed here (6.2 MB); the audio is not.
- Model weights are not included. Training uses [Qwen3](https://github.com/QwenLM/Qwen3) by Alibaba.

If you use the dataset, please cite the original MAC-SLU work.

---

## Related work (and what this repo adds)

Training tools into parameters is not new — tool names as special tokens (e.g. ToolkenGPT),
output-side function tokens, and function-calling SFT (Gorilla / ToolLLM) all explore related ideas.
What this repository adds is a **controlled 3×2 comparison** (train on P1/P2 × evaluate on P1/P2)
plus a **cost accounting** that separates prefill, decode and KV memory.

The long-form Chinese write-up with every design decision is in [docs/REPORT.md](docs/REPORT.md);
the shorter narrative version is [docs/BLOG.md](docs/BLOG.md).
