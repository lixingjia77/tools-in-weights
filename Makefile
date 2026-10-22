PY      := .venv/bin/python
MODEL   ?= models/Qwen3-4B
SUFFIX  ?= _m2
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

.PHONY: help setup data build verify prompts train-06b-full train-06b-lora train-4b-lora infer eval cost plot clean

help:
	@echo "make setup        创建 .venv 并安装依赖"
	@echo "make data         打印 MAC-SLU 下载命令"
	@echo "make build        清洗数据 + 生成 schema（确定性）"
	@echo "make prompts      渲染 P1/P2 prompt"
	@echo "make train-4b-lora   4B LoRA，用 P2（无清单）训练"
	@echo "make infer MODEL=... SUFFIX=...   在 test 上推理 P1/P2"
	@echo "make eval MODEL=...               解析 + 报告指标"
	@echo "make cost         测量 TTFT / 吞吐 / 前缀缓存"
	@echo "make plot         由测量结果生成 assets/throughput.svg"

setup:
	uv venv .venv
	uv pip install --python $(PY) -r requirements.txt

data:
	@echo 'huggingface-cli download Gatsby1984/MAC_SLU --repo-type dataset --include "label/*"'
	@echo '仓库已带 data/macslu/label/，通常不需要重新下载。'

build:
	python3 -I src/clean/build.py

verify:
	python3 -I src/clean/verify.py

prompts:
	$(PY) -I src/render.py --directive
	$(PY) -I src/render.py --splits train --paradigms P1 --directive

train-06b-full:
	$(PY) -I src/train.py --model models/Qwen3-0.6B --mode full \
	    --out models/Qwen3-0.6B-M2 --epochs 3

train-06b-lora:
	$(PY) -I src/train.py --model models/Qwen3-0.6B --mode lora \
	    --out models/Qwen3-0.6B-M2 --epochs 3

train-4b-lora:
	$(PY) -I src/train.py --model models/Qwen3-4B --mode lora \
	    --out models/Qwen3-4B-M2 --epochs 3 --batch 4 --grad-accum 8 --grad-ckpt

infer:
	$(PY) -I src/infer.py --model $(MODEL) --suffix $(SUFFIX) \
	    --inputs out/prompts/test_P1_d.jsonl out/prompts/test_P2_d.jsonl

eval:
	python3 -I src/eval.py out/preds/test_P1_d$(SUFFIX).jsonl out/preds/test_P2_d$(SUFFIX).jsonl

cost:
	$(PY) -I src/cost.py --model models/Qwen3-4B --apc 0 --n 100 --out out/cost/4b_apc0.json
	$(PY) -I src/cost.py --model models/Qwen3-4B --apc 1 --n 100 --out out/cost/4b_apc1.json

plot:
	python3 -I src/plot_throughput.py

clean:
	rm -rf out/cost out/preds
