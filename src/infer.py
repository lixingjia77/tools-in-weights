"""用 vLLM 跑 M1（base 模型）在 P1 / P2 上的推理。

这是「不需要训练」的那一步：拿到 P1 的天花板和 P2 的地板。
如果 P2 的地板不是 ≈0，说明 base 模型本来就能猜，实验不成立（REPORT §2.6）。

用法
----
# 快速试跑（每种范式 32 条）
python3 -I src/infer.py --limit 32

# 全量
python3 -I src/infer.py

# batch size 扫描（只跑吞吐/延迟，不存全量结果）
python3 -I src/infer.py --sweep 8 16 32 64 128 256
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(ROOT, "out", "prompts")
PREDS = os.path.join(ROOT, "out", "preds")
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B")

# vLLM 的 flashinfer 后端要 JIT 编译采样算子，靠 subprocess 调 `ninja`。
# ninja 装在 venv 的 bin 下，不在默认 PATH 里 —— 显式加进去。
_VENV_BIN = os.path.dirname(sys.executable)
if os.path.isdir(_VENV_BIN) and _VENV_BIN not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = _VENV_BIN + ":" + os.environ.get("PATH", "")

MAX_TOKENS = 256


def load_prompts(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def build_llm(model_dir, max_num_seqs, gpu_util, max_model_len):
    from vllm import LLM
    return LLM(model=model_dir,
               max_model_len=max_model_len,
               max_num_seqs=max_num_seqs,
               gpu_memory_utilization=gpu_util,
               dtype="bfloat16",
               disable_log_stats=False)


def run_batch(llm, prompts, max_tokens=MAX_TOKENS):
    """跑一批，返回 (outputs, wall_time, ttfts)"""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    wall = time.perf_counter() - t0

    ttfts, n_out = [], []
    for o in outs:
        n_out.append(len(o.outputs[0].token_ids))
        m = getattr(o, "metrics", None)
        if m is not None and getattr(m, "first_token_time", None) is not None:
            # vLLM 的 metrics 时间是绝对时间戳
            ttfts.append(m.first_token_time - m.arrival_time)
    return outs, wall, ttfts, n_out


def run_guided(llm, rows, tool_schema):
    """用 LLM.chat(tools=...) 做 schema 受限解码。

    注意：这只在 P1 上有意义 —— 对 P2 来说，把工具 schema 交给 grammar
    等同于把 schema 从后门塞给模型，P2 就不再是「无 schema」了。
    见 out/ablation/RESULTS.md §9。
    """
    from vllm import SamplingParams
    from render import SYSTEM_P1, DIRECTIVE
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    msgs = [[{"role": "system", "content": SYSTEM_P1 + "\n\n" + DIRECTIVE},
             {"role": "user", "content": r["query"]}] for r in rows]
    t0 = time.perf_counter()
    outs = llm.chat(msgs, sampling_params=sp, tools=tool_schema,
                    chat_template_kwargs={"enable_thinking": False})
    wall = time.perf_counter() - t0
    return outs, wall, [], [len(o.outputs[0].token_ids) for o in outs]


def do_sweep(model_dir, prompt_files, sizes, limit, gpu_util, max_model_len):
    """扫描 max_num_seqs，看吞吐与延迟。"""
    data = {}
    for p in prompt_files:
        base = os.path.basename(p).replace(".jsonl", "")
        data[base] = load_prompts(p, limit)
    print(f"扫描 max_num_seqs ∈ {sizes}；每种范式各 {limit} 条\n")
    print(f"{'batch':>6s}{'范式':>8s}{'条数':>6s}{'wall(s)':>9s}"
          f"{'条/s':>8s}{'tok/s':>9s}{'TTFT均值(ms)':>13s}")
    for bs in sizes:
        llm = build_llm(model_dir, bs, gpu_util, max_model_len)
        for name, rows in data.items():
            prompts = [r["prompt"] for r in rows]
            outs, wall, ttfts, n_out = run_batch(llm, prompts)
            tot_tok = sum(n_out)
            tt = f"{statistics.mean(ttfts) * 1000:12.1f}" if ttfts else "           -"
            print(f"{bs:6d}{name.split('_')[-1]:>8s}{len(rows):6d}{wall:9.2f}"
                  f"{len(rows)/wall:8.1f}{tot_tok/wall:9.1f}{tt}")
        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="每种范式只跑前 N 条")
    ap.add_argument("--sweep", type=int, nargs="*", default=None,
                    help="扫描这些 max_num_seqs")
    ap.add_argument("--sweep-limit", type=int, default=64)
    ap.add_argument("--freq", type=int, default=100, help="batch 内分块大小")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=10240,
                    help="P1 的 prompt 约 9528 tok（schema 占 9425），必须 ≥ 它 + max_tokens")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--guided", action="store_true",
                    help="用 LLM.chat(tools=...) 做 schema 受限解码（只对 P1 有意义，见 RESULTS §9）")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    inputs = args.inputs or [os.path.join(PROMPTS, f"{sp}_{para}.jsonl")
                             for sp in ("dev", "test") for para in ("P1", "P2")]
    inputs = [p for p in inputs if os.path.exists(p)]
    if not inputs:
        print(f"找不到 prompt 文件，先跑 src/render.py\n期望目录：{PROMPTS}")
        sys.exit(1)

    if args.sweep:
        do_sweep(args.model, inputs, args.sweep, args.sweep_limit, args.gpu_util,
                 args.max_model_len)
        return

    os.makedirs(PREDS, exist_ok=True)
    print(f"模型 {args.model}")
    print(f"max_num_seqs={args.max_num_seqs}  gpu_util={args.gpu_util}  "
          f"max_model_len={args.max_model_len}\n")
    llm = build_llm(args.model, args.max_num_seqs, args.gpu_util, args.max_model_len)

    tool_schema = None
    if args.guided:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from render import load_schema, SYSTEM_P1, DIRECTIVE
        tool_schema = load_schema()
        print(f"受限解码已开启：{len(tool_schema)} 个工具作为 grammar\n")

    for path in inputs:
        name = os.path.basename(path).replace(".jsonl", "")
        rows = load_prompts(path, args.limit)
        t0 = time.perf_counter()
        if args.guided:
            outs, wall, ttfts, n_out = run_guided(llm, rows, tool_schema)
        else:
            prompts = [r["prompt"] for r in rows]
            outs, wall, ttfts, n_out = run_batch(llm, prompts)
        out_path = os.path.join(PREDS, f"{name}{args.suffix}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                f.write(json.dumps({
                    "id": r["id"], "query": r["query"],
                    "output": o.outputs[0].text,
                    "n_tokens": len(o.outputs[0].token_ids),
                    "gold": r["gold"],
                }, ensure_ascii=False) + "\n")
        tt = f"{statistics.mean(ttfts)*1000:.0f}ms" if ttfts else "-"
        print(f"  {name:12s} {len(rows):5d} 条  {wall:7.2f}s  "
              f"{len(rows)/wall:6.1f} 条/s  {sum(n_out)/wall:7.1f} tok/s  "
              f"TTFT {tt}  → {os.path.relpath(out_path, ROOT)}")

    print(f"\n下一步：python3 -I src/eval.py out/preds/dev_P1.jsonl out/preds/dev_P2.jsonl "
          f"out/preds/test_P1.jsonl out/preds/test_P2.jsonl")


if __name__ == "__main__":
    main()
