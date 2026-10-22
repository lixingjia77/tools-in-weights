"""测量 P1 / P2 在推理侧的成本：单请求延迟（TTFT 代理）与批量吞吐。

两个自变量：
  1. 输入里有没有 schema（P1 vs P2）
  2. 开不开前缀缓存（vLLM APC）

为什么要分开跑两个进程：`enable_prefix_caching` 是引擎构造参数，切换必须重建引擎，
而 vLLM 不会把显存完全还回来。所以本脚本一次只跑一种 APC 设置，由外层循环跑两次。

指标
----
latency  : 顺序发单请求，`max_tokens=1`，端到端耗时 ≈ prefill + 1 个 decode step
           （decode 步两边相同，差异可归因于 prefill）
throughput: 一次 `generate` 提交全部 N 条，测总墙钟 → 请求/s、prefill token/s

用法
----
python3 -I src/cost.py --model models/Qwen3-4B --apc 0 --n 100 --out out/cost/4b_apc0.json
python3 -I src/cost.py --model models/Qwen3-4B --apc 1 --n 100 --out out/cost/4b_apc1.json
"""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(ROOT, "out", "prompts")

# vLLM 的 flashinfer 后端 JIT 编译时要调 ninja，它在 venv/bin 下、不在默认 PATH 里
os.environ["PATH"] = (os.path.join(ROOT, ".venv", "bin") + os.pathsep
                      + os.environ.get("PATH", ""))


def load_prompts(paradigm, n):
    path = os.path.join(PROMPTS, f"test_{paradigm}_d.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    # 等间隔抽样：不依赖前 N 条的顺序
    step = max(1, len(rows) // n)
    return [r["prompt"] for r in rows[::step][:n]], rows


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--apc", type=int, choices=[0, 1], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tag = f"{os.path.basename(args.model)}  apc={args.apc}"
    print(f"加载 {tag} …", flush=True)
    llm = LLM(model=args.model, dtype="bfloat16",
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util,
              enable_prefix_caching=bool(args.apc),
              max_num_seqs=256,
              disable_log_stats=True)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    results = {"model": args.model, "apc": bool(args.apc),
               "n": args.n, "max_tokens": args.max_tokens, "paradigms": {}}

    for paradigm in ["P1", "P2"]:
        prompts, _ = load_prompts(paradigm, args.n)

        # 预热：用最后一条，避免污染待测样本的缓存状态
        for _ in range(3):
            llm.generate([prompts[-1]], sp)

        # --- 单请求延迟（顺序）---
        lat = []
        for p in prompts:
            t0 = time.perf_counter()
            llm.generate([p], sp)
            lat.append((time.perf_counter() - t0) * 1000)  # ms

        # --- 批量吞吐 ---
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        batch_s = time.perf_counter() - t0

        n_tok = sum(len(llm.get_tokenizer()(p, add_special_tokens=False).input_ids)
                    for p in prompts)
        results["paradigms"][paradigm] = {
            "n": len(prompts),
            "prompt_tokens_mean": n_tok / len(prompts),
            "latency_ms": {
                "first": lat[0],
                "mean": statistics.mean(lat),
                "median": statistics.median(lat),
                "p90": pct(lat, 0.90),
                "min": min(lat), "max": max(lat),
            },
            "batch": {
                "seconds": batch_s,
                "req_per_s": len(prompts) / batch_s,
                "prompt_tok_per_s": n_tok / batch_s,
                "ms_per_req": batch_s * 1000 / len(prompts),
            },
        }
        r = results["paradigms"][paradigm]
        print(f"  {paradigm}: prompt {r['prompt_tokens_mean']:.0f} tok | "
              f"延迟 median {r['latency_ms']['median']:.1f} ms "
              f"p90 {r['latency_ms']['p90']:.1f} | "
              f"批量 {r['batch']['req_per_s']:.1f} req/s "
              f"({r['batch']['ms_per_req']:.2f} ms/req)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"→ {os.path.relpath(args.out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
