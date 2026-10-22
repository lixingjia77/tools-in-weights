"""在固定并发下测吞吐：req/s 与 tokens/s 分开报。

为什么不能只报 req/s：P1 和 P2 的每条请求做的功不一样（5647 vs 84 个输入 token），
req/s 把「请求便宜」和「引擎快」混在一起。tokens/s 才是同一把尺子。

三个口径：
  req/s       用户视角：单位时间服务多少条请求
  output tok/s  decode 视角：单位时间生成多少 token（用 ignore_eos 固定输出 32）
  input tok/s  prefill 视角：单位时间处理多少输入 token（开 APC 时是上界，含命中缓存的）

用法
----
python3 -I src/throughput.py --model models/Qwen3-4B --apc 1 --out out/cost/4b_tput_apc1.json
"""
import argparse
import json
import os
import statistics
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(ROOT, "out", "prompts")
os.environ["PATH"] = (os.path.join(ROOT, ".venv", "bin") + os.pathsep
                      + os.environ.get("PATH", ""))


def load_prompts(paradigm, n):
    rows = [json.loads(l) for l in
            open(os.path.join(PROMPTS, f"test_{paradigm}_d.jsonl"), encoding="utf-8")
            if l.strip()]
    step = max(1, len(rows) // n)
    return [r["prompt"] for r in rows[::step]][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--apc", type=int, choices=[0, 1], required=True)
    ap.add_argument("--n", type=int, default=100, help="候选池大小")
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 8, 32, 100])
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    tok_dir = args.model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=10240,
              gpu_memory_utilization=0.85,
              enable_prefix_caching=bool(args.apc),
              max_num_seqs=256, disable_log_stats=True)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True)

    result = {"model": args.model, "apc": bool(args.apc),
              "max_tokens": args.max_tokens, "rows": []}

    for paradigm in ["P1", "P2"]:
        pool = load_prompts(paradigm, args.n)
        # 预热
        llm.generate(pool[:4], sp)

        for c in args.concurrency:
            prompts = pool[:c] if c <= len(pool) else (pool * (c // len(pool) + 1))[:c]
            # APC 下先让前缀进缓存，再计时（稳态）
            if args.apc:
                llm.generate(prompts[:min(4, len(prompts))], sp)

            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            dt = time.perf_counter() - t0

            n_in = sum(len(tok(p, add_special_tokens=False).input_ids) for p in prompts)
            n_out = sum(len(o.token_ids) for r in outs for o in r.outputs)
            row = {
                "paradigm": paradigm, "concurrency": c,
                "wall_s": dt,
                "req_s": len(prompts) / dt,
                "out_tok_s": n_out / dt,
                "in_tok_s": n_in / dt,
                "in_tokens": n_in, "out_tokens": n_out,
            }
            result["rows"].append(row)
            print(f"  {paradigm} c={c:3d}  {dt*1000:8.1f} ms  "
                  f"{row['req_s']:8.2f} req/s  {row['out_tok_s']:9.1f} out-tok/s  "
                  f"{row['in_tok_s']:10.0f} in-tok/s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"→ {os.path.relpath(args.out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
