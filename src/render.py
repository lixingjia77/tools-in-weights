"""渲染 P1 / P2 的 prompt。

P1 = Qwen3 原生工具模板（schema 进 system message 最前面）
P2 = 同一个模板，但 **不传 tools**，system message 里只留格式说明

两者唯一的差别就是 `<tools>...</tools>` 整块。金标、解析器、评测脚本全部共用。

用法
----
python3 -I src/render.py                      # 渲染全部
python3 -I src/render.py --splits dev test    # 只渲染指定 split
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(ROOT, "out", "schema", "tools.json")
CLEANED = os.path.join(ROOT, "out", "cleaned")
OUT = os.path.join(ROOT, "out", "prompts")
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3-0.6B")

SYSTEM_P1 = "你是车载助手。"

# P2 的 system message：删掉 <tools> 块，**保留标准 tool call 的生成说明**
FORMAT_SPEC = (
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)
SYSTEM_P2 = "你是车载助手。\n\n" + FORMAT_SPEC

# 变体 directive：Qwen3 模板里的措辞是 "You may call one or more functions"，
# 对 0.6B 太弱 —— 实测 18% 的样本模型改用自然语言回答而不发 tool call。
# 这个 directive 对称地加在 P1 和 P2 上，保证两臂公平。
DIRECTIVE = "你必须调用工具来完成用户的请求，不要用自然语言直接回答。"


def load_schema():
    with open(SCHEMA, encoding="utf-8") as f:
        s = json.load(f)
    return [s[t] for t in sorted(s)]


def load_split(split):
    with open(os.path.join(CLEANED, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(tok, paradigm, query, tool_list=None, directive=False):
    """构造一条 prompt（已含 add_generation_prompt，可直接喂模型）。"""
    extra = ("\n\n" + DIRECTIVE) if directive else ""
    if paradigm == "P1":
        msgs = [{"role": "system", "content": SYSTEM_P1 + extra},
                {"role": "user", "content": query}]
        return tok.apply_chat_template(msgs, tools=tool_list, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    elif paradigm == "P2":
        msgs = [{"role": "system", "content": SYSTEM_P2 + extra},
                {"role": "user", "content": query}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    raise ValueError(paradigm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--paradigms", nargs="*", default=None)
    ap.add_argument("--directive", action="store_true",
                    help="在 system message 里加「必须调用工具」指令（变体）")
    ap.add_argument("--suffix", default=None, help="输出文件名后缀")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tool_list = load_schema()
    suffix = args.suffix if args.suffix is not None else ("_d" if args.directive else "")
    print(f"工具 {len(tool_list)} 个，来自 out/schema/tools.json"
          + (f"；变体 suffix={suffix}" if suffix else ""))

    os.makedirs(OUT, exist_ok=True)
    splits = args.splits or ["train", "dev", "test"]
    for split in splits:
        # train 只渲染 P2（M2 的训练数据）；dev/test 两种都渲染（评测）
        paradigms = args.paradigms or (["P2"] if split == "train" else ["P1", "P2"])
        rows = load_split(split)
        for para in paradigms:
            path = os.path.join(OUT, f"{split}_{para}{suffix}.jsonl")
            n = 0
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    p = build_prompt(tok, para, r["query"], tool_list,
                                     directive=args.directive)
                    rec = {"id": r["id"], "query": r["query"], "prompt": p,
                           "gold": {"tool": r["tool"], "arguments": r["arguments"]}}
                    if "seen" in r:
                        rec["seen"] = r["seen"]
                        rec["unseen_slots"] = r["unseen_slots"]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            print(f"  {split:6s} {para}  {n:6d} 条 → {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
