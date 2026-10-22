"""导出 S5.1 描述生成的证据清单。

写 description 需要知道：这个工具/参数是干什么的。
证据来源分两类：

  结构化证据（槽位名、取值、出现频次）—— 无争议
  示例 query —— **只用 train split**，不用 eval。
      理由：description 属于 schema（P1 在推理时给定的任务定义）。
      用 train 语料写描述，等同于在训练集上做 prompt engineering，是标准做法；
      用 eval 语料写则会携带评测集信息，构成泄漏。

输出
----
out/audit/desc_tools.tsv  每个工具：参数列表 + train 高频操作 + train 示例 query
out/audit/desc_slots.tsv  每个槽位：出现次数 / 覆盖工具数 / 高频取值

用法
----
python3 -I src/clean/dump_descriptions.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Rules, write_tsv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULES_DIR = os.path.join(ROOT, "rules")
OUT = os.path.join(ROOT, "out", "audit")
CLEANED = os.path.join(ROOT, "out", "cleaned")


def load(name):
    with open(os.path.join(CLEANED, f"{name}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    rules = Rules(RULES_DIR)
    cfg = rules.config
    os.makedirs(OUT, exist_ok=True)

    train = load("train")
    dev, test = load("dev"), load("test")

    schema_path = os.path.join(ROOT, "out", "schema", "tools.json")
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)

    # ---------- 工具级证据 ----------
    by_tool = collections.defaultdict(list)
    for r in train:
        by_tool[r["tool"]].append(r)

    rows = []
    for tool in sorted(schema, key=lambda t: -len(by_tool[t])):
        samples = by_tool[tool]
        props = schema[tool]["function"]["parameters"]["properties"]
        ops = collections.Counter(r["arguments"].get("操作") for r in samples
                                  if r["arguments"].get("操作"))
        # 示例 query 取 5 条，尽量覆盖不同的参数组合
        seen, examples = set(), []
        for r in samples:
            sig = tuple(sorted(r["arguments"]))
            if sig in seen:
                continue
            seen.add(sig)
            examples.append(r["query"])
            if len(examples) >= 5:
                break
        rows.append({
            "tool": tool,
            "train_n": len(samples),
            "params": " ".join(props.keys()),
            "top_seen": " | ".join(f"{s}×{c}" for s, c in
                                   collections.Counter(
                                       s for r in samples for s in r["arguments"]
                                   ).most_common(4)),
            "top_ops": " ".join(o for o, _ in ops.most_common(4)),
            "examples": " ‖ ".join(examples),
        })
    write_tsv(os.path.join(OUT, "desc_tools.tsv"), rows,
              ["tool", "train_n", "params", "top_seen", "top_ops", "examples"])

    # ---------- 槽位级证据 ----------
    val_ct = collections.defaultdict(collections.Counter)
    slot_tools = collections.defaultdict(set)
    for r in train:
        for slot, val in r["arguments"].items():
            val_ct[slot][val] += 1
            slot_tools[slot].add(r["tool"])

    # 出现在 schema 里的槽位（就是需要写描述的）
    schema_slots = collections.Counter()
    for tool, spec in schema.items():
        for slot in spec["function"]["parameters"]["properties"]:
            schema_slots[slot] += 1

    rows = []
    for slot, n_tool in schema_slots.most_common():
        vals = val_ct[slot]
        n = sum(vals.values())
        rows.append({
            "slot": slot,
            "in_tools": n_tool,
            "train_n": n,
            "distinct": len(vals),
            "has_enum": "Y" if "enum" in
                        schema[next(t for t in schema
                                    if slot in schema[t]["function"]["parameters"]["properties"])
                               ]["function"]["parameters"]["properties"][slot] else "N",
            "top_values": " | ".join(f"{v}×{c}" for v, c in vals.most_common(12)),
        })
    write_tsv(os.path.join(OUT, "desc_slots.tsv"), rows,
              ["slot", "in_tools", "train_n", "distinct", "has_enum", "top_values"])

    print(f"=== 需要写描述的对象 ===")
    print(f"  工具：{len(schema)} 个（另有工具级 description 也需要写，当前是占位符）")
    print(f"  槽位：{len(schema_slots)} 个")
    print(f"\n  out/audit/desc_tools.tsv")
    print(f"  out/audit/desc_slots.tsv")
    print(f"\n=== 槽位清单（出现工具数 / train 次数 / 取值种数 / 是否 enum）===")
    for r in rows[:45]:
        print(f"  {r['slot']:12s} 工具{r['in_tools']:3d}  n={r['train_n']:5d}  "
              f"取值{r['distinct']:4d}  enum={r['has_enum']}")


if __name__ == "__main__":
    main()
