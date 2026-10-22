"""导出 S4 的候选值清单（供 LLM 聚类 + 人工 review）。

这是 S4.2 流程的「① 枚举」步骤：纯统计，不涉及任何语义判断。
**只喂取值 + 频次，不喂 query、不喂答案** —— 保证映射表是标注词表的函数，
跟数据划分无关（见 docs/DECISIONS.md D4.3）。

输出
----
out/audit/s1_tool_counts.tsv   每个工具的 train/dev/test 样本数
out/audit/s3_slot_counts.tsv   每个槽位名的出现次数 / 取值种数 / 类型判定
out/audit/value_candidates.tsv 每个 (槽位, 原值) 的频次 —— 建 value_clusters.tsv 的输入

用法
----
python3 -I src/clean/dump_candidates.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Rules, write_tsv
from stages import s1_split, s2_keep_tools, s2_filter, s3_slots

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "macslu", "label")
RULES_DIR = os.path.join(ROOT, "rules")
OUT = os.path.join(ROOT, "out", "audit")


def main():
    rules = Rules(RULES_DIR)
    cfg = rules.config
    os.makedirs(OUT, exist_ok=True)

    # ---- S1 ----
    s1, reasons, raw_n = {}, {}, {}
    for sp in cfg["splits"]:
        s1[sp], reasons[sp], raw_n[sp] = s1_split(sp, DATA, rules)

    print("=== S1 ===")
    for sp in cfg["splits"]:
        print(f"  {sp:6s} 原始 {raw_n[sp]:6d}  S1 后 {len(s1[sp]):6d}  "
              f"丢弃 {dict(reasons[sp])}")

    # ---- S2 ----
    keep_tools, tool_counts = s2_keep_tools(s1["train"], rules)
    thr = cfg["min_tool_train_count"]
    dropped = sum(1 for _r, _q, t, _s in s1["train"] if t not in keep_tools)
    print(f"\n=== S2a 工具阈值 (train >= {thr}) ===")
    print(f"  工具 {len(keep_tools)} 保留 / {len(tool_counts)} 总数；丢弃样本 {dropped}")

    s2 = {sp: s2_filter(s1[sp], keep_tools)[0] for sp in cfg["splits"]}

    # ---- S3 ----
    s3 = {sp: s3_slots(s2[sp], rules) for sp in cfg["splits"]}

    # ---- 统计 ----
    all_slots = collections.Counter()
    slot_values = collections.defaultdict(collections.Counter)
    slot_tools = collections.defaultdict(set)
    slot_train = collections.defaultdict(collections.Counter)
    slot_eval = collections.defaultdict(collections.Counter)
    for sp in cfg["splits"]:
        is_eval = sp in cfg["eval_splits"]
        for _rid, _q, t, pairs in s3[sp]:
            for slot, val in pairs:
                all_slots[slot] += 1
                slot_tools[slot].add(t)
                slot_values[slot][val] += 1
                (slot_eval if is_eval else slot_train)[slot][val] += 1

    # ---- 工具计数表 ----
    rows = []
    for t in sorted(keep_tools, key=lambda x: -tool_counts[x]):
        rows.append({"tool": t,
                     "train": sum(1 for _r, _q, tt, _p in s3["train"] if tt == t),
                     "dev": sum(1 for _r, _q, tt, _p in s3["dev"] if tt == t),
                     "test": sum(1 for _r, _q, tt, _p in s3["test"] if tt == t)})
    write_tsv(os.path.join(OUT, "s1_tool_counts.tsv"), rows,
              ["tool", "train", "dev", "test"])

    # ---- 槽位名表 ----
    minslot = cfg["min_slot_count"]
    ft = cfg["free_text_ratio_threshold"]
    rows = []
    for slot, cnt in all_slots.most_common():
        nval = len(slot_values[slot])
        ratio = nval / cnt
        typ = ("丢弃" if cnt < minslot else
               "自由文本" if ratio > ft else
               "半开放" if ratio > 0.1 else "枚举")
        rows.append({"slot": slot, "count": cnt, "distinct": nval,
                     "ratio": f"{ratio:.3f}", "type": typ,
                     "n_tools": len(slot_tools[slot]),
                     "sample": " / ".join(v for v, _ in slot_values[slot].most_common(5))})
    write_tsv(os.path.join(OUT, "s3_slot_counts.tsv"), rows,
              ["slot", "count", "distinct", "ratio", "type", "n_tools", "sample"])

    print(f"\n=== S3 槽位名 ===")
    print(f"  总计 {len(all_slots)} 个；丢弃(<{minslot}) "
          f"{sum(1 for r in rows if r['type'] == '丢弃')} 个")
    for typ in ("枚举", "半开放", "自由文本"):
        sub = [r for r in rows if r["type"] == typ]
        print(f"  {typ:6s} {len(sub):4d} 个，样本 {sum(r['count'] for r in sub):6d}")

    # ---- 候选值清单（只列需要建表的槽位）----
    rows = []
    for slot, cnt in all_slots.most_common():
        if cnt < minslot or len(slot_values[slot]) / cnt > ft:
            continue                      # 丢弃 或 自由文本，不建表
        for val, vc in slot_values[slot].most_common():
            rows.append({"slot": slot, "raw": val,
                         "train": slot_train[slot].get(val, 0),
                         "eval": slot_eval[slot].get(val, 0),
                         "total": vc})
    write_tsv(os.path.join(OUT, "value_candidates.tsv"), rows,
              ["slot", "raw", "train", "eval", "total"])

    print(f"\n=== 候选值清单 ===")
    print(f"  需建表的 (槽位, 原值) 对：{len(rows)}")
    print(f"  涉及槽位：{len({r['slot'] for r in rows})}")
    print(f"  输出：{os.path.relpath(os.path.join(OUT, 'value_candidates.tsv'), ROOT)}")


if __name__ == "__main__":
    main()
