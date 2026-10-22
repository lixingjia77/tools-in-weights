"""S4 验收：覆盖率与集中度。

按 REPORT §2.4.3「验收指标」：
  覆盖率 —— 建表用 train，报数用 dev+test
  集中度 —— 合并前后每个 (工具,槽位) 的 top-1 占比

输出
----
out/audit/normalize_report.tsv   每个 (槽位) 的合并前后对比
out/audit/tool_slot_enum.tsv     每个 (工具,槽位) 的归一化前后取值种数
out/audit/unmapped_eval.tsv      **eval 中保持原形的值** —— 人工复查是否漏了簇

用法
----
python3 -I src/clean/verify.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Rules, write_tsv
from stages import s1_split, s2_keep_tools, s2_filter, s3_slots, s4_values

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "macslu", "label")
RULES_DIR = os.path.join(ROOT, "rules")
OUT = os.path.join(ROOT, "out", "audit")


def build_all(rules):
    """跑完 S1–S4。返回 (s4_rows, s3_rows, s1, keep_tools)

    s3_rows 是归一化前的状态，s4_rows 是归一化后。两者都要用于对比。
    """
    cfg = rules.config
    s1 = {sp: s1_split(sp, DATA, rules) for sp in cfg["splits"]}
    keep_tools, _counts = s2_keep_tools(s1["train"][0], rules)
    s3 = {}
    for sp in cfg["splits"]:
        kept, _dropped = s2_filter(s1[sp][0], keep_tools)
        s3[sp] = s3_slots(kept, rules)
    s4 = {sp: s4_values(s3[sp], rules)[0] for sp in cfg["splits"]}
    return s4, s3, s1, keep_tools


def top1(counter):
    if not counter:
        return 0.0
    return counter.most_common(1)[0][1] / sum(counter.values())


def main():
    rules = Rules(RULES_DIR)
    cfg = rules.config
    os.makedirs(OUT, exist_ok=True)
    data, s3, s1, keep_tools = build_all(rules)

    # ---------- 全局：合并前后 ----------
    pre_slot = collections.defaultdict(collections.Counter)
    post_slot = collections.defaultdict(collections.Counter)
    for sp in cfg["splits"]:
        for _r, _q, _t, pairs in s3[sp]:
            for slot, val in pairs:
                pre_slot[slot][val] += 1
        for _r, _q, _t, pairs in data[sp]:
            for slot, val in pairs:
                post_slot[slot][val] += 1

    rows = []
    for slot in sorted(post_slot, key=lambda s: -sum(post_slot[s].values())):
        pre, post = pre_slot[slot], post_slot[slot]
        rows.append({
            "slot": slot,
            "samples": sum(post.values()),
            "pre_distinct": len(pre),
            "post_distinct": len(post),
            "pre_top1": f"{top1(pre):.3f}",
            "post_top1": f"{top1(post):.3f}",
            "distinct_cut": f"{100 * (1 - len(post) / max(len(pre), 1)):.1f}%",
        })
    write_tsv(os.path.join(OUT, "normalize_report.tsv"), rows,
              ["slot", "samples", "pre_distinct", "post_distinct",
               "pre_top1", "post_top1", "distinct_cut"])

    print("=== 合并前后（按样本数降序，前 20）===")
    print(f"{'槽位':12s}{'样本':>7s}{'前取值':>7s}{'后取值':>7s}{'降幅':>8s}{'前top1':>8s}{'后top1':>8s}")
    for r in rows[:20]:
        print(f"{r['slot']:12s}{r['samples']:7d}{r['pre_distinct']:7d}"
              f"{r['post_distinct']:7d}{r['distinct_cut']:>8s}"
              f"{r['pre_top1']:>8s}{r['post_top1']:>8s}")

    # ---------- 槽位类型（自由文本 / 其余）----------
    ft = cfg["free_text_ratio_threshold"]
    slot_type = {}
    for slot, c in pre_slot.items():
        n = sum(c.values())
        slot_type[slot] = "自由文本" if (n and len(c) / n > ft) else "可建表"

    # ---------- 表覆盖率（验收指标①）----------
    print("\n=== 表覆盖率（验收指标①：eval 上应 ≥95%）===")
    cov = {}
    for sp in cfg["splits"]:
        tot = in_map = enum_tot = enum_in = 0
        for _r, _q, _t, pairs in s3[sp]:
            for slot, val in pairs:
                tot += 1
                hit = (slot, val) in rules.value_map
                in_map += hit
                if slot_type.get(slot) == "可建表":
                    enum_tot += 1
                    enum_in += hit
        cov[sp] = enum_in / max(enum_tot, 1)
        print(f"  {sp:6s} 全部 {tot:6d} 进表 {in_map:6d} {in_map/max(tot,1)*100:5.1f}%   "
              f"可建表 {enum_tot:6d} 进表 {enum_in:6d} {cov[sp]*100:5.1f}%")
    ev_cov = min(cov[sp] for sp in cfg["eval_splits"])
    print(f"  → eval 最小覆盖率 {ev_cov*100:.1f}%  "
          f"{'达标' if ev_cov >= cfg['value_map_coverage_target'] else '⚠️ 未达 95%，需补簇'}")

    # ---------- 合并覆盖率（在 S3 输出上算，即归一化前）----------
    print("\n=== 合并覆盖率（可建表槽位）===")
    for sp in cfg["splits"]:
        m = n = 0
        for _r, _q, _t, pairs in s3[sp]:
            for slot, val in pairs:
                if slot_type.get(slot) == "自由文本":
                    continue
                n += 1
                if rules.value_map.get((slot, val), val) != val:
                    m += 1
        print(f"  {sp:6s} 可建表槽位实例 {n:7d}  被合并 {m:6d}  {m / max(n,1) * 100:5.1f}%")

    # ---------- 每个 (工具,槽位) 的取值种数 ----------
    pre_ts = collections.defaultdict(collections.Counter)
    post_ts = collections.defaultdict(collections.Counter)
    for sp in cfg["splits"]:
        for _r, _q, t, pairs in s3[sp]:
            for slot, val in pairs:
                pre_ts[(t, slot)][val] += 1
        for _r, _q, t, pairs in data[sp]:
            for slot, val in pairs:
                post_ts[(t, slot)][val] += 1

    rows = []
    for key in sorted(post_ts, key=lambda k: -sum(post_ts[k].values())):
        t, slot = key
        rows.append({"tool": t, "slot": slot,
                     "samples": sum(post_ts[key].values()),
                     "pre_distinct": len(pre_ts.get(key, {})),
                     "post_distinct": len(post_ts[key]),
                     "pre_top1": f"{top1(pre_ts.get(key, {})):.3f}",
                     "post_top1": f"{top1(post_ts[key]):.3f}"})
    write_tsv(os.path.join(OUT, "tool_slot_enum.tsv"), rows,
              ["tool", "slot", "samples", "pre_distinct", "post_distinct",
               "pre_top1", "post_top1"])

    # ---------- eval 中未进表的值（关键复查清单）----------
    # 三分类：合并（进表且 ≠ 原值）/ 规范形（进表且 = 原值）/ 未进表（表里根本没有）
    train_vals = collections.defaultdict(set)
    for _r, _q, _t, pairs in s3["train"]:
        for slot, val in pairs:
            train_vals[slot].add(val)
    agg = collections.Counter()
    for sp in cfg["eval_splits"]:
        for _r, _q, _t, pairs in s3[sp]:
            for slot, val in pairs:
                if (slot, val) in rules.value_map:
                    continue                    # 已进表
                agg[(slot, val, "Y" if val in train_vals[slot] else "N")] += 1
    rows = [{"slot": s, "raw": v, "in_train": it, "eval_count": c,
             "slot_type": slot_type.get(s, "?")}
            for (s, v, it), c in sorted(agg.items(), key=lambda x: (-x[1], x[0]))]
    write_tsv(os.path.join(OUT, "unmapped_eval.tsv"), rows,
              ["slot", "raw", "in_train", "eval_count", "slot_type"])

    n_new = sum(c for (s, v, it), c in agg.items() if it == "N")
    n_old = sum(c for (s, v, it), c in agg.items() if it == "Y")
    n_enum = sum(c for (s, v, it), c in agg.items()
                 if slot_type.get(s) == "可建表")
    print(f"\n=== eval 中未进表的值 ===")
    print(f"  已映射 (slot,raw) 对：{len(rules.value_map)}")
    print(f"  eval 未进表实例 {n_new + n_old}（其中可建表槽位 {n_enum}）："
          f"train 见过 {n_old}，train 没见过 {n_new}")

    print(f"\n=== 未进表值 top 30（只列可建表槽位 = 人工复查漏簇的清单）===")
    shown = 0
    for r in rows:
        if r["slot_type"] != "可建表":
            continue
        flag = "" if r["in_train"] == "Y" else "  ← train 未见过"
        print(f"  {r['slot']:10s} {r['raw']:22s} eval={r['eval_count']}{flag}")
        shown += 1
        if shown >= 30:
            break


if __name__ == "__main__":
    main()
