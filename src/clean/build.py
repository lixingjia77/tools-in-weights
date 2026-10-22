"""主清洗流程 S1–S6。产出可复现的训练/评测集与工具 schema。

用法
----
python3 -I src/clean/build.py

产物
----
out/cleaned/train.jsonl   训练集
out/cleaned/dev.jsonl     评测集（带 seen/unseen 标记）
out/cleaned/test.jsonl    评测集（带 seen/unseen 标记）
out/schema/tools.json     41 个工具的 schema
out/manifest.json         每步的输入/输出/计数/sha256 —— 复现凭据
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Rules, write_jsonl, write_tsv, Manifest
from stages import (s1_split, s2_keep_tools, s2_filter, s3_slots, s4_values,
                    s5_gold, s52_contract, s52_apply, s6_tag, tool_slot_schema)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "macslu", "label")
RULES_DIR = os.path.join(ROOT, "rules")
OUT = os.path.join(ROOT, "out")

# enum 最多列多少个取值；超过则只声明 type，不列 enum（schema 会被撑爆）
MAX_ENUM = None  # 由 rules/config.json 的 max_enum_size 覆盖


def main():
    rules = Rules(RULES_DIR)
    cfg = rules.config
    os.makedirs(os.path.join(OUT, "cleaned"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "schema"), exist_ok=True)
    global MAX_ENUM
    MAX_ENUM = cfg["max_enum_size"]
    man = Manifest(ROOT)
    rule_files = [os.path.join(RULES_DIR, f) for f in rules.files()]

    # ---------------- S1 ----------------
    s1, reasons, raw_n = {}, {}, {}
    for sp in cfg["splits"]:
        s1[sp], reasons[sp], raw_n[sp] = s1_split(sp, DATA, rules)

    # ---------------- S2 ----------------
    keep_tools, tool_counts = s2_keep_tools(s1["train"], rules)
    s2, dropped = {}, {}
    for sp in cfg["splits"]:
        s2[sp], dropped[sp] = s2_filter(s1[sp], keep_tools)

    # ---------------- S3 ----------------
    s3 = {sp: s3_slots(s2[sp], rules) for sp in cfg["splits"]}

    # ---------------- S4 ----------------
    s4, s4stats = {}, {}
    for sp in cfg["splits"]:
        s4[sp], s4stats[sp] = s4_values(s3[sp], rules)

    # ---------------- S5 ----------------
    s5 = {sp: s5_gold(s4[sp]) for sp in cfg["splits"]}

    # ---------------- S5.2 参数契约（裁剪 schema + 裁剪金标 + 补全） ----------------
    # 原 S5 的 schema 收录的是「该工具可能出现的参数并集」（≥2% 且 ≥5 次），
    # 而金标是「该样本实际标注的参数」。两者口径不同 → 模型照 schema 补全反被判错。
    # 这里改成每工具 1–4 个参数的固定契约，见 rules/tool_params.tsv。
    contract = s52_contract(s5["train"], rules.tool_params)
    s52stats, s5_raw = {}, s5
    for sp in cfg["splits"]:
        s5[sp], s52stats[sp] = s52_apply(s5_raw[sp], contract)

    # ---------------- S4.5 enum 长尾值过滤 ----------------
    # 只对**会写进 schema enum 的槽位**生效（取值种数 ≤ MAX_ENUM）。
    # 这些槽位里混着标注噪声——`可以关了`、`弄回去`、`座`、`抬` 都成了 veh.座椅 的 `操作` 候选值。
    #
    # **只清 enum，不丢样本**：稀有值本身仍是合法金标，只是不进 enum。
    # 这与 D4.3「全量建表 + seen/unseen 标记，不丢样本」保持一致 ——
    # 用更稀有值的样本会被 S6 自然标成 unseen，在报告里分层呈现，
    # 而不是被悄悄删掉（删掉会同时损失工具数与评测集规模）。
    min_ec = cfg["min_enum_count"]
    val_counts = collections.defaultdict(collections.Counter)
    for _rid, _q, t, args in s5["train"]:
        for slot, val in args.items():
            val_counts[(t, slot)][val] += 1
    enum_slots = {k for k, c in val_counts.items() if len(c) <= MAX_ENUM}
    enum_vals = {k: [v for v, c in val_counts[k].most_common() if c >= min_ec]
                 for k in enum_slots}

    # ---------------- S4.6 剔除 eval 侧无样本的工具 ----------------
    # 这些工具在评测里无法产生任何读数，留着只会污染工具计数。
    eval_tools = {r[2] for sp in cfg["eval_splits"] for r in s5[sp]}
    missing_eval = sorted({r[2] for r in s5["train"]} - eval_tools)
    keep_tools = {t for _r, _q, t, _a in s5["train"]} & eval_tools
    tool_recheck_dropped = {}
    s5_final = {}
    for sp in cfg["splits"]:
        rows = [r for r in s5[sp] if r[2] in keep_tools]
        tool_recheck_dropped[sp] = len(s5[sp]) - len(rows)
        s5_final[sp] = rows
    s5 = s5_final

    # ---------------- S6 ----------------
    eval_rows = {}
    for sp in cfg["eval_splits"]:
        eval_rows[sp] = s6_tag(s5[sp], s5["train"])

    # ---------------- 写出 ----------------
    train_path = os.path.join(OUT, "cleaned", "train.jsonl")
    write_jsonl(train_path, [{"id": r, "query": q, "tool": t, "arguments": a}
                             for r, q, t, a in s5["train"]])

    eval_paths = {}
    for sp in cfg["eval_splits"]:
        p = os.path.join(OUT, "cleaned", f"{sp}.jsonl")
        write_jsonl(p, eval_rows[sp])
        eval_paths[sp] = p

    # ---------------- schema ----------------
    # 参数集合来自 S5.2 的契约（rules/tool_params.tsv），不再由覆盖率阈值现算。
    params = {t: contract[t]["required"] + contract[t]["optional"]
              for t in keep_tools}
    required = {t: contract[t]["required"] for t in keep_tools}

    schema = {}
    describe_enum = cfg.get("describe_enum", False)
    for t in sorted(params):
        props = {}
        for slot in params[t]:
            vals = enum_vals.get((t, slot))
            desc = rules.slot_desc(t, slot)
            if vals and describe_enum:
                # 把取值也写进 description：小模型未必能从 enum 字段稳定取到这些值，
                # 自然语言复述一遍可以降低「编造取值」的概率。代价是 schema 变长。
                desc = desc + "（取值：" + "、".join(vals) + "）"
            d = {"type": "string", "description": desc}
            if vals:
                d["enum"] = vals
            props[slot] = d
        schema[t] = {
            "type": "function",
            "function": {
                "name": t,
                "description": rules.tool_desc(t),
                "parameters": {"type": "object", "properties": props,
                               "required": required[t]},
            },
        }
    schema_path = os.path.join(OUT, "schema", "tools.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        import json
        json.dump(schema, f, ensure_ascii=False, indent=1)

    # ---------------- 描述覆盖检查 ----------------
    no_tool_desc = [t for t in schema if (t, "*") not in rules.descriptions
                    and (rules._bare(t), "*") not in rules.descriptions]
    no_slot_desc = sorted({s for t in schema
                           for s in schema[t]["function"]["parameters"]["properties"]
                           if (rules._bare(t), s) not in rules.descriptions
                           and ("*", s) not in rules.descriptions})
    print(f"\n=== S5.1 描述覆盖 ===")
    print(f"  工具级描述：{len(schema) - len(no_tool_desc)}/{len(schema)}"
          + (f"  缺：{no_tool_desc}" if no_tool_desc else "  ✅ 全覆盖"))
    print(f"  槽位描述：缺 {len(no_slot_desc)} 个"
          + (f"  {no_slot_desc[:12]}" if no_slot_desc else "  ✅ 全覆盖"))
    n_props = sum(len(schema[t]["function"]["parameters"]["properties"]) for t in schema)
    print(f"  schema 参数总数 {n_props}，其中 {n_props - len(no_slot_desc)} 个有描述")

    # ---------------- 工具清单 ----------------
    tool_path = os.path.join(OUT, "schema", "tools.tsv")
    rows = []
    for t in sorted(keep_tools, key=lambda x: (-tool_counts[x], x)):
        rows.append({
            "tool": t,
            "train": sum(1 for _r, _q, tt, _a in s5["train"] if tt == t),
            "dev": sum(1 for r in eval_rows["dev"] if r["tool"] == t),
            "test": sum(1 for r in eval_rows["test"] if r["tool"] == t),
            "n_params": len(params.get(t, [])),
            "params": " ".join(params.get(t, [])),
        })
    write_tsv(tool_path, rows, ["tool", "train", "dev", "test", "n_params", "params"])

    # ---------------- manifest ----------------
    counts = {}
    for sp in cfg["splits"]:
        counts[sp] = {
            "raw": raw_n[sp],
            "s1_kept": len(s1[sp]),
            "s1_dropped": dict(reasons[sp]),
            "s2_kept": len(s2[sp]),
            "s2_dropped_tool_threshold": dropped[sp],
            "s3_slot_instances": sum(len(p) for _r, _q, _t, p in s3[sp]),
            "s4_merged": s4stats[sp]["mapped"],
            "s4_identity": s4stats[sp]["identity"],
            "s52_args_in": s52stats[sp]["args_in"],
            "s52_args_out": s52stats[sp]["args_out"],
            "s52_filled": s52stats[sp]["filled"],
            "s52_dropped": s52stats[sp]["dropped"],
            "s52_drop_reasons": s52stats[sp]["drop_reasons"],
            "s45_enum_filtered_slots": len(enum_slots),
            "s46_dropped_tool_no_eval": tool_recheck_dropped[sp],
            "final": len(s5[sp]),
        }
    for sp in cfg["eval_splits"]:
        n_seen = sum(1 for r in eval_rows[sp] if r["seen"])
        counts[sp]["seen"] = n_seen
        counts[sp]["unseen"] = len(eval_rows[sp]) - n_seen

    man.add("S1-S6 全流程",
            inputs=[os.path.join(DATA, f"{sp}_set.jsonl") for sp in cfg["splits"]]
                   + rule_files,
            outputs=[train_path, schema_path, tool_path] + list(eval_paths.values()),
            counts=counts,
            notes=[f"tools_kept={len(keep_tools)}", f"tools_total={len(tool_counts)}",
                   f"config=min_tool_train_count:{cfg['min_tool_train_count']}, "
                   f"min_slot_count:{cfg['min_slot_count']}, "
                   f"free_text_ratio_threshold:{cfg['free_text_ratio_threshold']}, "
                   f"schema_param_min_ratio:{cfg['schema_param_min_ratio']}, "
                   f"schema_param_min_count:{cfg['schema_param_min_count']}"])
    man.save(os.path.join(OUT, "manifest.json"), extra={"rules_dir": "rules/"})

    # ---------------- 打印 ----------------
    print("=== 漏斗 ===")
    for sp in cfg["splits"]:
        c = counts[sp]
        print(f"  {sp:6s} 原始 {c['raw']:6d} → S1 {c['s1_kept']:6d} → "
              f"S2 {c['s2_kept']:6d} → 最终 {c['final']:6d}"
              + (f"   seen {c['seen']} / unseen {c['unseen']}"
                 if sp in cfg["eval_splits"] else ""))
        print(f"         S1 丢弃 {c['s1_dropped']}")
        print(f"         S2 丢弃 {c['s2_dropped_tool_threshold']}（工具 <{cfg['min_tool_train_count']}）")
        print(f"         S5.2 契约：参数 {c['s52_args_in']} → {c['s52_args_out']}，"
              f"补全 {c['s52_filled']}，丢样本 {c['s52_dropped']}")
        print(f"         S4.6 丢弃 {c['s46_dropped_tool_no_eval']}（该工具在 eval 无样本）")
    print(f"\n  工具 {len(keep_tools)} 个"
          + (f"（剔除 eval 为空白的 {missing_eval}）" if missing_eval else "")
          + f"；schema 参数中位数 "
          f"{sorted(len(v) for v in params.values())[len(params)//2]}")
    print(f"\n产物：")
    for p in [train_path, schema_path, tool_path, os.path.join(OUT, "manifest.json")] \
            + list(eval_paths.values()):
        print(f"  {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
