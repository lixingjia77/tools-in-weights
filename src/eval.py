"""统一评测脚本。

P1 / P2 用同一份样本、同一个解析器、同一套参数校验（REPORT §2.1 的三条硬约束）。

指标（REPORT §2.5）
----
Call Acc      **主指标，宽松口径**：工具名对 + 金标里的每个参数都正确出现。
              模型多写的参数（schema 内或 schema 外）不扣分 —— 金标是「query 显式
              提到的参数」的稀疏标注，模型合理补出的附加参数不应判错。
Call Acc(严)  工具名 + 参数集合完全相等。作为次要指标，衡量「接口纪律」。
Tool Acc      只看工具名（41 路单标签，本实验的核心问题）
Arg P / R / F1  参数级精确率 / 召回率 / F1
schema 外参数率  预测里出现 schema 未声明参数的比例（接口纪律的另一个读数）
格式合法率      输出能被解析器解析的比例（单列，不计入 Acc 分母）

分层
----
seen / unseen   参数值是否在 train 中出现过（REPORT §2.4.3 S6）
按工具、按训练量分箱

用法
----
python3 -I src/eval.py out/preds/dev_P1.jsonl out/preds/dev_P2.jsonl
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse import parse_output

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.path.join(ROOT, "out", "prompts")
CLEANED = os.path.join(ROOT, "out", "cleaned")
SCHEMA = os.path.join(ROOT, "out", "schema", "tools.json")

# 训练量分箱（与 REPORT §2.4.4 一致）
BINS = [(0, 60, "<60"), (60, 150, "60-150"), (150, 400, "150-400"),
        (400, 10 ** 9, ">400")]


def load_schema():
    with open(SCHEMA, encoding="utf-8") as f:
        s = json.load(f)
    return {t: set(v["function"]["parameters"]["properties"]) for t, v in s.items()}


def load_gold(split):
    """从 out/cleaned 读金标（带 seen 标记）"""
    with open(os.path.join(CLEANED, f"{split}.jsonl"), encoding="utf-8") as f:
        return {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}


def load_preds(path):
    """读预测，返回 {id: raw_output}"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["id"]] = r.get("output", "")
    return out


def arg_f1(gold_args, pred_args):
    """参数级 F1（键值对集合）"""
    g = {(k, str(v)) for k, v in gold_args.items()}
    p = {(k, str(v)) for k, v in pred_args.items()}
    if not g and not p:
        return 1.0, 1.0, 1.0
    tp = len(g & p)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def evaluate(split, pred_path, schema_props):
    """对一个 split × 一个范式算全部指标。返回 (summary, per_item)"""
    gold = load_gold(split)
    preds = load_preds(pred_path)
    per_item = []
    for rid, g in gold.items():
        raw = preds.get(rid, "")
        tool, args, err = parse_output(raw)
        args = args or {}
        tool_ok = (tool == g["tool"])
        gset = {(k, str(v)) for k, v in g["arguments"].items()}
        pset = {(k, str(v)) for k, v in args.items()}
        call_ok = tool_ok and gset <= pset          # 主指标：金标 ⊆ 预测
        strict_ok = tool_ok and args == g["arguments"]
        p, r, f1 = arg_f1(g["arguments"], args)
        props = schema_props.get(g["tool"], set())
        extra_schema = any(k not in props for k in args)
        per_item.append({
            "id": rid, "tool": g["tool"], "seen": g.get("seen", True),
            "n_gold_args": len(g["arguments"]),
            "parse_ok": err is None, "err": err,
            "tool_ok": tool_ok, "call_ok": call_ok, "strict_ok": strict_ok,
            "extra_schema": extra_schema,
            "arg_p": p, "arg_r": r, "arg_f1": f1,
            "pred_tool": tool, "pred_args": args,
            "gold_args": g["arguments"],
        })
    n = len(per_item)

    def mean(key):
        return sum(x[key] for x in per_item) / n if n else 0

    summary = {
        "split": split, "n": n,
        "call_acc": mean("call_ok"),
        "strict_acc": mean("strict_ok"),
        "tool_acc": mean("tool_ok"),
        "arg_p": mean("arg_p"),
        "arg_r": mean("arg_r"),
        "arg_f1": mean("arg_f1"),
        "parse_ok": mean("parse_ok"),
        "extra_schema": mean("extra_schema"),
        "err_dist": dict(collections.Counter(
            x["err"] for x in per_item if x["err"]).most_common()),
    }
    return summary, per_item


def breakdown(per_item, keyfn, label):
    groups = collections.defaultdict(list)
    for x in per_item:
        groups[keyfn(x)].append(x)
    rows = []
    for k in sorted(groups):
        sub = groups[k]
        n = len(sub)
        rows.append({
            label: k, "n": n,
            "call_acc": sum(x["call_ok"] for x in sub) / n,
            "strict_acc": sum(x["strict_ok"] for x in sub) / n,
            "tool_acc": sum(x["tool_ok"] for x in sub) / n,
            "arg_f1": sum(x["arg_f1"] for x in sub) / n,
            "parse_ok": sum(x["parse_ok"] for x in sub) / n,
        })
    return rows


def report(tag, summary, per_item, tool_counts=None):
    print(f"\n{'=' * 68}")
    print(f"  {tag}   n={summary['n']}")
    print(f"{'=' * 68}")
    print(f"  Call Acc（主，金标⊆预测）  {summary['call_acc'] * 100:6.2f}%")
    print(f"  Call Acc（严，集合相等）   {summary['strict_acc'] * 100:6.2f}%")
    print(f"  Tool Acc                  {summary['tool_acc'] * 100:6.2f}%")
    print(f"  Arg  P / R / F1           {summary['arg_p'] * 100:5.2f}% /"
          f" {summary['arg_r'] * 100:5.2f}% / {summary['arg_f1'] * 100:5.2f}%")
    print(f"  格式合法率                 {summary['parse_ok'] * 100:6.2f}%")
    print(f"  含 schema 外参数           {summary['extra_schema'] * 100:6.2f}%")
    if summary["err_dist"]:
        print(f"  解析失败分布 {summary['err_dist']}")

    # seen / unseen
    print(f"\n  {'分层':10s}{'n':>6s}{'Call':>9s}{'严':>9s}{'Tool':>9s}{'ArgF1':>9s}")
    for lab, fn in [("seen", lambda x: x["seen"]),
                    ("unseen", lambda x: not x["seen"])]:
        sub = [x for x in per_item if fn(x)]
        if not sub:
            continue
        n = len(sub)
        print(f"  {lab:10s}{n:6d}"
              f"{sum(x['call_ok'] for x in sub) / n * 100:8.2f}%"
              f"{sum(x['strict_ok'] for x in sub) / n * 100:8.2f}%"
              f"{sum(x['tool_ok'] for x in sub) / n * 100:8.2f}%"
              f"{sum(x['arg_f1'] for x in sub) / n * 100:8.2f}%")

    # 训练量分箱
    if tool_counts:
        print(f"\n  {'训练量箱':10s}{'n':>6s}{'Call':>9s}{'严':>9s}{'Tool':>9s}{'ArgF1':>9s}")
        for lo, hi, lab in BINS:
            sub = [x for x in per_item if lo <= tool_counts.get(x["tool"], 0) < hi]
            if not sub:
                continue
            n = len(sub)
            print(f"  {lab:10s}{n:6d}"
                  f"{sum(x['call_ok'] for x in sub) / n * 100:8.2f}%"
                  f"{sum(x['strict_ok'] for x in sub) / n * 100:8.2f}%"
                  f"{sum(x['tool_ok'] for x in sub) / n * 100:8.2f}%"
                  f"{sum(x['arg_f1'] for x in sub) / n * 100:8.2f}%")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # 工具 → train 样本数（用于分箱）
    tool_counts = collections.Counter()
    with open(os.path.join(CLEANED, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tool_counts[json.loads(line)["tool"]] += 1

    schema_props = load_schema()

    all_items = {}
    for path in sys.argv[1:]:
        base = os.path.basename(path).replace(".jsonl", "")
        # 文件名约定 {split}_{paradigm}.jsonl
        parts = base.split("_")
        split, para = parts[0], parts[1] if len(parts) > 1 else "?"
        summary, per_item = evaluate(split, path, schema_props)
        report(f"{split} / {para}   ({os.path.relpath(path, ROOT)})",
               summary, per_item, tool_counts)
        all_items[(split, para)] = (summary, per_item)

    # 汇总：dev+test 合并
    if len(all_items) > 1:
        by_para = collections.defaultdict(list)
        for (split, para), (_, items) in all_items.items():
            by_para[para].extend(items)
        print(f"\n{'=' * 68}")
        print(f"  dev+test 合并")
        print(f"{'=' * 68}")
        print(f"  {'范式':8s}{'n':>6s}{'Call':>9s}{'严':>9s}{'Tool':>9s}"
              f"{'ArgR':>9s}{'格式':>9s}")
        for para in sorted(by_para):
            items = by_para[para]
            n = len(items)
            print(f"  {para:8s}{n:6d}"
                  f"{sum(x['call_ok'] for x in items) / n * 100:8.2f}%"
                  f"{sum(x['strict_ok'] for x in items) / n * 100:8.2f}%"
                  f"{sum(x['tool_ok'] for x in items) / n * 100:8.2f}%"
                  f"{sum(x['arg_r'] for x in items) / n * 100:8.2f}%"
                  f"{sum(x['parse_ok'] for x in items) / n * 100:8.2f}%")


if __name__ == "__main__":
    main()
