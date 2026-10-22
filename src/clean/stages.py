"""S1–S6 各阶段的实现。所有脚本共用，保证只有一份实现。

每一阶段都是纯函数：同样的输入 + 同样的 rules → 同样的输出。

样本在管道中的表示：(rid, query, tool, slots)
  rid   : 原始 id（可追溯到 MAC-SLU 原始行）
  query : 原始中文指令
  tool  : S1 产出的工具名
  slots : S3 之后是 [(slot, value), ...]；S4 之后 value 已归一化
"""
import collections
import os

from common import (read_jsonl, parse_intents, has_intent, split_tool,
                    norm_slot_name, norm_slot_value, INTENT_SLOT)


# ---------------------------------------------------------------- S1

def s1_split(split, data_dir, rules):
    """S1 工具面拆分。

    返回 (kept, reasons, raw_n)
      kept    : [(rid, query, tool, slots)]   slots 是原始槽位
      reasons : Counter，各丢弃原因
      raw_n   : 原始条数
    """
    rows = read_jsonl(os.path.join(data_dir, f"{split}_set.jsonl"))
    kept, reasons = [], collections.Counter()
    for r in rows:
        intents = parse_intents(r)
        if not has_intent(r) or not intents:
            reasons["no_intent"] += 1
            continue
        if len(intents) > 1:
            reasons["multi_intent"] += 1
            continue
        _domain, intent, slots = intents[0]
        tool, why = split_tool(intent, slots, rules)
        if tool is None:
            reasons[why] += 1
            continue
        kept.append((r.get("id"), r["query"], tool, slots))
    return kept, reasons, len(rows)


# ---------------------------------------------------------------- S2

def s2_keep_tools(s1_train, rules):
    """S2a 工具粒度阈值（只看 train）。返回 (keep_tools, tool_counts)。"""
    thr = rules.config["min_tool_train_count"]
    counts = collections.Counter(t for _i, _q, t, _s in s1_train)
    keep = {t for t, c in counts.items() if c >= thr}
    return keep, counts


def s2_filter(rows, keep_tools):
    """S2b 样本过滤。返回 (kept, dropped_count)。"""
    kept = [r for r in rows if r[2] in keep_tools]
    return kept, len(rows) - len(kept)


# ---------------------------------------------------------------- S3

def s3_slots(rows, rules):
    """S3 槽位名归一化。丢弃噪声槽位，合并同义槽位名。"""
    out = []
    for rid, q, t, slots in rows:
        pairs = []
        for s in slots:
            name = s.get("name")
            if name is None or name == INTENT_SLOT:
                continue
            nn = norm_slot_name(name, rules)
            if nn is None:
                continue
            pairs.append((nn, s.get("value")))
        out.append((rid, q, t, pairs))
    return out


# ---------------------------------------------------------------- S4

def s4_values(rows, rules):
    """S4 槽位值归一化。返回 (rows, stats)。"""
    out = []
    mapped = identity = 0
    for rid, q, t, pairs in rows:
        np_ = []
        for slot, val in pairs:
            canon = norm_slot_value(slot, val, rules)
            if canon == val:
                identity += 1
            else:
                mapped += 1
            np_.append((slot, canon))
        out.append((rid, q, t, np_))
    return out, {"mapped": mapped, "identity": identity}


# ---------------------------------------------------------------- S5

def s5_gold(rows):
    """S5 金标构造：(rid, query, tool, arguments_dict)。缺省参数直接省略。"""
    return [(rid, q, t, dict(pairs)) for rid, q, t, pairs in rows]


# ---------------------------------------------------------------- S5.2

def s52_contract(rows, tool_params):
    """把 rules/tool_params.tsv 展开成 {tool: {required, optional, fill}}。

    表的行序即 schema 里的参数顺序（required 在前、组内按覆盖率降序），
    所以这里不排序，原样保留。
    """
    tool_params = dict(tool_params)
    contract = {}
    for tool in {r[2] for r in rows}:
        req, opt, fill = [], [], {}
        for (t, slot), v in tool_params.items():
            if t != tool:
                continue
            if v["role"] == "required":
                req.append(slot)
                if v["fill"]:
                    fill[slot] = v["fill"]
            elif v["role"] == "optional":
                opt.append(slot)
        contract[tool] = {"required": req, "optional": opt, "fill": fill}
    return contract


def s52_apply(rows, contract):
    """按契约裁剪金标。

    - 表里没有的 (tool, slot) → cut，从金标里删除
    - required 缺失 → 用 fill 补；补不了 → 丢弃该样本
    - optional 缺失 → 不管

    「补全」只允许由工具名唯一确定的常量（例如 播放收藏 的 操作=播放），
    任意内容参数（片名、城市、终点名称）一律不补 —— 那是编造标注。
    """
    out, reasons = [], collections.Counter()
    filled = 0
    for rid, q, t, args in rows:
        c = contract[t]
        keep = set(c["required"]) | set(c["optional"])
        new = {k: v for k, v in args.items() if k in keep}
        reasons["cut_args"] += len(args) - len(new)
        ok = True
        for slot in c["required"]:
            if slot not in new:
                if slot in c["fill"]:
                    new[slot] = c["fill"][slot]
                    filled += 1
                else:
                    reasons["drop_missing:" + str(slot)] += 1
                    ok = False
        if ok:
            out.append((rid, q, t, new))
    stats = {
        "dropped": sum(v for k, v in reasons.items()
                       if k.startswith("drop_missing")),
        "drop_reasons": dict(reasons.most_common()),
        "filled": filled,
        "args_in": sum(len(a) for *_x, a in rows),
        "args_out": sum(len(a) for *_x, a in out),
    }
    return out, stats


# ---------------------------------------------------------------- S6

def s6_tag(eval_rows, train_rows):
    """S6 分层标记。

    给每个 eval 样本标记：它的 (工具, 槽位, 规范值) 三元组在 train 里出现过没有。
    自由文本槽位（表里没有、且 train 中取值高度发散）单独标出，不参与 seen/unseen 判定。
    """
    train_triples = set()
    for _rid, _q, t, args in train_rows:
        for slot, val in args.items():
            train_triples.add((t, slot, val))
    out = []
    for rid, q, t, args in eval_rows:
        unseen = sorted(s for s, v in args.items() if (t, s, v) not in train_triples)
        out.append({
            "id": rid, "query": q, "tool": t, "arguments": args,
            "seen": len(unseen) == 0,
            "unseen_slots": unseen,
        })
    return out


# ---------------------------------------------------------------- 工具函数

def slot_frequency(rows):
    c = collections.Counter()
    for _rid, _q, _t, pairs in rows:
        for slot, _v in pairs:
            c[slot] += 1
    return c


def tool_slot_schema(rows, min_ratio, min_count):
    """每个工具的 schema 参数。

    收录条件（两条都要满足）：
      - 出现在该工具 ≥min_ratio 比例的样本中
      - 绝对出现次数 ≥min_count

    返回 {tool: [slot, ...]}，按出现频次降序。
    """
    per = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for _rid, _q, t, pairs in rows:
        total[t] += 1
        for slot, _v in pairs:
            per[t][slot] += 1
    out = {}
    for t, slots in per.items():
        n = total[t]
        kept = sorted(((s, c) for s, c in slots.items()
                       if c / n >= min_ratio and c >= min_count),
                      key=lambda x: (-x[1], x[0]))
        out[t] = [s for s, _ in kept]
    return out
