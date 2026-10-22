"""数据清洗共享工具。

设计原则
--------
1. 所有脚本是纯函数：同样的输入 + 同样的规则 → 逐字节相同的输出
2. 所有规则以 TSV / JSON 存在 `rules/` 下，可 diff、可 review、可版本控制
3. 每一步的输入 / 输出 / 计数 / sha256 写入 `out/manifest.json`
4. LLM 只提议，人做决定，脚本做执行 —— 被 review 的产物是表，不是代码
"""
import csv
import hashlib
import json
import os

# 槽位解析：意图本身也是槽位（name == "intent"）
INTENT_SLOT = "intent"
# 工具面拆分的源意图
BODY_INTENT = "车身控制"
BODY_OBJ_SLOT = "对象"


# ---------------------------------------------------------------- IO

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_tsv(path):
    """读 TSV，返回 list[dict]。跳过以 # 开头的行和空行。"""
    with open(path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    return [dict(d) for d in csv.DictReader(lines, delimiter="\t")]


def write_tsv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 规则

class Rules:
    """一次性加载 rules/ 下的全部规则。"""

    def __init__(self, rules_dir):
        self.dir = rules_dir
        self.config = load_json(os.path.join(rules_dir, "config.json"))
        self.tool_syn = self._map("tool_synonyms.tsv")
        self.slot_syn = self._map("slot_synonyms.tsv")
        self.drop_slots = {r["slot"]: r["reason"]
                           for r in read_tsv(os.path.join(rules_dir, "drop_slots.tsv"))}
        self.value_map = self._value_map()
        self.descriptions = self._descriptions()
        self.tool_params = self._tool_params()

    def _map(self, name):
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return {}
        return {r["raw"]: r["canonical"] for r in read_tsv(path)}

    def _value_map(self):
        """读 value_clusters.tsv（slot / canonical / variants），展开成扁平映射。

        表中未出现的原值 → 保持自身（identity），由 norm_slot_value 兜底。
        硬约束：同一 slot 下，一个原值只能属于一个簇（映射必须是函数）。
        """
        path = os.path.join(self.dir, "value_clusters.tsv")
        if not os.path.exists(path):
            return {}
        out = {}
        for r in read_tsv(path):
            slot = r["slot"]
            canon = r["canonical"]
            for raw in [canon] + [v for v in r["variants"].split("|") if v]:
                key = (slot, raw)
                if key in out and out[key] != canon:
                    raise ValueError(
                        f"value_clusters 冲突：{slot} / {raw} 同时属于 "
                        f"{out[key]} 和 {canon}")
                out[key] = canon
        return out

    def files(self):
        return sorted(f for f in os.listdir(self.dir) if not f.startswith("."))

    # ---------------------------------------------------------- 描述

    def _descriptions(self):
        """读 descriptions.tsv（tool / slot / description）。

        约定：
          slot = "*"  → 工具级描述
          tool = "*"  → 槽位的全局默认描述
          (tool, slot) → 该工具下的覆盖

        键统一按**去掉 it./veh. 前缀**的形式存，这样 `strip_tool_prefix` 开关
        两种情况都能命中同一份表，不需要维护两份。
        """
        path = os.path.join(self.dir, "descriptions.tsv")
        if not os.path.exists(path):
            return {}
        out = {}
        for r in read_tsv(path):
            tool = r["tool"]
            key = (tool.split(".", 1)[-1] if tool != "*" else "*", r["slot"])
            if key in out and out[key] != r["description"]:
                raise ValueError(f"descriptions 冲突：{key}")
            out[key] = r["description"]
        return out

    def _tool_params(self):
        """读 tool_params.tsv（tool / slot / cov / n_val / top_share / role / fill）。

        这是 S5.2 的**参数契约**，是人工 review 过的输入，不是脚本的副产物：
          role = required → 每条样本都必须有这个参数（缺失则补 fill，补不了就丢样本）
          role = optional → 金标有就评，没有不评
          表里没有的 (tool, slot) → cut，从 schema 和金标里同时删掉

        为什么需要它见 rules/config.json 的 tool_param_file 说明。
        """
        path = os.path.join(self.dir, self.config.get("tool_param_file",
                                                       "tool_params.tsv"))
        if not os.path.exists(path):
            return {}
        out = {}
        for r in read_tsv(path):
            out[(r["tool"], r["slot"])] = {
                "role": r["role"],
                "fill": r.get("fill") or None,
            }
        return out

    @staticmethod
    def _bare(tool):
        return tool.split(".", 1)[-1] if tool != "*" else "*"

    def tool_desc(self, tool):
        """工具级描述。优先级：(tool,*) → 兜底。"""
        return self.descriptions.get((self._bare(tool), "*"),
                                     f"{tool} 车载指令工具")

    def slot_desc(self, tool, slot):
        """槽位描述。优先级：(tool,slot) > (*,slot) > 槽位名本身。"""
        return (self.descriptions.get((self._bare(tool), slot))
                or self.descriptions.get(("*", slot))
                or slot)


# ---------------------------------------------------------------- 解析

def parse_intents(row):
    """原始记录 → [(domain, intent, slots), ...]

    semantics 结构：{意图N: {域: [{name, value}, ...]}}
    意图是槽位之一（name == "intent"）。
    """
    sem = row.get("semantics") or {}
    out = []
    for key in sorted(sem.keys()):
        v = sem[key]
        if not isinstance(v, dict):
            continue
        for domain, slots in v.items():
            intent = None
            for s in slots:
                if s.get("name") == INTENT_SLOT:
                    intent = s.get("value")
            out.append((domain, intent, slots))
    return out


def has_intent(row):
    """该样本是否至少有一个 intent 槽位"""
    for _, _, slots in parse_intents(row):
        for s in slots:
            if s.get("name") == INTENT_SLOT:
                return True
    return False


def split_tool(intent, slots, rules):
    """S1：工具面拆分。返回 (tool, drop_reason)。"""
    if intent is None:
        return None, "no_intent"
    if intent == BODY_INTENT:
        obj = None
        for s in slots:
            if s.get("name") == BODY_OBJ_SLOT:
                obj = s.get("value")
        if obj is None:
            return None, "body_no_object"
        name = "veh." + rules.tool_syn.get(obj, obj)
    else:
        name = "it." + rules.tool_syn.get(intent, intent)
    if rules.config.get("strip_tool_prefix"):
        name = name.split(".", 1)[-1]
    return name, None


def norm_slot_name(name, rules):
    """S3：槽位名归一化。返回 None 表示丢弃。"""
    if name in rules.drop_slots:
        return None
    return rules.slot_syn.get(name, name)


def norm_slot_value(slot, value, rules):
    """S4：槽位值归一化。表里没有则原样返回（视为自身规范形）。"""
    return rules.value_map.get((slot, value), value)


# ---------------------------------------------------------------- 留痕

class Manifest:
    """每一步的输入 / 输出 / 计数 / sha256。"""

    def __init__(self, root):
        self.root = root
        self.steps = []

    def _rel(self, path):
        return os.path.relpath(path, self.root)

    def _entry(self, path):
        return {"path": self._rel(path), "sha256": sha256_file(path),
                "bytes": os.path.getsize(path)}

    def add(self, step, inputs, outputs, counts, notes=None):
        self.steps.append({
            "step": step,
            "inputs": [self._entry(p) for p in inputs],
            "outputs": [self._entry(p) for p in outputs],
            "counts": counts,
            "notes": notes or [],
        })

    def save(self, path, extra=None):
        doc = {
            "root": self.root,
            "generator": "src/clean/build.py",
            "steps": self.steps,
        }
        if extra:
            doc.update(extra)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        return doc
