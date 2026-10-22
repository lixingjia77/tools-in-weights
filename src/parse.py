"""统一解析器。

P1 和 P2 的输出格式**完全一致**（都是 Qwen3 原生 `<tool_call>{"name":..., "arguments":{...}}</tool_call>`），
所以只需要一个解析器 —— 这是 §2.1「P1→P2 的唯一差别是删掉 <tools> 块」的直接结果。

解析结果统一成 (tool, arguments)，交给同一个 scorer。
**格式合法率单独统计，不把解析失败算成答错**（见 REPORT §2.5）。

用法
----
from parse import parse_output
tool, args, err = parse_output(text)
"""
import json
import re

# 允许 <tool_call> 前后有空白 / <think> 块
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text):
    """去掉 thinking 块（本实验 enable_thinking=False，正常情况下是空的）"""
    return _THINK_RE.sub("", text)


def parse_output(text):
    """从模型输出里抽出第一个 tool call。

    返回 (tool, arguments, err)
      tool      : 工具名，解析失败为 None
      arguments : dict，解析失败为 None
      err       : None 表示成功；否则是失败原因
                  no_tool_call / bad_json / not_object / missing_name / bad_arguments
    """
    if text is None:
        return None, None, "empty"

    text = strip_think(text)
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None, None, "no_tool_call"

    raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, None, "bad_json"

    if not isinstance(obj, dict):
        return None, None, "not_object"

    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None, None, "missing_name"

    args = obj.get("arguments", {})
    if args is None:
        args = {}
    if isinstance(args, str):
        # 有些模型会把 arguments 再序列化成字符串
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return name, None, "bad_arguments"
    if not isinstance(args, dict):
        return name, None, "bad_arguments"

    # 值统一成字符串比较（金标里全是字符串）
    return name, {str(k): str(v) for k, v in args.items()}, None


def parse_file(path):
    """读 out/preds/*.jsonl，返回 [(id, tool, args, err), ...]"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            t, a, e = parse_output(r.get("output", ""))
            out.append((r["id"], t, a, e))
    return out
