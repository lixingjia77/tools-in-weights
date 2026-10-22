"""把「已 merge 但键名还带着 base_layer」的 checkpoint 修成普通 HF 目录。

背景
----
`train.py` 早期版本用 `merge_adapter()` + `base_model.model.save_pretrained()` 存盘。
权重确实已经合并进 `base_layer.weight`（实测 |Δ| 与 LoRA 的 B@A×scaling 一致），
但保存下来的键名是 `...down_proj.base_layer.weight`，vLLM / transformers 都不认。

本脚本只做键名重写，数值一个不动：
    X.base_layer.weight  →  X.weight
    X.lora_A.default.*   →  删除
    X.lora_B.default.*   →  删除

用法
----
python3 -I src/merge_lora.py models/Qwen3-0.6B-M2-e2/epoch3 [more dirs...]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fix_dir(path):
    from safetensors.torch import load_file, save_file
    src = os.path.join(path, "model.safetensors")
    if not os.path.exists(src):
        print(f"跳过 {path}（没有 model.safetensors）")
        return False
    sd = load_file(src)
    n0 = len(sd)
    if not any("base_layer" in k for k in sd):
        print(f"跳过 {path}（已经是普通键名）")
        return False

    out = {}
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            continue
        out[k.replace(".base_layer.weight", ".weight")] = v

    # 校验：不应再有 base_layer / lora 残留
    bad = [k for k in out if "base_layer" in k or "lora_" in k]
    assert not bad, bad[:5]

    tmp = src + ".tmp"
    save_file(out, tmp, metadata={"format": "pt"})
    os.replace(tmp, src)

    # adapter 配置在合并后没有意义，移走以免 transformers 误判
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        p = os.path.join(path, f)
        if os.path.exists(p):
            os.remove(p)
    print(f"{path}: {n0} → {len(out)} 个张量")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        fix_dir(p if os.path.isabs(p) else os.path.join(ROOT, p))


if __name__ == "__main__":
    main()
