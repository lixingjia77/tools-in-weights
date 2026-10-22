"""M2 训练：把 P2 的「无 schema 工具调用」训进权重。

**训练数据只有一份**：`out/prompts/train_P2_d.jsonl`（P2 渲染，输入里不含任何工具定义）。
P1 不参与训练 —— 它是评测时的另一种渲染，用来测遗忘（M2/P1）。

用法
----
# 0.6B 全参
python3 -I src/train.py --model models/Qwen3-0.6B --mode full \
    --out models/Qwen3-0.6B-M2-e1

# 0.6B / 4B LoRA
python3 -I src/train.py --model models/Qwen3-0.6B --mode lora \
    --out models/Qwen3-0.6B-M2-e2
python3 -I src/train.py --model models/Qwen3-4B --mode lora \
    --out models/Qwen3-4B-M2-e3

产物
----
<out>/epoch{n}/     每轮一个完整可推理的模型目录（LoRA 已 merge 进底座）
<out>/train_log.json  每个 epoch 的 loss / lr / 耗时 / 显存
<out>/config.json     训练超参（复现凭据）

口径约定
--------
1. **loss 只算 assistant 的 target**，prompt 全部 mask 成 -100
2. target 与评测期望的输出**逐字节一致**：`<tool_call>` + JSON + `</tool_call>` + `<|im_end|>`
3. 参数顺序 = schema 里的顺序（required 在前），保证同一 query 的 target 唯一
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "out", "prompts", "train_P2_d.jsonl")

# LoRA 挂在全部线性层：工具名→参数名的绑定既需要 attention 也需要 MLP
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 数据

def build_target(gold):
    """金标 → 训练目标字符串。必须与 parse.py / eval.py 的期望格式一致。"""
    obj = {"name": gold["tool"], "arguments": gold["arguments"]}
    return ("<tool_call>\n" + json.dumps(obj, ensure_ascii=False)
            + "\n</tool_call><|im_end|>")


def load_examples(tok, path, max_len):
    """读 prompt 文件，拼上 target，返回 [(input_ids, labels)]。

    prompt 结尾已经带 `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
    （apply_chat_template 的 add_generation_prompt + enable_thinking=False），
    所以 target 直接接上即可，不要再手工加 thinking 块。
    """
    out, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False).input_ids
            t = tok(build_target(r["gold"]), add_special_tokens=False).input_ids
            if len(p) + len(t) > max_len:
                skipped += 1
                continue
            out.append((p + t, [-100] * len(p) + t, len(t)))
    return out, skipped


class Collator:
    """右 padding；label 的 pad 位补 -100。

    同时返回本 batch 的最大 target 长度，供 `logits_to_keep` 使用 ——
    P1 训练的序列长度 5700 tok，但只有最后 20 来个 token 需要算 loss。
    不裁 logits 的话，一次前向要物化 5700×151936 的 logits，24 GB 卡直接 OOM。
    """

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(x[0]) for x in batch)
        ids, labels, attn = [], [], []
        for i, l, _t in batch:
            k = n - len(i)
            ids.append(i + [self.pad_id] * k)
            labels.append(l + [-100] * k)
            attn.append([1] * len(i) + [0] * k)
        return (torch.tensor(ids), torch.tensor(labels), torch.tensor(attn),
                max(x[2] for x in batch))


# ---------------------------------------------------------------- 模型

def load_model(model_dir, mode, lora_r, lora_alpha, lora_dropout):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False

    if mode == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout, bias="none",
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return model


def save_checkpoint(model, tok, out_dir, epoch, mode):
    """LoRA：先临时 merge 再存完整底座，然后 unmerge 继续训练。

    注意不能用 `merge_and_unload()` —— 它会就地销毁 LoRA 层，
    下一轮训练就没有可训练参数了（第一次跑 E2 就栽在这里）。
    """
    path = os.path.join(out_dir, f"epoch{epoch}")
    os.makedirs(path, exist_ok=True)
    if mode == "lora":
        model.merge_adapter()
        model.base_model.model.save_pretrained(path, safe_serialization=True)
        model.unmerge_adapter()
        # save_pretrained 会把 LoRA 包装层的键名一起写出来（X.base_layer.weight），
        # vLLM 不认；这里就地重写成普通 HF 键名（数值不变）。
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from merge_lora import fix_dir
        fix_dir(path)
    else:
        model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    log(f"  已保存 {os.path.relpath(path, ROOT)}")
    return path


# ---------------------------------------------------------------- 训练

def lr_at(step, total, base_lr, warmup_ratio):
    warm = max(1, int(total * warmup_ratio))
    if step < warm:
        return base_lr * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    log(f"模型 {args.model}  mode={args.mode}")
    examples, skipped = load_examples(tok, args.data, args.max_len)
    if not examples:
        raise SystemExit(f"训练样本为 0（全部超长被丢弃 {skipped}）——检查 --max-len")
    log(f"训练样本 {len(examples)} 条（超长丢弃 {skipped}）")
    n_tok = sum(len(x[0]) for x in examples)
    log(f"序列长度 mean {n_tok / len(examples):.1f}，总 {n_tok / 1e6:.2f}M tok/epoch")

    model = load_model(args.model, args.mode, args.lora_r, args.lora_alpha,
                       args.lora_dropout)
    dev = "cuda"
    model.to(dev)
    if args.grad_ckpt:
        # 冻结底座 + checkpointing 时，输入必须 requires_grad，否则反传断链
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        log("  gradient checkpointing: on")
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    collate = Collator(tok.pad_token_id)
    steps_per_epoch = math.ceil(len(examples) / (args.batch * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    os.makedirs(args.out, exist_ok=True)
    history = []
    step = 0        # 微批计数（只用于日志）
    opt_step = 0    # 优化器步数（lr schedule 按这个走）
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(order)
        t0 = time.time()
        run_loss, run_n = 0.0, 0
        opt.zero_grad(set_to_none=True)

        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            batch = [examples[i] for i in idx]
            ids, labels, attn, tlen = collate(batch)
            ids, labels, attn = ids.to(dev), labels.to(dev), attn.to(dev)
            # 多留 1 位：第一个 target token 由前一个位置的 logits 预测
            keep = min(tlen + 1, labels.shape[1])

            for g in opt.param_groups:
                g["lr"] = lr_at(opt_step, total_steps, args.lr, args.warmup)
            out = model(input_ids=ids, attention_mask=attn,
                        labels=labels[:, -keep:], logits_to_keep=keep)
            loss = out.loss / args.grad_accum
            loss.backward()
            run_loss += out.loss.item()
            run_n += 1
            step += 1

            if step % args.grad_accum == 0 or b + args.batch >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

            if run_n % args.log_every == 0:
                log(f"  epoch {epoch} step {run_n}/{len(order) // args.batch + 1}"
                    f"  loss {run_loss / run_n:.4f}"
                    f"  lr {opt.param_groups[0]['lr']:.2e}"
                    f"  {time.time() - t0:.0f}s")

        rec = {"epoch": epoch, "loss": run_loss / max(1, run_n),
               "steps": opt_step, "seconds": round(time.time() - t0, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        history.append(rec)
        log(f"epoch {epoch} 完成  loss {rec['loss']:.4f}  "
            f"{rec['seconds']}s  峰值显存 {rec['peak_gb']} GB")
        save_checkpoint(model, tok, args.out, epoch, args.mode)

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"n_train": len(examples), "tokens_per_epoch": n_tok,
                "history": history, "lora_targets": LORA_TARGETS})
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log(f"\n日志 → {os.path.relpath(os.path.join(args.out, 'train_log.json'), ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen3-0.6B"))
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--mode", choices=["full", "lora"], default="lora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=None,
                    help="默认：full 2e-5 / lora 1e-4")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="梯度检查点（4B 在 24GB 上必开）")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="冒烟测试：只用前 N 条")
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 2e-5 if args.mode == "full" else 1e-4
    if args.limit:
        with open(args.data, encoding="utf-8") as f:
            lines = f.readlines()[:args.limit]
        tmp = os.path.join(args.out, "_smoke.jsonl")
        os.makedirs(args.out, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines)
        args.data = tmp

    train(args)


if __name__ == "__main__":
    main()
