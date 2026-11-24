#!/usr/bin/env python3
"""Full-parameter SFT of the continued-pretrained model on chat rows; loss only on assistant turns.

Row: {"id", "type", "messages": [{role, content}, ...]} (SearchSLM agent transcripts + plain chat). Rendering is ChatML, the same
as SearchSLM v4 (model/search_sft_train.py): <|im_start|>role\\ncontent<|im_end|>\\n, assistant body + <|im_end|> is the target.
fp32 master weights + bf16 autocast, AdamW, warmup + cosine to 10%, length-bucketed micro batches.
  python scripts/train_sft.py --init runs/stage2/final --data data/sft --out runs/sft --epochs 2 --lr 1e-5
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
IGNORE = -100


def encode(row: dict, tok) -> tuple[list[int], list[int]]:
    ids, labels = [], []
    for m in row["messages"]:
        if m["role"] == "assistant":
            head = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
            body = tok(m["content"] + "<|im_end|>", add_special_tokens=False).input_ids
            tail = tok("\n", add_special_tokens=False).input_ids
            ids += head + body + tail; labels += [IGNORE] * len(head) + body + [IGNORE] * len(tail)
        else:
            t = tok(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n", add_special_tokens=False).input_ids
            ids += t; labels += [IGNORE] * len(t)
    return ids, labels


def load(path: Path, tok, max_len: int):
    rows = [encode(json.loads(l), tok) for l in open(path, encoding="utf-8")]
    kept = [r for r in rows if len(r[0]) <= max_len and any(x != IGNORE for x in r[1])]
    return kept, len(rows) - len(kept)


def batches(rows, max_tokens: int, rng: random.Random):
    """Length-bucketed micro batches capped at max_tokens padded tokens; shuffled order."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]) + rng.random())
    out, cur, width = [], [], 0
    for i in order:
        w = max(width, len(rows[i][0]))
        if cur and w * (len(cur) + 1) > max_tokens:
            out.append(cur); cur, w = [], len(rows[i][0])
        cur.append(i); width = w
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(rows, idx, device):
    width = max(len(rows[i][0]) for i in idx)
    ids = torch.zeros(len(idx), width, dtype=torch.long); lab = torch.full((len(idx), width), IGNORE); att = torch.zeros(len(idx), width, dtype=torch.long)
    for r, i in enumerate(idx):
        n = len(rows[i][0]); ids[r, :n] = torch.tensor(rows[i][0]); lab[r, :n] = torch.tensor(rows[i][1]); att[r, :n] = 1
    return ids.to(device), lab.to(device), att.to(device)


def loss_sum(model, ids, lab, att):
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=att).logits
    lg = logits[:, :-1].float().reshape(-1, logits.shape[-1]); tg = lab[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(lg, tg, ignore_index=IGNORE, reduction="sum"), int((tg != IGNORE).sum())


def evaluate(model, rows, device, max_tokens):
    model.eval(); tot = n = 0
    with torch.no_grad():
        for idx in batches(rows, max_tokens, random.Random(0)):
            l, k = loss_sum(model, *collate(rows, idx, device)); tot += l.item(); n += k
    model.train()
    return round(tot / max(1, n), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rows-per-step", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16384, help="padded tokens per micro batch")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, help="smoke test")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ROOT / a.init)
    train, d_t = load(ROOT / a.data / "train.jsonl", tok, a.max_len)
    val, d_v = load(ROOT / a.data / "val.jsonl", tok, a.max_len)
    model = AutoModelForCausalLM.from_pretrained(ROOT / a.init, dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay = [p for p in model.parameters() if p.ndim >= 2]; no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), fused=device == "cuda")
    rng = random.Random(7)
    epoch_batches = [batches(train, a.max_tokens, rng) for _ in range(a.epochs)]
    per_step_mb = max(1, round(a.rows_per_step / (len(train) / len(epoch_batches[0]))))  # micro batches per optimizer step
    steps = [eb[i:i + per_step_mb] for eb in epoch_batches for i in range(0, len(eb), per_step_mb)]
    if a.max_steps:
        steps = steps[: a.max_steps]
    total, warm = len(steps), max(1, int(len(steps) * 0.03))
    cfg = {**vars(a), "device": device, "train_rows": len(train), "val_rows": len(val), "dropped": [d_t, d_v], "steps": total,
           "micro_batches_per_step": per_step_mb}
    (out / "config.json").write_text(json.dumps(cfg, indent=1)); print(json.dumps(cfg), flush=True)
    hist = [{"step": 0, "val_loss": evaluate(model, val, device, a.max_tokens)}]; print(hist[0], flush=True)
    t0 = time.time()
    for step, mbs in enumerate(steps, 1):
        f = step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g in opt.param_groups:
            g["lr"] = a.lr * f
        data = [collate(train, idx, device) for idx in mbs]
        n_tgt = sum(int((d[1][:, 1:] != IGNORE).sum()) for d in data)
        acc = 0.0
        for d in data:
            l, _ = loss_sum(model, *d); (l / n_tgt).backward(); acc += l.item() / n_tgt
        assert math.isfinite(acc), f"non-finite loss at step {step}"
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == total:
            el = time.time() - t0
            print(json.dumps({"step": step, "of": total, "loss": round(acc, 4), "gnorm": round(float(gn), 3), "eta_h": round((total - step) * el / step / 3600, 2)}), flush=True)
        if step % max(1, total // 4) == 0 or step == total:
            hist.append({"step": step, "val_loss": evaluate(model, val, device, a.max_tokens)}); print(hist[-1], flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
    model.to(torch.bfloat16).save_pretrained(out / "final", safe_serialization=True); tok.save_pretrained(out / "final")
    print(json.dumps({"done": True, "hours": round((time.time() - t0) / 3600, 2), "final_val": hist[-1]["val_loss"]}), flush=True)


if __name__ == "__main__":
    main()
