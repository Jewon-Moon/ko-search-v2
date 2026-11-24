#!/usr/bin/env python3
"""Continued pretraining of the transplanted model on data/cpt blocks.

stage 1: only the tied embedding trains (the new 48k rows learn to speak the frozen blocks' language), as in the ko32k recipe.
stage 2: everything trains, embedding at a higher LR than the blocks (it is still catching up).
fp32 master weights + bf16 autocast, AdamW, warmup + cosine to 10%, grad clip 1.0. Logs tokens/s and val loss; saves bf16 HF dirs.
  python scripts/train_cpt.py --stage 1 --init models/q35-2b-ko48k-init --out runs/stage1 --tokens 10e6 --lr 2e-4
  python scripts/train_cpt.py --stage 2 --init runs/stage1/final --out runs/stage2 --tokens 100e6 --lr 2e-5 --emb-lr 5e-5
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def val_loss(model, val: np.ndarray, device: str, micro: int, n_blocks: int = 64) -> float:
    model.eval(); tot = cnt = 0.0
    with torch.no_grad():
        for i in range(0, min(n_blocks, len(val)), micro):
            x = torch.from_numpy(val[i:i + micro].astype(np.int64)).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                l = model(x, labels=x).loss
            tot += l.item() * len(x); cnt += len(x)
    model.train()
    return round(tot / cnt, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/cpt")
    ap.add_argument("--tokens", type=float, required=True, help="training tokens for this stage")
    ap.add_argument("--lr", type=float, required=True, help="block LR (stage 2) or embedding LR (stage 1)")
    ap.add_argument("--emb-lr", type=float, help="stage 2 embedding LR (default = --lr)")
    ap.add_argument("--tokens-per-step", type=int, default=65536)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=0, help="also save a checkpoint every N steps (0 = only final)")
    ap.add_argument("--max-steps", type=int, help="smoke test: stop early")
    ap.add_argument("--skip-blocks", type=int, default=0, help="start reading train blocks here (stage 2 can skip stage 1's)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    train = np.load(ROOT / a.data / "train.npy", mmap_mode="r"); val = np.load(ROOT / a.data / "val.npy", mmap_mode="r")
    block = train.shape[1]
    try:
        import fla  # noqa: F401  (flash-linear-attention: fast Gated DeltaNet kernels; without it training is several x slower)
        kernels = "fla"
    except ImportError:
        kernels = "torch-fallback"
    model = AutoModelForCausalLM.from_pretrained(ROOT / a.init, dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    emb = model.get_input_embeddings().weight
    assert emb.data_ptr() == model.get_output_embeddings().weight.data_ptr(), "embedding must stay tied"
    for p in model.parameters():
        p.requires_grad_(a.stage == 2)
    emb.requires_grad_(True)
    if a.stage == 1:
        groups = [{"params": [emb], "lr": a.lr, "weight_decay": 0.0}]
    else:
        decay = [p for n, p in model.named_parameters() if p is not emb and p.ndim >= 2]
        no_decay = [p for n, p in model.named_parameters() if p is not emb and p.ndim < 2]
        groups = [{"params": [emb], "lr": a.emb_lr or a.lr, "weight_decay": 0.0},
                  {"params": decay, "lr": a.lr, "weight_decay": 0.01}, {"params": no_decay, "lr": a.lr, "weight_decay": 0.0}]
    for g in groups:
        g["peak"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=device == "cuda")
    per_step = a.tokens_per_step // block
    total = math.ceil(a.tokens / (per_step * block))
    if a.max_steps:
        total = min(total, a.max_steps)
    need = a.skip_blocks + total * per_step
    assert need <= len(train), f"needs {need} blocks, data has {len(train)}"
    warm = max(1, int(total * a.warmup))
    cfg = {**vars(a), "device": device, "kernels": kernels, "steps": total, "blocks_per_step": per_step, "block": block,
           "trainable": sum(p.numel() for g in groups for p in g["params"])}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg), flush=True)
    hist = [{"step": 0, "val_loss": val_loss(model, val, device, a.micro)}]
    print(hist[0], flush=True)
    t0, seen = time.time(), 0
    for step in range(1, total + 1):
        f = step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        for g in opt.param_groups:
            g["lr"] = g["peak"] * f
        s0 = a.skip_blocks + (step - 1) * per_step
        acc = 0.0
        for m in range(0, per_step, a.micro):
            x = torch.from_numpy(train[s0 + m: s0 + min(m + a.micro, per_step)].astype(np.int64)).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                loss = model(x, labels=x).loss * len(x) / per_step
            loss.backward(); acc += loss.item()
        assert math.isfinite(acc), f"non-finite loss at step {step}"
        gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        seen += per_step * block
        if step % 10 == 0 or step == total:
            el = time.time() - t0
            print(json.dumps({"step": step, "of": total, "loss": round(acc, 4), "gnorm": round(float(gn), 3), "lr_f": round(f, 3),
                              "tok_s": round(seen / el), "eta_h": round((total - step) * el / step / 3600, 2)}), flush=True)
        if step % a.eval_every == 0 or step == total:
            hist.append({"step": step, "tokens": seen, "val_loss": val_loss(model, val, device, a.micro)})
            print(hist[-1], flush=True)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
        if a.save_every and step % a.save_every == 0 and step != total:
            sd = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items() if k != "lm_head.weight"}  # tied; keep fp32 master
            model.save_pretrained(out / f"step{step}", state_dict=sd, safe_serialization=True)
    final = out / "final"
    model.to(torch.bfloat16).save_pretrained(final, safe_serialization=True)
    AutoTokenizer.from_pretrained(ROOT / a.init).save_pretrained(final)
    print(json.dumps({"done": True, "hours": round((time.time() - t0) / 3600, 2), "final_val": hist[-1]["val_loss"]}), flush=True)


if __name__ == "__main__":
    main()
