#!/usr/bin/env python3
"""Feasibility check of a transplanted model: NLL per character vs the original base, one train step, MPS train throughput.
  python scripts/check_model.py --model models/q35-2b-ko48k-init --base models/Qwen3.5-2B-Base
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TEXTS = [
    "대한민국의 수도는 서울특별시이며, 행정 구역상 특별시로 분류된다. 서울은 한강을 끼고 있으며 조선 시대부터 수도 역할을 해 왔다.",
    "이순신은 조선 중기의 무신으로, 임진왜란 때 수군을 이끌고 여러 해전에서 승리하였다. 그는 1545년에 한성에서 태어났다.",
    "김치찌개는 김치와 돼지고기, 두부 등을 넣고 끓인 한국의 대표적인 찌개 요리이다. 잘 익은 김치를 쓰면 맛이 깊어진다.",
    '{"think": "고유명사가 있는 사실 질문이라 검색합니다.", "action": "search", "query": "이순신 생년"}',
]


def nll_per_char(model, tok, device) -> float:
    tot, chars = 0.0, 0
    for t in TEXTS:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        tot += out.loss.item() * (ids.shape[1] - 1); chars += len(t)
    return tot / chars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=1)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    res = {}
    if a.base:
        bt = AutoTokenizer.from_pretrained(ROOT / a.base)
        from transplant import load_text_model
        bm = load_text_model(ROOT / a.base).to(dev).eval()
        res["base_nll_per_char"] = round(nll_per_char(bm, bt, dev), 4)
        del bm
        if dev == "mps":
            torch.mps.empty_cache()
    tok = AutoTokenizer.from_pretrained(ROOT / a.model)
    model = AutoModelForCausalLM.from_pretrained(ROOT / a.model, dtype=torch.bfloat16).to(dev)
    res["model_class"] = type(model).__name__
    res["tied"] = model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    model.eval(); res["init_nll_per_char"] = round(nll_per_char(model, tok, dev), 4)
    res["tokens_per_100ch"] = round(100 * sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in TEXTS) / sum(map(len, TEXTS)), 1)
    # one full-parameter train step at the planned length; bf16 weights, fp32 AdamW states would be the real setup
    model.train(); model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    x = torch.randint(0, len(tok), (a.micro, a.seq), device=dev)
    times = []
    for i in range(3):
        t0 = time.time()
        loss = model(x, labels=x).loss
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if dev == "mps":
            torch.mps.synchronize()
        times.append(time.time() - t0)
        assert math.isfinite(loss.item())
    step = min(times[1:])
    res["train_step_s"] = round(step, 2)
    res["train_tokens_per_s"] = round(a.micro * a.seq / step)
    res["hours_for_100M_tokens_on_this_mac"] = round(1e8 / res["train_tokens_per_s"] / 3600, 1)
    if dev == "mps":
        res["mps_peak_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
