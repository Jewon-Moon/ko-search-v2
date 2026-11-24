#!/usr/bin/env python3
"""Recovery check: NLL per character on held-out text (data/cpt/val.npy decoded), base Qwen3.5 with its own tokenizer vs our
checkpoints with tokenizer v2 — per-character so different tokenizers compare fairly. Also a few greedy continuations.
  python scripts/eval_cpt.py --models models/q35-2b-ko48k-init runs/stage2/final --n 40
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PROMPTS = ["대한민국의 수도는", "이순신은 조선 중기의", "김치찌개를 맛있게 끓이려면", '{"think": "고유명사가 있는 사실 질문이라 검색합니다.", "action": "search", "query": "']


def texts(n: int) -> list[str]:
    tok = AutoTokenizer.from_pretrained(ROOT / "tokenizer/ko48k-v2-t100"); eos = tok.convert_tokens_to_ids("<|endoftext|>")
    flat = np.load(ROOT / "data/cpt/val.npy").reshape(-1).tolist()
    docs, cur = [], []
    for t in flat:
        if t == eos:
            if cur: docs.append(tok.decode(cur)); cur = []
        else:
            cur.append(t)
    docs = [d for d in docs if len(d) > 300]
    return [d[:2000] for d in docs[:: max(1, len(docs) // n)][:n]]


@torch.no_grad()
def nll_char(model, tok, ts, dev):
    tot = ch = 0.0
    for t in ts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        tot += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1); ch += len(t)
    return round(tot / ch, 4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--base", default="models/Qwen3.5-2B-Base"); a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; ts = texts(a.n); res = {"texts": len(ts), "chars": sum(map(len, ts))}
    from transplant import load_text_model
    m = load_text_model(ROOT / a.base).to(dev).eval(); res["base_qwen35"] = nll_char(m, AutoTokenizer.from_pretrained(ROOT / a.base), ts, dev); del m
    for p in a.models:
        tok = AutoTokenizer.from_pretrained(ROOT / p); m = AutoModelForCausalLM.from_pretrained(ROOT / p, dtype=torch.bfloat16).to(dev).eval()
        r = {"nll_per_char": nll_char(m, tok, ts, dev), "samples": {}}
        for q in PROMPTS:
            ids = tok(q, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            out = m.generate(ids, max_new_tokens=40, do_sample=False)
            r["samples"][q] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        res[p] = r; del m
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
