#!/usr/bin/env python3
"""Filter, decontaminate, tokenize (tokenizer v2) and pack the fetched text into uint16 blocks for continued pretraining.

Per source: length/Hangul/repetition filters, exact dedup, long docs cut into <=20k-char pieces, then any doc sharing an 8-word
n-gram with an eval text is dropped (SearchSLM eval/tune/dev/sealed questions + gold passages, KLUE-MRC dev, KorQuAD dev,
Ko-StrategyQA). Docs are shuffled and taken until the source's token budget; 0.5% of docs (<=300k tokens) go to val.
Output: data/cpt/{train,val}.npy (n_blocks x 2048 uint16, EOS between docs) + data/cpt/build_report.json.
Self-contained so it can run with `python -I` (it reads downloaded files).
  python -I scripts/build_cpt.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "searchslm"
BUDGET = {"fineweb2": 60_000_000, "finepdfs": 20_000_000, "wiki_ko": 20_000_000}  # tokens
BLOCK = 2048
HANGUL = re.compile(r"[가-힣]")
LETTER = re.compile(r"[^\W\d_]")
WORDS = re.compile(r"\S+")


def ngrams(text: str, n: int = 8) -> set:
    w = WORDS.findall(text)
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def eval_ngrams() -> tuple[set, dict]:
    texts, count = [], {}
    gen = OLD / "data/search/gen"
    pids = set()
    for pat in ("items_v6_eval*.jsonl", "items_tune_h2.jsonl", "items_dev*.jsonl", "sealed_test_v1.jsonl", "items_heldout*.jsonl"):
        for f in gen.glob(pat):
            n0 = len(texts)
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                texts += [str(r.get("question", "")), str(r.get("question_orig", "")), str(r.get("answer", ""))]
                pids.update(r.get("gold", []) + r.get("positives", []) + ([r["topic_passage"]] if r.get("topic_passage") else []))
            count[f.name] = len(texts) - n0
    for layer in ("local", "web"):
        for l in open(OLD / f"data/search/corpus_v2/{layer}.jsonl", encoding="utf-8"):
            if '"passage_id"' in l:
                d = json.loads(l)
                if d["passage_id"] in pids:
                    texts.append(d["text"])
    count["gold_passages"] = len(pids)
    import pyarrow.parquet as pq
    for r in pq.read_table(OLD / "data/external/klue_mrc/validation.parquet", columns=["context", "question"]).to_pylist():
        texts += [r["context"], r["question"]]
    kq = json.load(open(OLD / "data/external/korquad/KorQuAD_v1.0_dev.json", encoding="utf-8"))
    for art in kq["data"]:
        for p in art["paragraphs"]:
            texts.append(p["context"]); texts += [q["question"] for q in p["qas"]]
    for f in ("ko_strategyqa/passages.jsonl", "ko_strategyqa/queries.jsonl"):
        for l in open(OLD / "data/external" / f, encoding="utf-8"):
            r = json.loads(l); texts.append(str(r.get("text") or r.get("contents") or r.get("query") or ""))
    out = set()
    for t in texts:
        out |= ngrams(unicodedata.normalize("NFC", t))
    count["eval_texts"] = len(texts); count["ngrams"] = len(out)
    return out, count


def keep(text: str) -> bool:
    if len(text) < 200:
        return False
    letters = LETTER.findall(text)
    if not letters or len(HANGUL.findall(text)) / len(letters) < 0.4:
        return False
    lines = [l for l in text.split("\n") if l.strip()]
    return len(lines) < 10 or len(set(lines)) / len(lines) >= 0.5


def pieces(text: str, size: int = 20000):
    for i in range(0, len(text), size):
        p = text[i:i + size]
        if len(p) >= 200:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", default="tokenizer/ko48k-v2-t100")
    ap.add_argument("--out", default="data/cpt")
    ap.add_argument("--budget", nargs=3, type=float, metavar=("FINEWEB2", "FINEPDFS", "WIKI"), help="token budgets (default 60M 20M 20M)")
    a = ap.parse_args()
    if a.budget:  # val docs are the same for any budget: shuffle seeds and filters do not depend on it
        BUDGET.update(zip(("fineweb2", "finepdfs", "wiki_ko"), (int(x) for x in a.budget)))
    tok = AutoTokenizer.from_pretrained(ROOT / a.tokenizer)
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    bad, ev = eval_ngrams()
    print("eval n-grams", ev, flush=True)
    rep, train_ids, val_ids = {"eval": ev, "sources": {}}, [], []
    for src, budget in BUDGET.items():
        seen, docs, st = set(), [], {"read": 0, "filtered": 0, "dup": 0, "contaminated": 0}
        for l in open(ROOT / f"data/raw/{src}.jsonl", encoding="utf-8"):
            st["read"] += 1
            t = unicodedata.normalize("NFC", json.loads(l)["text"]).strip()
            if not keep(t):
                st["filtered"] += 1; continue
            h = hashlib.md5(t.encode()).digest()
            if h in seen:
                st["dup"] += 1; continue
            seen.add(h)
            for p in pieces(t):
                if ngrams(p) & bad:
                    st["contaminated"] += 1
                else:
                    docs.append(p)
        random.Random(src).shuffle(docs)
        n_val = max(1, len(docs) // 200)
        vt = tt = 0
        for i, d in enumerate(docs):
            ids = tok(d, add_special_tokens=False)["input_ids"] + [eos]
            if i < n_val and vt < 300_000:
                val_ids += ids; vt += len(ids)
            else:
                train_ids += ids; tt += len(ids)
                if tt >= budget:
                    break
        st |= {"docs_kept": len(docs), "docs_used": i + 1, "train_tokens": tt, "val_tokens": vt}
        rep["sources"][src] = st
        print(src, st, flush=True)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train_ids), ("val", val_ids)):
        n = len(ids) // BLOCK
        arr = np.asarray(ids[: n * BLOCK], dtype=np.uint16).reshape(n, BLOCK)
        if name == "train":
            arr = arr[np.random.default_rng(0).permutation(n)]
        np.save(out / f"{name}.npy", arr)
        rep[name] = {"blocks": n, "tokens": n * BLOCK}
    assert len(tok) < 65536
    rep |= {"tokenizer": a.tokenizer, "block": BLOCK, "eos_id": eos, "budget": BUDGET}
    (out / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: rep[k] for k in ("train", "val")}), flush=True)


if __name__ == "__main__":
    main()
