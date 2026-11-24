#!/usr/bin/env python3
"""KO tokenizer v2: word-level Korean BPE whose word tokens do not depend on what precedes the word.

Changes from ko32k (outputs/real/mf2/tokenizer-ko-32000):
- pre-tokenizer: single digits + GPT-2 regex (llama.cpp pre-type "refact"/"starcoder", also EXAONE's), so a quote or bracket
  never glues onto a word ('"이' tokens are impossible); only a space may lead a word.
- twins: --twin-rate r appends r of the space-led words again without the space (newline-separated), so frequent words get a
  no-space twin and a word copied into a JSON string can reuse the word token instead of being re-spelled.
- corpus: AI-Hub-free (FineWeb-2 ko, MIRACL-ko wiki, FineWeb ko-en mixed, StarCoder JSON/code), so the tokenizer can travel.
  python tokenizer/train_v2.py --vocab 32000 --twin-rate 0.1 --out outputs/tokenizer_v2/ko32k-v2-t10
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from tokenizers import Regex, pre_tokenizers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import CONTROL_TOKENS, assert_control_tokens_atomic, non_special_added_tokens, normalize_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WIKI_EVAL_MOD = 50  # every 50th wiki doc is held out for the benchmark
SOURCES = [  # (path, kind, max characters)
    ("data/processed/fineweb_general_train.txt", "txt", None),
    ("data/search/corpus_v2/local.jsonl", "wiki", 120_000_000),
    ("data/processed/mixed/korean_english_mixed.txt", "txt", None),
    ("data/processed/structured/json_extra_train.txt", "txt", None),
    ("data/processed/structured/structured_train.txt", "txt", None),
    ("data/processed/mixed/source_code.txt", "txt", 12_000_000),
]
WORD = re.compile(r"(?<= )([^\W\d_]{2,})")


def docs():
    for path, kind, cap in SOURCES:
        n = 0
        for i, line in enumerate(open(ROOT / path, encoding="utf-8")):
            if kind == "wiki":
                r = json.loads(line)
                if int(r["doc_id"].split(":")[1]) % WIKI_EVAL_MOD == 0:
                    continue
                line = r["text"]
            t = normalize_text(line)
            if not t:
                continue
            yield t
            n += len(t)
            if cap and n >= cap:
                break


def with_twins(texts, rate: float, seed: int = 0):
    rng = random.Random(seed)
    for t in texts:
        yield t
        if rate:
            tw = [w for w in WORD.findall(t) if rng.random() < rate]
            if tw:
                yield "\n".join(tw)


def batches(it, n=1000):
    b = []
    for x in it:
        b.append(x)
        if len(b) == n:
            yield b; b = []
    if b:
        yield b


def pre_v2():
    return pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--twin-rate", type=float, default=0.1)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B", help="special tokens + chat template donor (Qwen3-1.7B shares it)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = AutoTokenizer.from_pretrained(a.base)
    base.backend_tokenizer.pre_tokenizer = pre_v2()  # train_new_from_iterator keeps the backend's pre-tokenizer
    keep = non_special_added_tokens(base)
    tok = base.train_new_from_iterator(batches(with_twins(docs(), a.twin_rate)), vocab_size=a.vocab - len(keep),
                                       new_special_tokens=CONTROL_TOKENS, min_frequency=a.min_frequency)
    tok.add_special_tokens({"additional_special_tokens": list(dict.fromkeys([str(x) for x in tok.additional_special_tokens] + CONTROL_TOKENS))},
                           replace_additional_special_tokens=True)
    tok.add_tokens(keep, special_tokens=False)
    assert_control_tokens_atomic(tok)
    assert tok.backend_tokenizer.pre_tokenizer.__getstate__() == pre_v2().__getstate__(), "pre-tokenizer was not kept"
    out = ROOT / a.out
    tok.save_pretrained(out)
    meta = {"vocab_requested": a.vocab, "vocab_actual": len(tok), "twin_rate": a.twin_rate, "min_frequency": a.min_frequency,
            "pre_tokenizer": "Digits(individual) + ByteLevel GPT-2 regex (llama.cpp refact/starcoder)", "base": a.base,
            "sources": SOURCES, "wiki_eval_mod": WIKI_EVAL_MOD}
    (out / "training_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
