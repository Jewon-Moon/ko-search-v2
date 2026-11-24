#!/usr/bin/env python3
"""Benchmark tokenizer v2 candidates against ko32k and Qwen3 on what the v2 changes are for.

Metrics (held-out text only; wiki docs with doc_id % 50 == 0 were never trained on):
- tok/100ch: tokens per 100 characters (Korean general, wiki, code/JSON) -> CPU prefill cost
- consistent: share of word occurrences segmented identically after a space and after a quote (the JSON-copy case),
  weighted by frequency; first_same: only the first piece compared (the first-syllable-drop case)
- syll_whole: share of the 2,350 KS X 1001 Hangul syllables that are a single no-space token
- roundtrip: decode(encode(x)) == x on every held-out line
  python tokenizer/eval_v2.py outputs/tokenizer_v2/ko32k-v2-t10 ...
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import normalize_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = {"qwen3": "Qwen/Qwen3-0.6B", "ko32k": "outputs/release/qwen3-0.6b-ko32k-stage2-signal2678-base"}


def eval_sets() -> dict[str, list[str]]:
    wiki = []
    for line in open(ROOT / "data/search/corpus_v2/local.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if int(r["doc_id"].split(":")[1]) % 50 == 0:
            wiki.append(normalize_text(r["text"]))
            if len(wiki) >= 3000:
                break
    rd = lambda p: [t for t in (normalize_text(l) for l in open(ROOT / p, encoding="utf-8")) if t]
    return {"ko_web": rd("data/processed/fineweb_general_holdout.txt") + rd("data/korean_eval.txt"), "ko_wiki": wiki,
            "json_code": rd("data/processed/structured/json_extra_valid.txt")[:3000] + rd("data/processed/structured/structured_valid.txt")[:2000]}


def syllables_2350() -> list[str]:
    out = []
    for c in range(0xAC00, 0xD7A4):
        try:
            chr(c).encode("euc_kr"); out.append(chr(c))
        except UnicodeEncodeError:
            pass
    return out


def pieces(tok, text: str) -> list[str]:
    return [tok.convert_tokens_to_string([t]) for t in tok.tokenize(text)]


def consistency(tok, words: Counter) -> tuple[float, float]:
    same = first = total = 0
    for w, n in words.items():
        a = pieces(tok, " " + w); b = pieces(tok, '"' + w)
        if b and b[0].startswith('"'):  # strip the quote (alone, or glued onto the first piece as in ko32k)
            b = ([b[0][1:]] if len(b[0]) > 1 else []) + b[1:]
        a = [a[0].lstrip(" ")] + a[1:] if a else a
        a = [x for x in a if x]
        same += n * (a == b); first += n * (bool(a) and bool(b) and a[0] == b[0]); total += n
    return same / total, first / total


def main() -> None:
    sets = eval_sets()
    words = Counter(w for t in sets["ko_web"] + sets["ko_wiki"] for w in re.findall(r"(?<= )([가-힣]{2,})", t))
    top = Counter(dict(words.most_common(20000)))
    syl = syllables_2350()
    cands = {**REF, **{Path(p).name: p for p in sys.argv[1:]}}
    rows = {}
    for name, path in cands.items():
        tok = AutoTokenizer.from_pretrained(path)
        r = {"vocab": len(tok)}
        for k, lines in sets.items():
            r[f"{k} tok/100ch"] = round(100 * sum(len(tok(x, add_special_tokens=False)["input_ids"]) for x in lines) / sum(map(len, lines)), 1)
        r["consistent"], r["first_same"] = (round(x, 3) for x in consistency(tok, top))
        r["syll_whole"] = round(sum(len(tok.tokenize(s)) == 1 for s in syl) / len(syl), 3)
        r["roundtrip"] = all(tok.decode(tok(x, add_special_tokens=False)["input_ids"]) == x for x in sets["ko_web"][:300] + sets["json_code"][:300])
        r["examples"] = {s: pieces(tok, s) for s in ('"이순신 생년"', " 이순신", '{"action": "search", "query": "김치찌개"}')}
        rows[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "examples"}, ensure_ascii=False), flush=True)
    out = ROOT / "outputs/tokenizer_v2/benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"words_scored": len(top), "word_occurrences": sum(top.values()), "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
