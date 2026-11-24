#!/usr/bin/env python3
"""Live-search eval set (user 2026-11-19): 300 items -> data/live_eval/{dev,test}.jsonl (100 / 200, stratified by source).
  K  KorQuAD 1.0 dev questions (140; stable wiki facts, gold = first annotated answer, self-contained questions only)
  V  v6 answerable items (100; B type: explain / namesake / human, proportional)
  R  2026 events from ko.wikipedia "2026년" (60; hand-written, data/live_eval/realtime_2026.jsonl)
Dev is for harness design only; test is run once at the end. Grading: search_baseline.score (teacher judge, T.same).
  python -I scripts/build_live_eval.py
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "searchslm"
OUT = ROOT / "data/live_eval"
DEICTIC = re.compile(r"(^|\s)(이|그|저|위|해당|이런|그런)\s|그는|그녀|이것|그것|이곳|그곳|여기|본문|위의|이들|그들")  # needs the KorQuAD paragraph


def korquad(n: int, rng: random.Random) -> list[dict]:
    d = json.load(open(OLD / "data/external/korquad/KorQuAD_v1.0_dev.json", encoding="utf-8"))
    paras = [p for a in d["data"] for p in a["paragraphs"]]
    rng.shuffle(paras)
    out = []
    for p in paras:  # one question per paragraph
        qs = [qa for qa in p["qas"] if not DEICTIC.search(qa["question"]) and len(qa["question"]) >= 12]
        if qs:
            qa = rng.choice(qs)
            out.append({"id": f"live-kq-{qa['id']}", "type": "K", "kind": "korquad", "question": qa["question"], "answer": qa["answers"][0]["text"]})
        if len(out) == n:
            return out
    raise ValueError("not enough KorQuAD questions")


def v6(n: int, rng: random.Random) -> list[dict]:
    items = [json.loads(l) for l in open(OLD / "data/search/gen/items_v6_eval_all.jsonl", encoding="utf-8")]
    items = [it for it in items if it["valid"] and it["type"] == "B"]
    by = {}
    for it in items:
        by.setdefault(it["kind"], []).append(it)
    out = []
    for kind, xs in sorted(by.items()):
        rng.shuffle(xs)
        out += xs[: round(n * len(xs) / len(items))]
    return [{"id": "live-" + it["id"], "type": "V", "kind": it["kind"], "question": it["question"], "answer": it["answer"]} for it in out[:n]]


def realtime() -> list[dict]:
    return [{"id": r["id"], "type": "R", "kind": "realtime_2026", "question": r["question"], "answer": r["answer"]}
            for r in map(json.loads, open(OUT / "realtime_2026.jsonl", encoding="utf-8"))]


def main() -> None:
    rng = random.Random(20261119)
    groups = [korquad(140, rng), v6(100, rng), realtime()]
    dev, test = [], []
    for g in groups:
        rng.shuffle(g)
        k = round(len(g) / 3)
        dev += g[:k]; test += g[k:]
    for it in dev + test:
        it.update(gold=[], valid=True, d_none=False)
    assert len({it["question"] for it in dev + test}) == len(dev) + len(test), "duplicate question"
    for name, rows in (("dev", dev), ("test", test)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), dict(Counter(r["type"] for r in rows)))


if __name__ == "__main__":
    main()
