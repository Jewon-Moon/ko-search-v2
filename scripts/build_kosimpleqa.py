#!/usr/bin/env python3
"""KoSimpleQA (naver-ai/KoSimpleQA, CC BY 4.0, 1,000 Korean factual questions) -> data/kosimpleqa/{dev,test}.jsonl.
Dev 50 (harness design only) / test 200 (run once at the end), stratified by category, fixed seed. The other 750 stay unused.
  python -I scripts/build_kosimpleqa.py
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "searchslm/data/external/kosimpleqa/kosimpleqa.json"
OUT = ROOT / "data/kosimpleqa"


def main(n_dev: int = 50, n_test: int = 200) -> None:
    rows = json.load(open(SRC, encoding="utf-8"))
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["category"]].append({"id": f"kosqa-{i:04d}", "type": "S", "kind": r["category"], "question": r["content"].strip(),
                                  "answer": str(r["answer"]).strip(), "gold": [], "valid": True, "d_none": False})
    rng = random.Random(20261119)
    for xs in by.values():
        rng.shuffle(xs)
    order = []  # round-robin over categories: any prefix keeps the category mix
    while any(by.values()):
        for c in sorted(by):
            if by[c]:
                order.append(by[c].pop())
    dev, test = order[:n_dev], order[n_dev:n_dev + n_test]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, xs in (("dev", dev), ("test", test)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(name, len(xs), dict(Counter(x["kind"] for x in xs)))


if __name__ == "__main__":
    main()
