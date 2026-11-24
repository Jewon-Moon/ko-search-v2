#!/usr/bin/env python3
"""RL question pool for V2 (user 2026-11-19): short-answer factual questions over the local index (MIRACL ko wiki + web).
  human  MIRACL ko train questions with short answers (items_v6_train_human, real user questions)
  B/C/D  generated items over the local corpus (items_train*, valid only; D = hidden-document items, see search_rl.load_items)
  K      KorQuAD 1.0 train (optional, only if the file was downloaded), self-contained questions, short answers
Every KoSimpleQA question (all 1,000, not only dev/test) and every v6 eval question is excluded, plus near-duplicates
(char-bigram Jaccard >= 0.6). Output: data/rl_v2/items.jsonl (+ val.jsonl, 64 items).
  python -I scripts/build_rl_items.py
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "searchslm"
GEN = OLD / "data/search/gen"
OUT = ROOT / "data/rl_v2"
KORQUAD_TRAIN = OLD / "data/external/korquad/KorQuAD_v1.0_train.json"
NOSUBJ = re.compile(r"[가-힣]{1,3}(한|된|던|했던)\s")  # "추진한 교원개혁에서 ..." opens on a clause with no subject
DEICTIC = re.compile(r"(^|\s)(이|그|저|위|해당|이런|그런)\s|그는|그녀|이것|그것|이곳|그곳|여기|본문|위의|이들|그들")  # needs the paragraph
N_KORQUAD, N_GEN_D, N_GEN_BC = 1500, 200, 250  # generated items capped (~20%): passage-first style, unlike real questions
EXPLAIN = re.compile(r"어떤 내용|내용은|설명|무엇을 (다루|의미|규정)|왜|어떻게|이유|특징|차이")


def bigrams(s: str) -> set:
    s = re.sub(r"\W+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def korquad(rng: random.Random) -> list[dict]:
    if not KORQUAD_TRAIN.exists():
        print("KorQuAD train not downloaded: skipped")
        return []
    paras = [p for a in json.load(open(KORQUAD_TRAIN, encoding="utf-8"))["data"] for p in a["paragraphs"]]
    rng.shuffle(paras)
    out = []
    for p in paras:  # one question per paragraph
        qs = [qa for qa in p["qas"] if not DEICTIC.search(qa["question"]) and not NOSUBJ.match(qa["question"]) and len(qa["question"]) >= 12 and len(qa["answers"][0]["text"]) <= 20]
        if qs:
            qa = rng.choice(qs)
            out.append({"id": f"rl-kq-{qa['id']}", "type": "K", "kind": "korquad", "question": qa["question"], "answer": qa["answers"][0]["text"],
                        "gold": [], "in_corpus": True, "d_none": False, "valid": True})
        if len(out) == N_KORQUAD:
            break
    return out


def main() -> None:
    rng = random.Random(20261119)
    banned = [r["content"] for r in json.load(open(OLD / "data/external/kosimpleqa/kosimpleqa.json", encoding="utf-8"))]
    banned += [r["question"] for f in GEN.glob("items_v6_eval*.jsonl") for r in rows(f)]
    banned_bg = [bigrams(q) for q in banned]
    banned_exact = {re.sub(r"\W+", "", q) for q in banned}

    human = [dict(r, kind="human") for r in rows(GEN / "items_v6_train_human.jsonl") if r.get("valid")]
    gen = [r for f in ("items_train.jsonl", "items_train2.jsonl", "items_train_d4.jsonl") for r in rows(GEN / f)
           if r.get("valid") and r.get("type") in ("B", "C", "D") and len(str(r.get("answer", ""))) <= 30]
    rng.shuffle(gen)
    # rule reward needs one short entity answer: explain-type questions and list answers give false "wrong" (-0.5) rewards
    gen = [r for r in gen if len(str(r["answer"])) <= 15 and "," not in str(r["answer"]) and not EXPLAIN.search(r["question"])]
    human = [r for r in human if len(str(r["answer"])) <= 15 and "," not in str(r["answer"])]
    # questions V2 already saw in SFT carry little RL signal (memorized trajectories): 70% of the first pool
    sft = {re.sub(r"\W+", "", m["content"].split(":", 1)[-1]) for l in open(ROOT / "data/sft_v2/train.jsonl", encoding="utf-8")
           for m in json.loads(l)["messages"] if m["role"] == "user" and m["content"].startswith("질문")}
    pool, seen = [], set()
    for r in human + gen + korquad(rng):
        key = re.sub(r"\W+", "", r["question"])
        if key in seen or key in banned_exact or key in sft:
            continue
        bg = bigrams(r["question"])
        if any(jaccard(bg, b) >= 0.6 for b in banned_bg):
            continue
        seen.add(key); pool.append(r)
    pool = [r for r in pool if r.get("kind") != "human" and r.get("type") == "D"][:N_GEN_D] + \
           [r for r in pool if r.get("kind") != "human" and r.get("type") in ("B", "C")][:N_GEN_BC] + \
           [r for r in pool if r.get("kind") == "human" or r.get("type") == "K"]
    rng.shuffle(pool)
    val, train = pool[:64], pool[64:]
    assert not {re.sub(r"\W+", "", r["question"]) for r in pool} & (banned_exact | sft)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rs in (("items", train), ("val", val)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rs), dict(Counter(r.get("kind") if r.get("kind") in ("human", "korquad") else r["type"] for r in rs)))


if __name__ == "__main__":
    main()
