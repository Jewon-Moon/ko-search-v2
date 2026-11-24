#!/usr/bin/env python3
"""Assemble data/sft/{train,val}.jsonl from:
  - SearchSLM v4 agent rows (raw display, 3,822 train / 332 val; teacher Gemma 4 = Apache-2.0)
  - new accepted trajectories from scripts/gen_sft_traj.py (curation marks stripped -> same raw display as v4)
  - plain Korean chat (aya_dataset Korean rows, oasst2 ko root prompt + best reply), capped, with a generic system prompt
Run with python -I (reads downloaded files).
  python -I scripts/assemble_sft.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "searchslm"
MARK = re.compile(r" · (?:최신|이전 기록|\[\d+\]번과 중복)\)")
MARK_LINE = "같은 주제의 메모·대화 기록이 여럿이면 날짜에 '최신' 또는 '이전 기록'이 붙고, 거의 같은 내용이면 '[n]번과 중복'이 붙습니다.\n"
CHAT_SYS = "당신은 도움이 되는 한국어 어시스턴트입니다."


def unmark(m: dict) -> dict:  # same as searchslm/corpus/search_sft_export.py
    if m["role"] == "system":
        return m | {"content": m["content"].replace(MARK_LINE, "")}
    if m["role"] == "user" and m["content"].startswith("[검색 결과]"):
        return m | {"content": MARK.sub(")", m["content"])}
    return m


def chat_rows(cap: int, rng: random.Random) -> list[dict]:
    import pyarrow.parquet as pq
    raw = ROOT / "data/sft_raw"
    rows = []
    aya = pq.read_table(raw / "aya_dataset/data/train-00000-of-00001.parquet", columns=["inputs", "targets", "language"]).to_pylist()
    rows += [("aya", r["inputs"], r["targets"]) for r in aya if r["language"] == "Korean" and r["inputs"] and r["targets"]]
    msgs = []
    for f in sorted((raw / "oasst2/data").glob("*.parquet")):
        msgs += pq.read_table(f, columns=["message_id", "parent_id", "text", "role", "lang", "rank"]).to_pylist()
    by_id = {m["message_id"]: m for m in msgs}
    best = {}
    for m in msgs:
        p = by_id.get(m["parent_id"]) if m["parent_id"] else None
        if m["role"] == "assistant" and m["lang"] == "ko" and p and p["parent_id"] is None and p["lang"] == "ko":
            if p["message_id"] not in best or (m["rank"] is not None and (best[p["message_id"]]["rank"] is None or m["rank"] < best[p["message_id"]]["rank"])):
                best[p["message_id"]] = m
    rows += [("oasst2", by_id[pid]["text"], m["text"]) for pid, m in best.items()]
    rng.shuffle(rows)
    return [{"id": f"chat-{src}-{i:05d}", "type": "chat", "messages": [{"role": "system", "content": CHAT_SYS}, {"role": "user", "content": q},
                                                                        {"role": "assistant", "content": a}]} for i, (src, q, a) in enumerate(rows[:cap])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", nargs="+", default=["data/sft_traj/traj.jsonl"])
    ap.add_argument("--cap", nargs="*", default=[], help="kind=N caps on new trajectories, e.g. klue_ans=3000")
    ap.add_argument("--chat-cap", type=int, default=1200)
    ap.add_argument("--out", default="data/sft")
    a = ap.parse_args()
    rng = random.Random(11)
    v4t = [json.loads(l) for l in open(OLD / "data/search/sft_v4/train.jsonl", encoding="utf-8")]
    v4v = [json.loads(l) for l in open(OLD / "data/search/sft_v4/val.jsonl", encoding="utf-8")]
    new, seen_q = [], set()
    for t in a.traj:
        for l in open(ROOT / t, encoding="utf-8"):
            r = json.loads(l)
            q = r["item"]["question"]
            if r["accepted"] and r["messages"] and q not in seen_q:  # a question generated twice keeps its first trajectory
                seen_q.add(q)
                new.append({"id": r["id"], "type": r["kind"], "messages": [unmark(m) for m in r["messages"] if m["content"]]})
    rng.shuffle(new)
    caps = {k: int(v) for k, v in (c.split("=") for c in a.cap)}
    kept_n = Counter()
    new = [r for r in new if (kept_n.update([r["type"]]) or kept_n[r["type"]] <= caps.get(r["type"], 10**9))]
    n_val = max(1, len(new) // 20)
    chat = chat_rows(a.chat_cap, rng)
    c_val = max(1, len(chat) // 20)
    train = v4t + new[n_val:] + chat[c_val:]; val = v4v + new[:n_val] + chat[:c_val]
    rng.shuffle(train)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"train": len(train), "val": len(val), "train_types": dict(Counter(r["type"] for r in train)),
           "sources": {"v4": "SearchSLM sft_v4 (teacher Gemma 4 26B-A4B, Apache-2.0; passages MIRACL/Wikipedia CC BY-SA, FineWeb ODC-By)",
                       "new": "gen_sft_traj.py over KLUE-MRC (CC BY-SA 4.0), TyDi QA (Apache-2.0), MKQA (CC BY-SA 3.0)",
                       "chat": "aya_dataset (Apache-2.0), oasst2 (Apache-2.0)"}}
    (out / "assemble_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
