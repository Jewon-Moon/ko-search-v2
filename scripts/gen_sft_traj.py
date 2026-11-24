#!/usr/bin/env python3
"""New SFT trajectories from open data, with the SearchSLM teacher pipeline (Gemma 4 26B, Apache-2.0, LM Studio)
and its fixed search index (MIRACL-ko wiki + FineWeb-2 sample). Run with the old project's venv (it has the retriever stack):
  searchslm/.venv/bin/python scripts/gen_sft_traj.py --klue 1400 --tydi 600 --mkqa 600

kinds:
  klue_ans / klue_imp  KLUE-MRC train: the question's passage (550-char window) is injected into the FIRST search result block at a
                       random rank; impossible items (same topic, answer absent) must end in "찾을 수 없습니다" after a web search.
  tydi                 TyDi QA GoldP Korean train: same injection, answerable only.
  mkqa                 MKQA ko (CC BY-SA 3.0): plain search over the index; only correct + grounded trajectories are kept.
Every question that also appears in an eval set (SearchSLM eval/tune/dev/sealed, KLUE dev, KorQuAD dev, Ko-StrategyQA) is dropped.
Output: data/sft_traj/traj.jsonl (one row per item, with grades) — resumable.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import random
import re
import sys
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "searchslm"
sys.path.insert(0, str(OLD / "corpus"))
import search_teacher as T  # noqa: E402

RAW = ROOT / "data/sft_raw"
_local = threading.local()
_orig_search = T.search


def norm_q(q: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(q)))


def qhash(q: str) -> str:
    return hashlib.sha1(norm_q(q).encode()).hexdigest()


def eval_questions() -> set:
    """sha1 of normalized eval questions. On a remote box only data/eval_hashes.txt is shipped (eval items never leave the Mac)."""
    hf = ROOT / "data/eval_hashes.txt"
    if hf.exists():
        return set(hf.read_text().split())
    qs = set()
    for f in (OLD / "data/search/gen").glob("*.jsonl"):
        if any(s in f.name for s in ("eval", "tune", "dev", "sealed", "heldout")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); qs.add(norm_q(r.get("question", ""))); qs.add(norm_q(r.get("question_orig", "")))
    import pyarrow.parquet as pq
    qs |= {norm_q(q) for q in pq.read_table(OLD / "data/external/klue_mrc/validation.parquet", columns=["question"]).column("question").to_pylist()}
    kq = json.load(open(OLD / "data/external/korquad/KorQuAD_v1.0_dev.json", encoding="utf-8"))
    qs |= {norm_q(q["question"]) for a in kq["data"] for p in a["paragraphs"] for q in p["qas"]}
    for l in open(OLD / "data/external/ko_strategyqa/queries.jsonl", encoding="utf-8"):
        r = json.loads(l); qs.add(norm_q(r.get("text") or r.get("query") or ""))
    qs.discard("")
    return {qhash(q) for q in qs}


def window(ctx: str, answer: str | None, question: str, size: int = 550) -> str:
    """A <=size-char slice: around the answer if given, else the slice sharing most question bigrams."""
    if len(ctx) <= size:
        return ctx
    if answer and answer in ctx:
        i = ctx.index(answer); s = max(0, min(len(ctx) - size, i - size // 2))
        return ctx[s:s + size]
    qb = {question[i:i + 2] for i in range(len(question) - 1)}
    best = max(range(0, len(ctx) - size + 1, 100), key=lambda s: len(qb & {ctx[s + i:s + i + 2] for i in range(size - 1)}))
    return ctx[best:best + size]


def inj_search(name, q, k=T.K, exclude=None):
    hits = _orig_search(name, q, k, exclude)
    inj = getattr(_local, "inject", None)
    if inj and not getattr(_local, "used", False):  # only the first search block carries the item's passage
        _local.used = True
        hits = [h for h in hits if h["passage_id"] != inj["passage_id"]][: k - 1]
        hits.insert(getattr(_local, "rank", 0), inj)
    return hits


T.search = inj_search


def items(a) -> list[dict]:
    import pyarrow.parquet as pq
    rng = random.Random(17); evq = eval_questions(); out = []
    klue = pq.read_table(RAW / "klue/mrc/train-00000-of-00001.parquet").to_pylist(); rng.shuffle(klue)
    n_imp = n_ans = 0
    for r in klue:
        if qhash(r["question"]) in evq:
            continue
        imp = str(r["is_impossible"]) == "True"
        if (imp and n_imp >= (a.klue_imp if a.klue_imp is not None else a.klue // 2)) or (not imp and n_ans >= (a.klue_ans if a.klue_ans is not None else a.klue // 2)):
            continue
        av = r["answers"] if isinstance(r["answers"], dict) else ast.literal_eval(r["answers"])
        ans = None if imp else av["text"][0]
        text = window(r["context"], ans, r["question"])
        if ans and ans not in text:
            continue
        pid = "klue:" + r["guid"]
        out.append({"id": f"sft-{'klue_imp' if imp else 'klue_ans'}-{len(out):05d}", "kind": "klue_imp" if imp else "klue_ans", "question": r["question"],
                    "answer": "찾을 수 없습니다" if imp else ans, "gold": [] if imp else [pid], "type": "D" if imp else "B", "d_none": imp,
                    "in_corpus": True, "hops": 1, "inject": {"passage_id": pid, "doc_id": pid, "title": r["title"], "text": text, "source_type": "web", "date": None}})
        n_imp += imp; n_ans += not imp
        if n_imp + n_ans >= (a.klue if a.klue_ans is None else a.klue_ans + a.klue_imp):
            break
    ty = [r for r in pq.read_table(RAW / "tydiqa/secondary_task/train-00000-of-00001.parquet").to_pylist() if r["id"].startswith("korean")]
    rng.shuffle(ty)
    for r in ty[: a.tydi * 2]:
        if qhash(r["question"]) in evq or sum(o.get("kind") == "tydi" for o in out) >= a.tydi:
            continue
        ans = r["answers"]["text"][0]; text = window(r["context"], ans, r["question"])
        if ans not in text:
            continue
        pid = "tydi:" + r["id"]
        out.append({"id": f"sft-tydi-{len(out):05d}", "kind": "tydi", "question": r["question"], "answer": ans, "gold": [pid], "type": "B", "d_none": False,
                    "in_corpus": True, "hops": 1, "inject": {"passage_id": pid, "doc_id": pid, "title": r["title"], "text": text, "source_type": "wiki", "date": None}})
    for f in a.items_file or []:  # pre-made items (search_v6_items.py kinds, MIRACL human): graded by T.grade
        for l in open(f, encoding="utf-8"):
            it = json.loads(l)
            if it.get("valid", True) and qhash(it["question"]) not in evq:
                out.append({**it, "id": f"sft-{it.get('kind', 'item')}-{len(out):05d}", "kind": it.get("kind", "item")})
    mk = [json.loads(l) for l in gzip.open(RAW / "mkqa/mkqa.jsonl.gz", "rt", encoding="utf-8")]; rng.shuffle(mk)
    n = 0
    for r in mk:
        q = r["queries"].get("ko"); ans = [x for x in r["answers"].get("ko", []) if x.get("text") and x["type"] not in ("unanswerable", "long_answer")]
        if not q or not ans or qhash(q) in evq:
            continue
        out.append({"id": f"sft-mkqa-{len(out):05d}", "kind": "mkqa", "question": q, "answer": ans[0]["text"], "gold": [], "type": "B", "d_none": False,
                    "in_corpus": True, "hops": 1, "mkqa": True})
        n += 1
        if n >= a.mkqa:
            break
    done_q = set()
    for f in a.done or []:
        done_q |= {qhash(r["item"]["question"]) for r in map(json.loads, open(f, encoding="utf-8")) if r["accepted"]}  # rejected/failed rows get another try
    return [o for o in out if qhash(o["question"]) not in done_q]


def grade_mkqa(item, run, texts) -> dict:
    f = run["final"] or {}; ans = str(f.get("answer", ""))
    n_s = sum(s["action"] == "search" for s in run["steps"]); n_w = sum(s["action"] == "web_search" for s in run["steps"])
    cited = " ".join(texts.get(c, "") for c in run["cited"])
    g = {"finished": bool(run["final"]), "correct": T.same(item["question"], item["answer"], ans), "searched": n_s + n_w > 0}
    g["grounded"] = bool(run["cited"]) and g["correct"] and T.same(item["question"], item["answer"], T.read_answer(item["question"], [cited]))
    g["n_search"], g["n_web"] = n_s, n_w
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--klue", type=int, default=1400); ap.add_argument("--tydi", type=int, default=600); ap.add_argument("--mkqa", type=int, default=600)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--out", default="data/sft_traj/traj.jsonl")
    ap.add_argument("--klue-ans", type=int, help="answerable KLUE cap (overrides --klue // 2)")
    ap.add_argument("--klue-imp", type=int, help="impossible KLUE cap (overrides --klue // 2)")
    ap.add_argument("--items-file", nargs="*", help="extra item jsonl files (search_v6_items.py output, MIRACL human items)")
    ap.add_argument("--done", nargs="*", help="earlier traj files: their questions are skipped")
    a = ap.parse_args()
    its = items(a)
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [it for it in its if it["id"] not in done]
    print(json.dumps({"items": dict(Counter(i["kind"] for i in its)), "todo": len(todo)}), flush=True)
    texts = {}
    for name in ("local", "web"):
        for l in open(OLD / f"data/search/corpus_v2/{name}.jsonl", encoding="utf-8"):
            r = json.loads(l); texts[r["passage_id"]] = r["title"] + ": " + r["text"]
    for it in its:
        if it.get("inject"):
            texts[it["inject"]["passage_id"]] = it["inject"]["title"] + ": " + it["inject"]["text"]
    T.retriever("local"); T.retriever("web")
    wl, kept, tried, t0 = threading.Lock(), Counter(), Counter(), time.time()

    def work(it):
        best = None
        for att in range(a.attempts):
            _local.inject = it.get("inject"); _local.used = False
            _local.rank = int(hashlib.md5((it["id"] + str(att)).encode()).hexdigest(), 16) % 3
            extra = T.HINTS[it["hint"]] if it.get("hint") in T.HINTS else T.HINT_NOWHERE if it.get("d_none") else ""
            try:
                run = T.run_agent(it, 0.7 if att == 0 else 0.9, extra=extra)
            except Exception as e:  # e.g. HTTP 400 when a long transcript overflows the server slot: count as a failed attempt
                print(f"[warn] {it['id']} attempt {att}: {type(e).__name__} {str(e)[:120]}", flush=True)
                run = {"messages": [], "steps": [], "final": None, "format_fails": 0, "shown": {}, "cited": [], "harness": []}
            g = grade_mkqa(it, run, texts) if it.get("mkqa") else T.grade(it, run, texts)
            if it.get("d_none"):
                g["hint_not_leaked"] = not any(w in str(st.get("think", "")) for st in run["steps"] for w in ("안내", "힌트", "비공개"))
            ok = all(v for k, v in g.items() if not k.startswith("n_") and k != "gold_cited")
            best = {"id": it["id"], "kind": it["kind"], "type": it["type"], "item": {k: v for k, v in it.items() if k != "inject"}, "attempt": att,
                    "accepted": ok, "grades": g, "messages": run["messages"]}
            if ok:
                break
        _local.inject = None
        with wl:
            tried[it["kind"]] += 1; kept[it["kind"]] += best["accepted"]
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(best, ensure_ascii=False) + "\n")
            if sum(tried.values()) % 50 == 0:
                print(json.dumps({"done": sum(tried.values()), "of": len(todo), "secs": round(time.time() - t0), "kept": dict(kept), "tried": dict(tried)}), flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print(json.dumps({"final": True, "kept": dict(kept), "tried": dict(tried), "secs": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
