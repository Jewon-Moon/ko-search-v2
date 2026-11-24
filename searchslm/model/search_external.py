#!/usr/bin/env python3
"""External checks with official, judge-free metrics (data in data/external/, downloaded 2026-11-13, never tuned on).

  korquad    KorQuAD 1.0 dev, fixed sample (one question per paragraph, seed 0), scored by the official
             evaluate-v1.0.py (EM / char-F1). Settings:
               reader  the question's own paragraph is the only search result (split into <=550-char chunks because
                       the display cuts passages at 600 chars); our agent vs 0.6B RAG vs 4B RAG read the same chunks.
               open    the full system searches its own index as in production (Korean Wikipedia is in it, so the
                       paragraph may or may not be found).
  klue       KLUE-MRC dev, 100 per question type (paraphrase / multi-sentence reasoning / unanswerable), reader setting:
             gave-up rate per type (should be high only on unanswerable) and "contains" on the answerable types.
  sqa        Ko-StrategyQA dev yes/no multi-hop QA (answers from NomaDamas/Ko-StrategyQA), reader (every annotator's
             evidence paragraphs) and open (own retrieval over its corpus); accuracy, unparsed counts as wrong.
  strategyqa Ko-StrategyQA dev sub-questions -> passage retrieval over its 9,251-passage corpus, our retriever
             (BM25 / e5 / hybrid / +KURE rerank), Recall@8, nDCG@10, MRR@10 against its qrels.

  python model/search_external.py strategyqa
  python model/search_external.py korquad --n 300 --systems "agent:student=...@curated2+h#nfweb.mhskip.xlate.noans" rag:stage2 rag:e4b
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "corpus"))
import search_baseline as SB  # noqa: E402

T = SB.T
import os
OUT = Path(os.environ["EXT_OUT"]) if os.environ.get("EXT_OUT") else ROOT / "outputs/external"  # EXT_OUT: e.g. ko-search-v2/runs/eval/external


def chunks(text: str, size: int = 550) -> list[str]:
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?。])\s+", text):
        if cur and len(cur) + len(s) + 1 > size:
            out.append(cur); cur = ""
        cur = (cur + " " + s).strip()
        while len(cur) > size:  # a single very long sentence
            out.append(cur[:size]); cur = cur[size:]
    return out + ([cur] if cur else [])


def korquad_sample(n: int) -> tuple[list[dict], dict]:
    d = json.load(open(EXT / "korquad/KorQuAD_v1.0_dev.json", encoding="utf-8"))
    rng = random.Random(0)
    paras = [(a["title"], p) for a in d["data"] for p in a["paragraphs"] if p["qas"]]
    pick = rng.sample(paras, n)
    items, sub = [], {}
    for title, p in pick:
        qa = rng.choice(p["qas"])
        hits = [{"passage_id": f"korquad:{qa['id']}#{i}", "title": title, "text": c, "source_type": "wiki", "date": None}
                for i, c in enumerate(chunks(p["context"]))][:8]
        items.append({"id": qa["id"], "question": qa["question"], "answers": [a["text"] for a in qa["answers"]], "hits": hits})
        sub.setdefault(title, []).append({"context": p["context"], "qas": [qa]})
    return items, {"version": d["version"], "data": [{"title": t, "paragraphs": ps} for t, ps in sub.items()]}


def run_korquad(a) -> None:
    items, subset = korquad_sample(a.n)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "korquad_subset.json").write_text(json.dumps(subset, ensure_ascii=False))
    res = json.loads((OUT / "korquad.json").read_text()) if (OUT / "korquad.json").exists() else {}
    for name in a.systems:
        for setting in (["reader", "open"] if name.startswith("agent") else ["reader"]):
            key = f"{name} | {setting}"
            t0 = time.time()
            preds = answers(name, setting, items)
            pf = OUT / f"korquad_preds_{re.sub(r'[^A-Za-z0-9._+@-]+', '_', key)}.json"
            pf.write_text(json.dumps(preds, ensure_ascii=False))
            # the official scorer, unmodified, as a separate isolated process
            sc = json.loads(subprocess.run([sys.executable, "-I", str(EXT / "korquad/evaluate-v1.0.py"), str(OUT / "korquad_subset.json"), str(pf)],
                                           capture_output=True, text=True, check=True).stdout)
            # sentence answers get EM 0 by construction, so also report "contains" (normalized gold inside the answer, the
            # usual match metric for generative QA), with the official normalizer
            norm = official().normalize_answer
            hit = sum(any(norm(g) in norm(preds[it["id"]]) for g in it["answers"]) for it in items)
            res[key] = {**{k: round(v, 1) for k, v in sc.items()}, "contains": round(100 * hit / len(items), 1), "n": len(items), "secs": round(time.time() - t0)}
            print(key, res[key], flush=True)
            (OUT / "korquad.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def official():
    """The downloaded KorQuAD scorer as a module (read before use: functions only, main is guarded)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("korquad_eval", EXT / "korquad/evaluate-v1.0.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def answers(name: str, setting: str, items: list[dict], index: str | None = None) -> dict[str, str]:
    """reader: each item's own hits are the only search result. open: production search, or another index if given."""
    if not name.startswith("agent"):
        call = SB.stage2() if name == "rag:stage2" else SB.lmstudio({"rag:e4b": "baseline-e4b"}[name])
        return {it["id"]: call(it["question"], it["hits"] if setting == "reader" else T.search(index, it["question"])).strip() for it in items}
    spec, _, flags = name.partition("#")
    T.H2 = set(flags.split(".")) - {""}
    h = 1 if spec.endswith("+h") else 0
    base = spec.removesuffix("+h")
    llm = None if base == "agent:teacher" else SB.student(base.split("=", 1)[1])  # agent:teacher = the LM Studio teacher runs the same loop
    real_search, out = T.search, {}
    for it in items:
        if setting == "reader":  # every search returns the question's own paragraph
            T.search = lambda name_, q, k=T.K, exclude=None, _h=it["hits"]: list(_h)
        elif index:
            T.search = lambda name_, q, k=T.K, exclude=None: real_search(index, q, k, exclude)
        try:
            r = T.run_agent({"question": it["question"], "type": "B", "gold": []}, 0.0, llm=llm, harness=h)
        finally:
            T.search = real_search
        out[it["id"]] = str((r["final"] or {}).get("answer", "")).strip()
    return out


NF = ("찾을 수 없", "확인할 수 없", "알 수 없", "나와 있지 않", "언급되어 있지 않", "언급되지 않", "포함되어 있지 않", "제공되지 않")
YN = "(예 또는 아니오로 답하세요.)"


def gave_up(ans: str) -> bool:
    return not ans.strip() or any(k in ans for k in NF)


def yes_no(ans: str) -> bool | None:
    """Leading 예/네 -> True, 아니(요/오) -> False; anything else (incl. giving up) is unparsed and scored wrong."""
    a = re.sub(r"^[\s'\"“‘(\[]+", "", ans)
    if re.match(r"(예|네|맞습니다|그렇습니다)(?![가-힣])|(예|네)[,.!]", a):
        return True
    if re.match(r"아니(요|오|에요|ㅂ니다|다)?(?![가-힣])|아닙니다|아니요|아니오", a):
        return False
    return None


def klue_sample(n_per: int) -> list[dict]:
    import pandas as pd
    d = pd.read_parquet(EXT / "klue_mrc/validation.parquet")
    rng, items = random.Random(0), []
    for qt in ("1", "2", "3"):  # 1 paraphrase, 2 multi-sentence reasoning, 3 unanswerable (KLUE paper)
        rows = [r for r in d[d.question_type.astype(str) == qt].itertuples() if len(chunks(r.context)) <= 8]
        for r in rng.sample(rows, n_per):
            hits = [{"passage_id": f"klue:{r.guid}#{i}", "title": r.title, "text": c, "source_type": "news", "date": None}
                    for i, c in enumerate(chunks(r.context))]
            items.append({"id": r.guid, "question": r.question, "qtype": qt, "answers": list(r.answers["text"]), "hits": hits})
    return items


def run_klue(a) -> None:
    items = klue_sample(a.n // 3)
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "klue_mrc.json").read_text()) if (OUT / "klue_mrc.json").exists() else {}
    norm = official().normalize_answer
    for name in a.systems:
        t0 = time.time()
        preds = answers(name, "reader", items)
        (OUT / f"klue_preds_{re.sub(r'[^A-Za-z0-9._+@-]+', '_', name)}.json").write_text(json.dumps(preds, ensure_ascii=False))
        r = {}
        for qt, lab in (("1", "paraphrase"), ("2", "multi_sentence"), ("3", "unanswerable")):
            s = [it for it in items if it["qtype"] == qt]
            r[lab] = {"gave_up": round(100 * sum(gave_up(preds[it["id"]]) for it in s) / len(s), 1), "n": len(s)}
            if qt != "3":
                r[lab]["contains"] = round(100 * sum(any(norm(g) in norm(preds[it["id"]]) for g in it["answers"]) for it in s) / len(s), 1)
        r["secs"] = round(time.time() - t0)
        res[name] = r
        print(name, r, flush=True)
        (OUT / "klue_mrc.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def sqa_sample(n: int) -> list[dict]:
    d = json.load(open(EXT / "ko_strategyqa_qa/ko-strategy-qa_dev.json", encoding="utf-8"))
    corpus = {r["_id"]: r for r in map(json.loads, open(EXT / "ko_strategyqa/corpus.jsonl", encoding="utf-8"))}
    def ids(x):
        return [y for z in x for y in ids(z)] if isinstance(x, list) else [x] if x in corpus else []
    items = []
    for qid, v in d.items():
        ev = list(dict.fromkeys(ids(v["evidence"])))[:8]  # every annotator's evidence paragraphs
        if ev:
            items.append({"id": qid, "question": f"{v['question']} {YN}", "gold": v["answer"] == "True",
                          "hits": [{"passage_id": p, "title": corpus[p]["title"], "text": corpus[p]["text"][:600], "source_type": "wiki", "date": None} for p in ev]})
    return random.Random(0).sample(items, min(n, len(items)))


def run_sqa(a) -> None:
    items = sqa_sample(a.n)
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "strategyqa_qa.json").read_text()) if (OUT / "strategyqa_qa.json").exists() else {}
    for name in a.systems:
        for setting in ("reader", "open"):
            key, t0 = f"{name} | {setting}", time.time()
            preds = answers(name, setting, items, index="kostrategyqa")
            (OUT / f"sqa_preds_{re.sub(r'[^A-Za-z0-9._+@-]+', '_', key)}.json").write_text(json.dumps(preds, ensure_ascii=False))
            yn = {it["id"]: yes_no(preds[it["id"]]) for it in items}
            res[key] = {"acc": round(100 * sum(yn[it["id"]] == it["gold"] for it in items) / len(items), 1),
                        "unparsed": round(100 * sum(v is None for v in yn.values()) / len(items), 1),
                        "gave_up": round(100 * sum(gave_up(preds[it["id"]]) for it in items) / len(items), 1),
                        "gold_yes": round(100 * sum(it["gold"] for it in items) / len(items), 1), "n": len(items), "secs": round(time.time() - t0)}
            print(key, res[key], flush=True)
            (OUT / "strategyqa_qa.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def run_strategyqa(a) -> None:
    from search_index import IDX, Retriever, build
    d = EXT / "ko_strategyqa"
    if not (IDX / "kostrategyqa.meta.jsonl").exists():
        rows = [json.loads(l) for l in open(d / "corpus.jsonl", encoding="utf-8")]
        with open(d / "passages.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"passage_id": r["_id"], "title": r["title"], "text": r["text"], "source_type": "wiki", "doc_id": r["title"]}, ensure_ascii=False) + "\n")
        build("kostrategyqa", [str((d / "passages.jsonl").relative_to(ROOT))])
    rel: dict[str, set] = {}
    for l in open(d / "qrels/dev.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["score"] > 0:
            rel.setdefault(r["query-id"], set()).add(r["corpus-id"])
    qs = {r["_id"]: r["text"] for r in map(json.loads, open(d / "queries.jsonl", encoding="utf-8")) if r["_id"] in rel}
    R = Retriever("kostrategyqa")
    res = {"n_queries": len(qs)}
    for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        t0 = time.time(); rec = ndcg = mrr = 0.0
        for qid, q in qs.items():
            got = [m["passage_id"] for m in R.search(q, k=10, mode=mode)]
            g = rel[qid]
            rec += len(g & set(got[:8])) / len(g)
            ndcg += sum(1 / math.log2(i + 2) for i, p in enumerate(got) if p in g) / sum(1 / math.log2(i + 2) for i in range(min(10, len(g))))
            mrr += next((1 / (i + 1) for i, p in enumerate(got) if p in g), 0)
        n = len(qs)
        res[mode] = {"recall@8": round(rec / n, 3), "ndcg@10": round(ndcg / n, 3), "mrr@10": round(mrr / n, 3), "secs": round(time.time() - t0)}
        print(mode, res[mode], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "strategyqa_retrieval.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=("korquad", "strategyqa", "klue", "sqa"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--systems", nargs="+", default=["rag:stage2"])
    a = ap.parse_args()
    {"korquad": run_korquad, "strategyqa": run_strategyqa, "klue": run_klue, "sqa": run_sqa}[a.task](a)


if __name__ == "__main__":
    assert chunks("가. " * 400)[0].endswith("가.") and all(len(c) <= 550 for c in chunks("a" * 2000))
    assert yes_no("예, 필요합니다.") is True and yes_no("아니요. 없습니다.") is False and yes_no("아닙니다.") is False
    assert yes_no("네") is True and yes_no("예술가는 ...") is None and yes_no("찾을 수 없습니다.") is None and yes_no("아니타는") is None
    assert gave_up("찾을 수 없습니다.") and gave_up("") and not gave_up("서울입니다.")
    main()
