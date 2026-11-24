#!/usr/bin/env python3
"""SearchSLM baselines on the dev item set (never the sealed test).

- rag:<model>  classic one-shot RAG: hybrid+rerank top-5 local (+noise) and top-3 web, one answer call.
    models: stage2 (our 0.6B base, few-shot completion via transformers), e4b (Gemma 4 E4B, LM Studio)
- agent:teacher  Gemma 4 26B running the SearchSLM action loop (reference upper bound).
Correctness is judged by the teacher (same() in search_teacher), so absolute numbers are lenient; compare rows.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRED_DIR: Path | None = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corpus"))
import search_teacher as T  # noqa: E402

CLOSED_SYS = "검색 없이 알고 있는 지식만으로 질문에 짧게 답하세요. 확실히 모르면 \"모릅니다\"라고 답하세요."
RAG_SYS = "검색 결과 문단만 근거로 질문에 1~2문장으로 답하세요. 근거가 없으면 \"찾을 수 없습니다\"라고 답하세요. 검색 없이 답할 수 있는 계산·번역·인사는 그냥 답하세요."


def retrieve(q: str) -> list[dict]:
    if T.LIVE_ALL:  # live eval: one real web search, no fixed index
        return T.search("web", q, k=8)
    return T.search("local", q, k=5) + T.search("web", q, k=3)


def rag_sys() -> str:
    if T.LIVE_ALL:  # live results need today's date to answer "올해 ..." questions
        import search_live
        return f"오늘은 {search_live.today()}입니다. " + RAG_SYS
    return RAG_SYS


def ctx(hits) -> str:
    return "\n".join(T.fmt(i + 1, h) for i, h in enumerate(hits))


def lmstudio(model: str, closed: bool = False):
    def call(q, hits):
        msgs = ([{"role": "system", "content": CLOSED_SYS}, {"role": "user", "content": q}] if closed else  # closed book: the paper's setting
                [{"role": "system", "content": rag_sys()}, {"role": "user", "content": f"[검색 결과]\n{ctx(hits)}\n\n질문: {q}"}])
        body = json.dumps({"model": model, "messages": msgs, "max_tokens": 1500, "temperature": 0, **T.REASONING.get(model, {})}).encode()  # E4B thinks before answering
        r = json.load(urllib.request.urlopen(urllib.request.Request(T.API, body, {"Content-Type": "application/json"}), timeout=600))
        return r["choices"][0]["message"]["content"] or ""
    return call


def stage2():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = ROOT / "outputs/release/qwen3-0.6b-ko32k-stage2-signal2678-base"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("mps").eval()
    shot = ("문단:\n[1] (wiki · 2019-02-01) 지미 카터: 제임스 얼 카터 주니어는 민주당 출신 미국 39번째 대통령이다.\n"
            "질문: 지미 카터는 미국 몇 번째 대통령이었나?\n답: 지미 카터는 미국의 39번째 대통령입니다.\n\n"
            "문단:\n[1] (wiki · 2019-02-01) 로마: 로마는 이탈리아의 수도이다.\n질문: 한국 프로야구 올해 우승팀은?\n답: 찾을 수 없습니다.\n\n")

    @torch.no_grad()
    def call(q, hits):
        p = f"{RAG_SYS}\n\n{shot}문단:\n{ctx(hits)}\n질문: {q}\n답:"
        ids = tok(p, return_tensors="pt").input_ids[:, -3500:].to("mps")
        out = model.generate(ids, max_new_tokens=60, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
    return call


def student(spec: str):
    """agent:student=<adapter dir>@<curated|raw>: the LoRA-tuned 0.6B runs T.run_agent in place of the teacher.
    The system prompt is the one the adapter was trained with (first row of its training data)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(ROOT / "model"))
    import manager_lora
    adapter, display = spec.split("@")
    if adapter.startswith("lms:"):  # a GGUF loaded in LM Studio under this identifier (deployment quant check), same raw ChatML prompt
        T.AGENT = (ROOT / "prompts/agent_system.txt").read_text(encoding="utf-8")  # system prompt of sft_v4 training rows
        T.CURATE = display in ("curated", "curated2")
        T.CURATE_MODE = "content" if display == "curated2" else "title"
        url =T.API.replace("/chat/completions", "/completions")

        def lms_chat(messages, max_tokens=400, temperature=0.0):  # greedy like the HF path
            p = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
            body = json.dumps({"model": adapter[4:], "prompt": p, "max_tokens": max_tokens, "temperature": 0.0, "stop": ["<|im_end|>"]}).encode()
            return json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=900))["choices"][0]["text"]
        return lms_chat
    if adapter.startswith("hf:"):  # ko-search-v2: a full HF checkpoint (needs a transformers that knows its arch), v4 raw agent prompt
        tok = AutoTokenizer.from_pretrained(adapter[3:])
        model = AutoModelForCausalLM.from_pretrained(adapter[3:], dtype=torch.bfloat16).to("mps").eval()
        T.AGENT = (ROOT / "prompts/agent_system.txt").read_text(encoding="utf-8")  # system prompt of sft_v4 training rows
    else:
        base = ROOT / "outputs/release/qwen3-0.6b-ko32k-stage2-signal2678-base"
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to("mps")
        manager_lora.load_adapter(model, ROOT / adapter)
        cfg = json.loads((ROOT / adapter).parent.joinpath("training_config.json").read_text())
        T.AGENT = json.loads(open(ROOT / cfg["data"] / "train.jsonl", encoding="utf-8").readline())["messages"][0]["content"]
    T.CURATE = display in ("curated", "curated2")
    T.CURATE_MODE = "content" if display == "curated2" else "title"  # curated2: same title OR question subject (v2)
    end = tok.convert_tokens_to_ids("<|im_end|>")

    @torch.no_grad()
    def chat(messages, max_tokens=400, temperature=0.0):  # greedy; same plain ChatML rendering as search_sft_train
        p = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to("mps")
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False, eos_token_id=end)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return chat


def score(it, ans, ids, n_search=1):
    nf = any(k in ans for k in ("찾을 수 없", "확인할 수 없", "알 수 없", "couldn't find", "could not find", "can't find"))
    ok = nf if it.get("d_none") else T.same(it["question"], it["answer"], ans)
    gold = set(it["gold"][-1:]) if it["type"] == "C" else set(it["gold"])
    return {"type": it["type"], "correct": ok, "gold_in_context": bool(gold & set(ids)) if gold else None, "n_search": n_search,
            "id": it["id"], "question": it["question"], "gold": it["answer"], "pred": ans, "d_none": bool(it.get("d_none"))}


def run(name: str, items: list[dict], workers: int) -> dict:
    t0 = time.time()
    fired = defaultdict(int)
    if name.startswith("agent"):
        spec, _, flags = name.partition("#")  # "#a.b": Harness² candidate rules switched on for this run
        T.H2 = set(flags.split(".")) - {""}
        h = 2 if spec.endswith("+h2") else 1 if spec.endswith("+h") else 0  # runtime harness level (search_teacher.run_agent)
        base = spec.removesuffix("+h2").removesuffix("+h")
        if base.startswith("agent:student="):
            llm = student(base.split("=", 1)[1])
        elif base.startswith("agent:lm="):  # another LM Studio model running the same agent loop (teacher candidates)
            mid = base.split("=", 1)[1]
            llm = lambda msgs, mt=400, temp=0.0: T.chat(msgs, mt, temp, model=mid)
        else:
            llm = None

        def one(it):
            r = T.run_agent(it, 0.0, llm=llm, harness=h)
            for f in r["harness"]:
                fired[f] += 1
            ids = set(r["shown"])
            n = sum(s["action"] != "answer" for s in r["steps"])
            row = score(it, str((r["final"] or {}).get("answer", "")), ids, n)
            row["trace"] = [{k: st[k] for k in ("action", "query", "results", "answer") if k in st} for st in r["steps"]]
            row["fired"] = r["harness"]
            if "question_ko" in r:  # translator run: keep the Korean leg for inspection
                row["question_ko"] = r["question_ko"]
            return row
    else:
        closed = name.startswith("closed:")  # closed:<LM Studio id>: no search at all
        call = stage2() if name == "rag:stage2" else lmstudio(name.split(":", 1)[1] if closed else {"rag:e4b": "baseline-e4b", "rag:teacher": T.MODEL}[name], closed)

        def one(it):
            if closed:
                return score(it, re.sub(r"<think>.*?</think>", "", call(it["question"], []), flags=re.S).strip(), [], 0)
            hits = retrieve(it["question"])
            ex = set(it["gold"]) | ({"doc:" + g.split("#")[0] for g in it["gold"]} if it.get("exclude_doc") else set()) if it.get("d_none") else set()
            hits = [h for h in hits if h["passage_id"] not in ex and "doc:" + str(h.get("doc_id")) not in ex]
            return score(it, call(it["question"], hits), [h["passage_id"] for h in hits])
    w = 1 if name == "rag:stage2" or name.startswith("agent:student") else workers  # the local torch model is not thread-safe
    with ThreadPoolExecutor(w) as ex:
        rows = list(ex.map(one, items))
    if PRED_DIR:  # per-item rows for re-judging with another model
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        with open(PRED_DIR / (re.sub(r"[^A-Za-z0-9._+@-]+", "_", name.replace("agent:student=outputs/vast_runs/", "")) + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r); by["all"].append(r)
    return {t: {"n": len(v), "accuracy": round(sum(r["correct"] for r in v) / len(v), 3),
                "gold_in_context": round(sum(bool(r["gold_in_context"]) for r in v if r["gold_in_context"] is not None) / max(1, sum(r["gold_in_context"] is not None for r in v)), 3),
                "avg_search": round(sum(r["n_search"] for r in v) / len(v), 2)} for t, v in sorted(by.items())} | {"secs": round(time.time() - t0), "harness": dict(fired)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="data/search/gen/items_dev.jsonl")
    ap.add_argument("--systems", nargs="+", default=["rag:stage2", "rag:e4b", "agent:teacher"])
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="outputs/search_dataset/baselines_dev.json")
    ap.add_argument("--preds", help="directory for per-item prediction rows")
    ap.add_argument("--set", choices=("scored", "tune"), default="scored")
    ap.add_argument("--title-map", help="json passage_id -> displayed title (timeline records saved under differing titles)")
    ap.add_argument("--extra-items", nargs="*", default=[], help="tune set only: more item files (e.g. fresh D items)")
    ap.add_argument("--extra-n", type=int, default=100)
    ap.add_argument("--live", action="store_true", help="every search is a real web search (Serper); items run as given, no per-type pick")
    a = ap.parse_args()
    if a.preds:
        PRED_DIR = ROOT / a.preds
    items = [json.loads(l) for l in open(ROOT / a.items, encoding="utf-8")]
    items = [it for it in items if it["valid"]]
    import random
    rng = random.Random(9)
    for it in items:  # sealed items arrive with d_none fixed; dev items draw it here (unchanged stream: no key -> same rng calls)
        if it["type"] == "D" and "d_none" not in it and rng.random() < 0.33:
            it["d_none"] = True
    pick = []
    for t in "ABCDE":
        pick += [it for it in items if it["type"] == t][: a.per_type]
    if a.live:
        T.LIVE_ALL, pick = True, items
    if a.set == "tune":  # harness tuning uses the dev items the scored set leaves out, plus extra items (never heldout/sealed)
        scored = {it["id"] for it in pick}
        pick = [it for it in items if it["id"] not in scored]
        for f in a.extra_items:
            extra = [json.loads(l) for l in open(ROOT / f, encoding="utf-8")]
            extra = [it for it in extra if it["valid"]]
            for it in extra:
                if it["type"] == "D" and rng.random() < 0.33:
                    it["d_none"] = True
            pick += extra[: a.extra_n]
    if a.title_map:
        T.TITLE_MAP = json.load(open(ROOT / a.title_map, encoding="utf-8"))
    if not a.live:
        T.retriever("local"); T.retriever("web")
    path = ROOT / a.out
    out = json.loads(path.read_text()) if path.exists() else {}
    out |= {"items": len(pick), "per_type": a.per_type}
    for s in a.systems:
        out[s] = run(s, pick, a.workers)
        print(s, json.dumps(out[s], ensure_ascii=False), flush=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
