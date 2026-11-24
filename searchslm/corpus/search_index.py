#!/usr/bin/env python3
"""SearchSLM retriever: char-bigram BM25 + multilingual-e5-small dense, fused by RRF, reranked by KURE-v2 (ColBERT MaxSim).

One index per layer group: "local" (local + noise passages, the `search` action) and "web" (`web_search`).
Build:   python corpus/search_index.py build --name local --inputs data/search/corpus_v2/local.jsonl
Evaluate on MIRACL dev: python corpus/search_index.py eval
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "data/search/index"
MODELS = ROOT / "data/search/models"
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"  # cuda: the Vast RL box


def toks(s: str) -> list[str]:
    out = []
    for w in re.findall(r"[가-힣]+|[a-zA-Z]+|\d+", s.lower()):
        out += [w] if len(w) < 2 else [w[i:i + 2] for i in range(len(w) - 1)]
    return out


# ---------------- BM25 (CSR postings in numpy) ----------------
class BM25:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.ptr, self.doc, self.tf, self.dl = z["ptr"], z["doc"], z["tf"], z["dl"]
        self.vocab = {t: i for i, t in enumerate(json.loads(path.with_suffix(".vocab.json").read_text()))}
        self.N = len(self.dl); self.avg = self.dl.mean()
        df = np.diff(self.ptr)
        self.idf = np.log(1 + (self.N - df + 0.5) / (df + 0.5)).astype(np.float32)

    @staticmethod
    def build(texts: list[str], path: Path) -> None:
        vocab: dict[str, int] = {}
        t_ids, d_ids, tfs, dl = [], [], [], np.zeros(len(texts), np.int32)
        for d, text in enumerate(texts):
            c = Counter(toks(text)); dl[d] = sum(c.values())
            for t, f in c.items():
                t_ids.append(vocab.setdefault(t, len(vocab))); d_ids.append(d); tfs.append(f)
        t_ids = np.asarray(t_ids, np.int32); order = np.argsort(t_ids, kind="stable")
        ptr = np.zeros(len(vocab) + 1, np.int64); np.add.at(ptr, t_ids + 1, 1); ptr = np.cumsum(ptr)
        np.savez(path, ptr=ptr, doc=np.asarray(d_ids, np.int32)[order], tf=np.asarray(tfs, np.float32)[order], dl=dl)
        path.with_suffix(".vocab.json").write_text(json.dumps(list(vocab), ensure_ascii=False))

    def scores(self, q: str, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        s = np.zeros(self.N, np.float32)
        for t in set(toks(q)):
            i = self.vocab.get(t)
            if i is None:
                continue
            a, e = self.ptr[i], self.ptr[i + 1]
            d, f = self.doc[a:e], self.tf[a:e]
            s[d] += self.idf[i] * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[d] / self.avg))
        return s


# ---------------- dense: multilingual-e5-small ----------------
class E5:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODELS / "e5-small")
        self.model = AutoModel.from_pretrained(MODELS / "e5-small").to(DEV).eval().half()

    @torch.no_grad()
    def encode(self, texts: list[str], prefix: str, bs: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            x = self.tok([prefix + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEV)
            h = self.model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            out.append(v.float().cpu().numpy().astype(np.float16))
        return np.concatenate(out)


# ---------------- reranker: KURE-v2 ColBERT MaxSim ----------------
class Kure:
    def __init__(self):
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
        p = MODELS / "kure-v2"
        self.tok = AutoTokenizer.from_pretrained(p)
        self.model = AutoModel.from_pretrained(p, dtype=torch.float16).to(DEV).eval()
        self.dense = [{k: v.to(DEV, torch.float16) for k, v in load_file(p / f"{d}/model.safetensors").items()} for d in ("1_Dense", "2_Dense", "3_Dense")]
        cfg = json.loads((p / "config_sentence_transformers.json").read_text())
        self.skip = {i for w in cfg["skiplist_words"] for i in self.tok(w, add_special_tokens=False).input_ids if len(self.tok(w, add_special_tokens=False).input_ids) == 1}

    def _project(self, h):
        for w in self.dense:  # pylate Dense: linear(x) (+ residual(x) when present)
            y = h @ w["linear.weight"].T
            h = y + h @ w["residual.weight"].T if "residual.weight" in w else y
        return torch.nn.functional.normalize(h, dim=-1)

    @torch.no_grad()
    def query(self, q: str):
        x = self.tok("[Q] " + q, padding="max_length", truncation=True, max_length=64, return_tensors="pt", return_token_type_ids=False)
        att = x["attention_mask"].clone()
        x["input_ids"][x["attention_mask"] == 0] = self.tok.mask_token_id  # query expansion with [MASK], not attended
        h = self.model(input_ids=x["input_ids"].to(DEV), attention_mask=att.to(DEV)).last_hidden_state
        return self._project(h)[0]

    @torch.no_grad()
    def docs(self, texts: list[str]):
        x = self.tok(["[D] " + t for t in texts], padding=True, truncation=True, max_length=320, return_tensors="pt", return_token_type_ids=False).to(DEV)
        h = self._project(self.model(**x).last_hidden_state)
        keep = x["attention_mask"].bool()
        for s in self.skip:
            keep &= x["input_ids"] != s
        return h, keep

    def rerank(self, q: str, texts: list[str]) -> np.ndarray:
        qv = self.query(q)
        dv, keep = self.docs(texts)
        sim = torch.einsum("qd,ntd->nqt", qv, dv).masked_fill(~keep[:, None, :], -1e4)
        return sim.max(-1).values.sum(-1).float().cpu().numpy()


# ---------------- hybrid retriever ----------------
class Retriever:
    def __init__(self, name: str, rerank: bool = True):
        self.meta = [json.loads(l) for l in open(IDX / f"{name}.meta.jsonl", encoding="utf-8")]
        self.bm = BM25(IDX / f"{name}.bm25.npz")
        self.emb = np.load(IDX / f"{name}.e5.npy")
        self.e5 = E5(); self.emb_t = torch.from_numpy(self.emb).to(DEV)
        self.kure = Kure() if rerank else None

    def search(self, q: str, k: int = 8, pool: int = 50, mode: str = "hybrid+rerank") -> list[dict]:
        if mode == "bm25":
            return [self.meta[i] for i in np.argsort(-self.bm.scores(q))[:k]]
        qv = torch.from_numpy(self.e5.encode([q], "query: ")).to(DEV)
        ds = (self.emb_t @ qv[0]).float().cpu().numpy()
        if mode == "dense":
            return [self.meta[i] for i in np.argsort(-ds)[:k]]
        bs = self.bm.scores(q)
        rrf: dict[int, float] = {}
        top_b = np.argsort(-bs)[:pool]; top_d = np.argsort(-ds)[:pool]
        for ranking in (top_b, top_d):
            for r, i in enumerate(ranking):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1 / (60 + r)
        cand = sorted(rrf, key=lambda i: -rrf[i])[:pool]
        if mode == "hybrid" or self.kure is None:
            return [self.meta[i] for i in cand[:k]]
        sc = self.kure.rerank(q, [self.meta[i]["title"] + " " + self.meta[i]["text"] for i in cand])
        return [self.meta[cand[j]] for j in np.argsort(-sc)[:k]]


VISIBLE = ("passage_id", "title", "text", "source_type", "date", "url")


def build(name: str, inputs: list[str]) -> None:
    IDX.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for f in inputs for l in open(ROOT / f, encoding="utf-8")]
    t0 = time.time()
    with open(IDX / f"{name}.meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:  # keep split/layer/doc_id for evaluation; the agent only ever sees VISIBLE fields
            f.write(json.dumps({k: r.get(k) for k in VISIBLE + ("doc_id", "layer", "split")}, ensure_ascii=False) + "\n")
    texts = [r["title"] + " " + r["text"] for r in rows]
    BM25.build(texts, IDX / f"{name}.bm25.npz"); print(f"[{name}] bm25 {len(rows)} passages {time.time()-t0:.0f}s", flush=True)
    np.save(IDX / f"{name}.e5.npy", E5().encode(texts, "passage: ")); print(f"[{name}] e5 done {time.time()-t0:.0f}s", flush=True)


def evaluate(k: int = 8) -> dict:
    topics = dict(csv.reader(open(ROOT / "data/search/miracl-ko/topics/topics.miracl-v1.0-ko-dev.tsv"), delimiter="\t"))
    rel: dict[str, set] = {}
    for qid, _, pid, lab in csv.reader(open(ROOT / "data/search/miracl-ko/qrels/qrels.miracl-v1.0-ko-dev.tsv"), delimiter="\t"):
        if lab == "1":
            rel.setdefault(qid, set()).add("miracl:" + pid)
    R = Retriever("local")
    res = {}
    for mode in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        t0 = time.time(); rec = ndcg = mrr = 0.0
        for qid, q in topics.items():
            got = [m["passage_id"] for m in R.search(q, k=10, mode=mode)]
            g = rel.get(qid, set())
            rec += len(g & set(got[:k])) / max(1, len(g))
            dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(got) if p in g)
            ndcg += dcg / sum(1 / math.log2(i + 2) for i in range(min(10, len(g)))) if g else 0
            mrr += next((1 / (i + 1) for i, p in enumerate(got) if p in g), 0)
        n = len(topics)
        res[mode] = {f"recall@{k}": round(rec / n, 4), "ndcg@10": round(ndcg / n, 4), "mrr@10": round(mrr / n, 4),
                     "ms_per_query": round((time.time() - t0) * 1000 / n)}
        print(mode, res[mode], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--name", required=True); b.add_argument("--inputs", nargs="+", required=True)
    sub.add_parser("eval")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.name, a.inputs)
    else:
        out = evaluate()
        (ROOT / "outputs/search_dataset/retriever_eval_miracl_dev.json").write_text(json.dumps(out, indent=1))
