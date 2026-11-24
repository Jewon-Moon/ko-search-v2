#!/usr/bin/env python3
"""Online RL (GRPO) for ko-search-v2 V2 (full HF checkpoint, Qwen3.5-2B + tokenizer v2), forked from search_rl.py (0.6B LoRA).
V2 changes: full fine-tuning with fp32 master weights (bf16 autocast), raw result display (the eval's @raw), items from
ko-search-v2/data/rl_v2, a penalty for wrong answers (--wrong) so "not found" beats a confident wrong answer (SimpleQA),
and --passk to measure pass@G without training.


Per step: Q train questions x G sampled rollouts of the agent loop (search / web_search / answer, the same loop and
result display as search_teacher.run_agent, harness off). Each rollout gets a rule reward; advantages are the reward
minus the group mean over the group std; the loss is the policy-gradient term on the student's own sampled tokens plus
a KL penalty to the frozen starting adapter. Groups whose rollouts all score the same carry no signal and are skipped.

Reward (no LLM judge on the training box): 1 if the gold answer string is in the answer or >= 60% of its words are
(agreement with the Gemma judge on 3,608 saved predictions: 91.7%); "nowhere" D items need a not-found reply.
0 for answers over 200 characters (no candidate-listing), -0.2 for a rollout that never answers, -0.05 for using all
four searches.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corpus"))
sys.path.insert(0, str(ROOT / "model"))
import search_teacher as T  # noqa: E402

NF = ("찾을 수 없", "확인할 수 없", "알 수 없")
NORM = lambda s: re.sub(r"[\s\"'“”‘’.,·()\[\]「」『』<>《》:;!?~\-]", "", str(s)).lower()
MAX_CTX = 3800
WRONG = 0.0  # reward for a wrong (non-empty, not not-found) answer on an answerable item; set by --wrong
K2 = ROOT.parent  # repository root


def reward(item: dict, answer: str | None, n_search: int) -> float:
    if answer is None:
        return -0.2
    r = 0.0
    if len(answer) <= 200:
        if item.get("d_none"):
            r = float(any(k in answer for k in NF))
        else:
            g, p = NORM(item["answer"]), NORM(answer)
            words = [NORM(w) for w in re.findall(r"[가-힣A-Za-z0-9]+", str(item["answer"]))]
            words = [w for w in words if len(w) >= 2]
            hit = bool(g) and g in p or bool(words) and sum(w in p for w in words) / len(words) >= 0.6
            nf = any(k in answer for k in NF)
            r = 1.0 if hit and not nf else 0.0 if nf else WRONG
    return r - (0.05 if n_search >= T.MAX_SEARCH else 0.0)


def selftest() -> None:
    global WRONG
    keep, WRONG = WRONG, -0.5
    it = {"answer": "10만원"}
    assert reward(it, "현재 회비는 10만원입니다.", 1) == 1.0
    assert reward(it, "현재 회비는 7만원입니다.", 1) == WRONG
    assert reward({"answer": "알프레트 랑데"}, "랑데 지 인자는 알프레트 랑데에서 왔다.", 1) == 1.0
    assert reward({"answer": "비디 아이(Beady Eye)"}, "리암은 비디 아라는 밴드를 결성했다.", 1) == -0.5  # misspelled name
    assert reward(it, "10만원 " + "또는 다른 값 " * 40, 1) == 0.0  # > 200 chars
    assert reward({"answer": "x", "d_none": True}, "찾을 수 없습니다.", 2) == 1.0
    assert reward({"answer": "x", "d_none": True}, "아마 3종입니다.", 2) == 0.0
    assert reward(it, "찾을 수 없습니다. 10만원일 수도", 1) == 0.0  # not-found wins
    assert reward(it, None, 4) == -0.2
    assert abs(reward(it, "10만원", 4) - 0.95) < 1e-9
    WRONG = keep
    print("reward selftest ok")


class Policy:
    def __init__(self, path: Path, device: str, train: bool):
        """Full checkpoint: trainable copy in fp32 (bf16 updates at lr 1e-6 round to zero), frozen reference in bf16."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32 if train else torch.bfloat16).to(device)
        self.config = {"start": str(path)}
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(train)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.end
        self.head = self.tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self.nl = self.tok("\n", add_special_tokens=False).input_ids

    def ids(self, role: str, content: str) -> list[int]:
        return self.tok(f"<|im_start|>{role}\n{content}<|im_end|>\n", add_special_tokens=False).input_ids

    @torch.no_grad()
    def generate(self, contexts: list[list[int]], temperature: float, max_new: int) -> list[list[int]]:
        """Left-padded batch generation; returns the sampled ids up to and including <|im_end|>."""
        self.model.eval()
        width = max(len(c) for c in contexts)
        inp = torch.full((len(contexts), width), self.pad, dtype=torch.long)
        att = torch.zeros(len(contexts), width, dtype=torch.long)
        for i, c in enumerate(contexts):
            inp[i, width - len(c):] = torch.tensor(c); att[i, width - len(c):] = 1
        kw = dict(do_sample=temperature > 0, max_new_tokens=max_new, eos_token_id=self.end, pad_token_id=self.pad)
        if temperature > 0:
            kw.update(temperature=temperature, top_p=1.0, top_k=0)
        with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16):
            out = self.model.generate(input_ids=inp.to(self.device), attention_mask=att.to(self.device), **kw)
        res = []
        for row in out[:, width:].tolist():
            cut = row.index(self.end) + 1 if self.end in row else len(row)
            res.append(row[:cut])
        return res

    def token_logprobs(self, ids: list[int], labels: list[bool]) -> torch.Tensor:
        """Log-probabilities of the trainable (student) tokens of one trajectory."""
        x = torch.tensor([ids], device=self.device)
        m = torch.tensor(labels[1:], device=self.device)
        with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16):
            hidden = self.model.model(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
            logits = self.model.lm_head(hidden[m]).float()
        tgt = x[0, 1:][m]
        return torch.log_softmax(logits, -1).gather(1, tgt[:, None])[:, 0]


def rollout(pol: Policy, item: dict, g: int, temperature: float, max_new: int) -> list[dict]:
    return rollout_many(pol, [item], g, temperature, max_new)[0]


def rollout_many(pol: Policy, items: list[dict], g: int, temperature: float, max_new: int, chunk: int = 32) -> list[list[dict]]:
    """G trajectories per question for several questions, all generated together per turn (in chunks of `chunk`
    sequences): one question at a time left the GPU 58% idle (trial: 140 of 150 s per step were rollouts)."""
    trajs = []
    for qi, item in enumerate(items):
        start = pol.ids("system", T.AGENT) + pol.ids("user", "질문: " + item["question"])
        exclude = set(item["gold"]) if item.get("in_corpus") is False and item.get("d_none") else set()
        if exclude:
            exclude |= {"doc:" + x.split("#")[0] for x in item["gold"]}
        trajs += [{"q": qi, "item": item, "exclude": exclude, "ids": list(start), "lab": [False] * len(start), "seen": {},
                   "shown": {}, "n_search": 0, "answer": None, "done": False} for _ in range(g)]
    for turn in range(T.MAX_SEARCH + 1):
        act = [t for t in trajs if not t["done"]]
        if not act:
            break
        if turn == T.MAX_SEARCH:
            for t in act:
                m = pol.ids("user", "검색 횟수를 모두 썼습니다. 이제 반드시 answer 행동을 출력하세요.")
                t["ids"] += m; t["lab"] += [False] * len(m)
        act.sort(key=lambda t: len(t["ids"]))  # similar lengths per chunk: less padding
        gens, c, size = [], 0, chunk
        while c < len(act):
            try:
                gens += pol.generate([t["ids"] + pol.head for t in act[c:c + size]], temperature, max_new); c += size
            except torch.cuda.OutOfMemoryError:  # long late-turn contexts: halve the chunk and retry
                torch.cuda.empty_cache(); size = max(1, size // 2)
        for t, gen in zip(act, gens):
            t["ids"] += pol.head + gen + pol.nl
            t["lab"] += [False] * len(pol.head) + [True] * len(gen) + [False] * len(pol.nl)
            j = T.last_json(pol.tok.decode(gen, skip_special_tokens=True))
            if not j or j.get("action") not in ("search", "web_search", "answer") or turn == T.MAX_SEARCH and j.get("action") != "answer":
                t["done"] = True; continue
            if j["action"] == "answer":
                t["answer"] = str(j.get("answer", "")); t["done"] = True; continue
            item = t["item"]
            hits = T.search("local" if j["action"] == "search" else "web", str(j.get("query", "")), exclude=t["exclude"])
            t["n_search"] += 1
            for h in hits:
                t["shown"].setdefault(h["passage_id"], len(t["shown"]) + 1); t["seen"][h["passage_id"]] = h
            marks = T.curate_marks(hits, t["seen"], t["shown"], question=item["question"]) if T.CURATE else {}
            body = "\n".join(T.fmt(t["shown"][h["passage_id"]], h, marks.get(h["passage_id"], "")) for h in hits) or "결과 없음"
            m = pol.ids("user", "[검색 결과]\n" + body)
            t["ids"] += m; t["lab"] += [False] * len(m)
            if len(t["ids"]) > MAX_CTX:
                t["done"] = True
    for t in trajs:
        t["reward"] = reward(t["item"], t["answer"], t["n_search"])
    return [[t for t in trajs if t["q"] == qi] for qi in range(len(items))]


def load_items(n_val: int, seed: int) -> tuple[list[dict], list[dict]]:
    """ko-search-v2/data/rl_v2 (scripts/build_rl_items.py); n_val is ignored: val.jsonl is fixed."""
    rd = lambda n: [json.loads(l) for l in open(K2 / f"data/rl_v2/{n}.jsonl", encoding="utf-8")]
    items, val = rd("items"), rd("val")
    rng = random.Random(seed)
    for it in items + val:  # a third of D items are "nowhere" items with their source document hidden
        if it["type"] == "D" and rng.random() < 1 / 3:
            it["d_none"] = True; it["exclude_doc"] = True
        it.setdefault("gold", [])
    return items, val


@torch.no_grad()
def validate(pol: Policy, val: list[dict], max_new: int) -> dict:
    by = defaultdict(list)
    for it, tr in zip(val, rollout_many(pol, val, 1, 0.0, max_new)):
        r = tr[0]["reward"]
        by[it["type"]].append(r); by["all"].append(r)
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(K2 / "runs/sft_v2/final"))
    ap.add_argument("--out", default=str(K2 / "runs/rl_v2"))
    ap.add_argument("--wrong", type=float, default=-0.5, help="reward for a wrong answer (not-found = 0, correct = 1)")
    ap.add_argument("--passk", type=int, default=0, help="only measure pass@G on this many train items, no training")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--q", type=int, default=8, help="questions per step")
    ap.add_argument("--g", type=int, default=8, help="rollouts per question")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=350)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    T.AGENT = (ROOT / "prompts/agent_system.txt").read_text(encoding="utf-8")  # same prompt as the eval's student()
    global WRONG
    WRONG = a.wrong
    T.CURATE = False  # V2 is evaluated with the raw display (@raw)
    T.retriever("local"); T.retriever("web")
    train, val = load_items(a.val_n, a.seed)
    if a.passk:
        pol = Policy(Path(a.model), device, train=False)
        items = random.Random(a.seed).sample(train, a.passk)
        groups = rollout_many(pol, items, a.g, a.temperature, a.max_new)
        hit = [max(t["reward"] for t in g) >= 0.95 for g in groups]
        mixed = [len({round(t["reward"], 2) for t in g}) > 1 for g in groups]
        res = {"items": len(items), "g": a.g, f"pass@{a.g}": round(sum(hit) / len(hit), 3),
               "mean_reward": round(sum(t["reward"] for g in groups for t in g) / (len(items) * a.g), 3),
               "groups_with_signal": round(sum(mixed) / len(mixed), 3)}
        print(json.dumps(res), flush=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        with open(Path(a.out) / "passk.jsonl", "w", encoding="utf-8") as f:
            for it, g in zip(items, groups):
                f.write(json.dumps({"id": it.get("id"), "question": it["question"], "answer": it["answer"],
                                    "answers": [t["answer"] for t in g], "rewards": [t["reward"] for t in g]}, ensure_ascii=False) + "\n")
        return
    pol = Policy(Path(a.model), device, train=True)
    ref = Policy(Path(a.model), device, train=False)
    params = [p for p in pol.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    cfg = {"rl": {k: v for k, v in vars(a).items() if k != "selftest"}, "start": a.model}
    print(f"[rl] device={device} train={len(train)} val={len(val)} trainable={sum(p.numel() for p in params)}", flush=True)
    rng = random.Random(a.seed)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        batch = rng.sample(train, a.q)
        groups = list(zip(batch, rollout_many(pol, batch, a.g, a.temperature, a.max_new)))
        t_roll = time.time() - t0
        pol.model.train(); opt.zero_grad()
        n_used, kls, losses = 0, [], []
        todo = []
        for it, trajs in groups:
            rs = torch.tensor([t["reward"] for t in trajs])
            if rs.std() < 1e-6:
                continue
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            todo += [(t, float(adv[i])) for i, t in enumerate(trajs) if any(t["lab"])]
        for t, A in todo:
            lp = pol.token_logprobs(t["ids"], t["lab"])
            with torch.no_grad():
                rlp = ref.token_logprobs(t["ids"], t["lab"])
            d = rlp - lp
            kl = torch.exp(d) - d - 1  # k3 estimator, >= 0
            loss = (-A * lp + a.beta * kl).mean() / len(todo)
            loss.backward()
            kls.append(float(kl.mean())); losses.append(float(loss) * len(todo)); n_used += 1
        if n_used:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        by = defaultdict(list)
        for it, trajs in groups:
            for tr in trajs:
                by[it["type"]].append(tr["reward"]); by["all"].append(tr["reward"])
        rec = {"step": step, "reward": {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())},
               "groups_with_signal": sum(1 for _, tr in groups if torch.tensor([x["reward"] for x in tr]).std() >= 1e-6),
               "kl": round(sum(kls) / len(kls), 4) if kls else 0.0, "loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
               "searches": round(sum(t["n_search"] for _, tr in groups for t in tr) / (a.q * a.g), 2),
               "answered": round(sum(t["answer"] is not None for _, tr in groups for t in tr) / (a.q * a.g), 2),
               "secs": round(time.time() - t0), "rollout_secs": round(t_roll)}
        if step % a.val_every == 0 or step == 1:
            rec["val"] = validate(pol, val, a.max_new)
        log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            pol.model.save_pretrained(out / f"step{step}", safe_serialization=True)
            pol.tok.save_pretrained(out / f"step{step}")
            (out / f"step{step}" / "rl_config.json").write_text(json.dumps({**cfg, "step": step}, ensure_ascii=False, indent=1))
    print("RL_DONE", flush=True)


if __name__ == "__main__":
    main()
