#!/usr/bin/env python3
"""SearchSLM teacher pipeline v2 (spec v0 §7): items (A–E) -> verification gates -> agent trajectories -> trajectory gates.

Teacher: Gemma 4 26B-A4B served by LM Studio (OpenAI-compatible, localhost:1234).
  python corpus/search_teacher.py noise                       # build the noise layer (E) from local train/heldout docs
  python corpus/search_teacher.py items --split train --n 6000 --out data/search/gen/items_train.jsonl
  python corpus/search_teacher.py traj  --items data/search/gen/items_train.jsonl --out data/search/gen/traj_train.jsonl
Every stage appends and resumes (skips ids already written), so an interrupted run can be restarted.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "corpus"))
import os
API, MODEL = os.environ.get("TEACHER_API", "http://localhost:1234/v1/chat/completions"), "teacher-bench"  # TEACHER_API: e.g. a tunnelled remote llama-server
TODAY = "2026년 11월 5일"
K, MAX_SEARCH = 8, 4
TYPE_RATIO = {"A": 0.10, "B": 0.30, "C": 0.25, "D": 0.15, "E": 0.20}
STATS = Counter()
_lock = threading.Lock()


# ---------------- LLM ----------------
REASONING = {"muse": {"reasoning_effort": "low"}, "qwen-agent": {"reasoning_effort": "none"}}  # reasoning models: thinking tokens count against max_tokens, so give room


def chat(messages, max_tokens=400, temperature=0.7, model: str | None = None) -> str:
    m = model or MODEL
    extra = REASONING.get(m, {})
    body = json.dumps({"model": m, "messages": messages, "max_tokens": max_tokens + (1500 if extra else 0), "temperature": temperature, **extra}).encode()
    err = None
    for i in range(12):  # up to ~6 min: rides out an LM Studio model crash while the watchdog reloads it
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(API, body, {"Content-Type": "application/json"}), timeout=900))
            with _lock:
                STATS["calls"] += 1; STATS["in"] += r["usage"]["prompt_tokens"]; STATS["out"] += r["usage"]["completion_tokens"]
            return r["choices"][0]["message"]["content"] or ""
        except Exception as e:  # LM Studio occasionally drops a connection under load, or the model crashed
            err = e; time.sleep(3 if i < 3 else 30)
    raise err


def last_json(s: str):
    starts = [m.start() for m in re.finditer(r"\{", s)]
    ends = [m.end() for m in re.finditer(r"\}", s)]
    for st in starts:  # outermost object first
        for en in reversed(ends):
            if en > st:
                try:
                    return json.loads(s[st:en])
                except Exception:
                    pass
    return None


def ask_json(system: str, user: str, max_tokens=400, temperature=0.7):
    return last_json(chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens, temperature)) or {}


norm = lambda s: re.sub(r"[\s\"'“”‘’.,·()\[\]「」『』<>《》:;!?~\-]", "", str(s)).lower()
BAD = ("이 문서", "위 글", "본문", "이 문단", "위 문단", "문단 A", "문단 B", "주어진", "제시된", "위의",
       "이번 ", "해당 ", "위 ", "이 연구", "이 기사", "이 글", "이 사업", "이 제품", "이 행사")  # not self-contained


JUDGE_SYS = "예측이 정답과 같은 사실을 말하면 '예', 아니면 '아니오'만 출력하세요."


def same(question: str, gold: str, pred: str, model: str | None = None, max_tokens: int = 4) -> bool:
    if not pred:
        return False
    if norm(gold) and norm(gold) in norm(pred):
        return True
    out = chat([{"role": "system", "content": JUDGE_SYS},
                {"role": "user", "content": f"질문: {question}\n정답: {gold}\n예측: {pred}"}], max_tokens, 0, model=model)
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().startswith("예")


def natural(question: str, answer: str) -> bool:
    out = chat([{"role": "system", "content": "정답이 질문이 묻는 대상(인물·장소·연도·수량·명칭 등)과 유형이 맞고, 그 자체로 완결된 자연스러운 답이면 '예', 아니면 '아니오'만 출력하세요."},
                {"role": "user", "content": f"질문: {question}\n정답: {answer}"}], 4, 0)
    return out.strip().startswith("예")


def read_answer(question: str, passages: list[str]) -> str:
    ctx = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages)) or "(문단 없음)"
    j = ask_json("주어진 문단만 근거로 질문에 짧게 답하세요. 문단으로 답할 수 없으면 answer를 \"모름\"으로 하세요. JSON 한 줄: {\"answer\": \"...\"}",
                 f"문단:\n{ctx}\n\n질문: {question}", 120, 0)
    return str(j.get("answer", ""))


# ---------------- corpus access ----------------
def load_layer(name: str) -> list[dict]:
    return [json.loads(l) for l in open(ROOT / f"data/search/corpus_v2/{name}.jsonl", encoding="utf-8")]


_retr: dict = {}
_rlock = threading.RLock()  # search() holds it while retriever() lazily loads


def retriever(name: str):
    with _rlock:
        if name not in _retr:
            from search_index import Retriever
            _retr[name] = Retriever(name)
        return _retr[name]


def search(name: str, q: str, k: int = K, exclude: set | None = None) -> list[dict]:
    """exclude: passage ids, or "doc:<doc_id>" entries that hide every passage of a document."""
    if LIVE_ALL or (name == "web" and "live" in H2):  # live: web_search goes to Serper; LIVE_ALL (live eval): local search -> today's Korean Wikipedia
        import search_live
        return search_live.wiki(q, k) if LIVE_ALL and name == "local" else search_live.search(q, k)
    ex = exclude or set()
    docs = {e[4:] for e in ex if e.startswith("doc:")}
    with _rlock:  # one MPS user at a time
        hits = retriever(name).search(q, k=k + len(ex) + (16 if docs else 0))
    hits = [h for h in hits if h["passage_id"] not in ex and h.get("doc_id") not in docs][:k]
    return [dict(h, title=TITLE_MAP[h["passage_id"]]) if h["passage_id"] in TITLE_MAP else h for h in hits]


LIVE_ALL = False  # set by the live eval: no fixed index anywhere, every search is a real web search
TITLE_MAP: dict[str, str] = {}  # passage_id -> displayed title (11/11 realism test: timeline records with differing titles)


# ---------------- item generators (one per type) ----------------
GEN_B = """당신은 한국어 검색 에이전트 학습 데이터 출제자입니다. 문단을 보고 사용자가 실제로 검색창에 물을 법한 사실 질문 1개를 만드세요.
- 질문은 문단을 본 적 없는 사람이 묻는 것처럼 자연스럽게, 문단 문장을 그대로 베끼지 말고 다른 표현으로 씁니다.
- '이 문서', '위 글', '본문', '주어진' 같은 표현 금지. 자기완결적이어야 합니다.
- 정답은 문단에 그대로 나오는 짧은 구(25자 이내). 질문 안에 정답이 들어가면 안 됩니다.
- 문단이 광고·목록·잡담이라 물을 사실이 없으면 {"skip": true}.
JSON 한 줄: {"question": "...", "answer": "..."}"""

GEN_C = """당신은 2단계(bridge) 검색 질문 출제자입니다. 문단 A는 '{y}'를 언급하고, 문단 B는 '{y}'에 대한 설명입니다.
1) 문단 A만 보고 답이 '{y}'가 되는 하위 질문 q1을 만듭니다.
2) 문단 B만 보고 '{y}'에 대한 사실을 묻는 하위 질문 q2와 그 정답 a2(문단 B에 그대로 나오는 25자 이내 구)를 만듭니다.
3) q1과 q2를 합쳐 '{y}'라는 이름을 쓰지 않는 하나의 자연스러운 질문을 만듭니다. 답은 a2이고, 질문이 묻는 대상과 a2의 유형(인물/장소/연도/수량/명칭)이 맞아야 합니다.
- '이 문서', '위 글', '문단 A' 같은 표현 금지. 자기완결적이어야 합니다. 질문 안에 a2가 들어가면 안 됩니다.
- 자연스러운 2단계 질문을 만들 수 없으면 {{"skip": true}}.
JSON 한 줄: {{"q1": "...", "q2": "...", "question": "...", "answer": "..."}}"""

GEN_A = """검색 없이 답할 수 있는 한국어 사용자 요청 {n}개를 만드세요. 종류를 골고루: 사칙연산·단위 환산, 짧은 번역, 인사·잡담, 문장 다듬기, 초등 수준 상식.
각 항목에 짧은 정답(또는 모범 응답의 핵심 구)을 붙이세요. 실시간 정보나 전문 지식이 필요한 질문은 넣지 마세요.
JSON 한 줄: {{"items": [{{"question": "...", "answer": "..."}}, ...]}}"""
GEN_A_TR = """검색 없이 답할 수 있는 짧은 번역 요청 {n}개를 만드세요. 한국어→영어와 영어→한국어가 대부분이고 일본어·중국어가 조금 섞입니다.
단어 하나 또는 짧은 일상 문장만, 묻는 말투는 다양하게('~를 영어로?', '~ 번역해 줘', '~는 한국어로 뭐야?'). 정답은 가장 표준적인 번역 하나.
JSON 한 줄: {{"items": [{{"question": "...", "answer": "..."}}, ...]}}"""


def gen_B(p: dict) -> dict | None:
    j = ask_json(GEN_B, f"제목: {p['title']}\n문단: {p['text']}", 200)
    if j.get("skip"):
        return None
    q, a = str(j.get("question", "")), str(j.get("answer", ""))
    gates = {"format": bool(q and a), "answer_in_passage": norm(a) in norm(p["text"]), "answer_not_in_question": norm(a) not in norm(q),
             "no_meta_phrase": not any(b in q for b in BAD), "answer_short": len(a) <= 30}
    if all(gates.values()):
        gates["answerable_with_gold"] = same(q, a, read_answer(q, [p["title"] + ": " + p["text"]]))
    return {"question": q, "answer": a, "gold": [p["passage_id"]], "hops": 1, "needs_search": True, "in_corpus": True, "gates": gates}


def gen_C(pair) -> dict | None:
    pa, pb, y = pair
    j = ask_json(GEN_C.format(y=y), f"문단 A (제목: {pa['title']}): {pa['text']}\n\n문단 B (제목: {pb['title']}): {pb['text']}", 300)
    if j.get("skip"):
        return None
    q, a = str(j.get("question", "")), str(j.get("answer", ""))
    gates = {"format": bool(q and a), "answer_in_B": norm(a) in norm(pb["text"]), "bridge_hidden": norm(y) not in norm(q),
             "answer_not_in_question": norm(a) not in norm(q), "no_meta_phrase": not any(b in q for b in BAD), "answer_short": len(a) <= 30}
    if all(gates.values()):
        both = [pa["title"] + ": " + pa["text"], pb["title"] + ": " + pb["text"]]
        gates["answerable_with_both"] = same(q, a, read_answer(q, both))
        if gates["answerable_with_both"]:
            gates["not_answerable_with_A_only"] = not same(q, a, read_answer(q, both[:1]))
        if all(gates.values()):
            gates["natural_answer"] = natural(q, a)
    return {"question": q, "answer": a, "bridge": y, "q1": j.get("q1"), "q2": j.get("q2"), "gold": [pa["passage_id"], pb["passage_id"]],
            "hops": 2, "needs_search": True, "in_corpus": True, "gates": gates}


def gen_D(p: dict) -> dict | None:
    it = gen_B(p)
    if it is None:
        return None
    if all(it["gates"].values()):
        local = search("local", it["question"])
        it["gates"]["not_in_local"] = not same(it["question"], it["answer"],
                                               read_answer(it["question"], [h["title"] + ": " + h["text"] for h in local]))
    it.update(in_corpus=False)
    return it


def gen_A(n: int, prompt: str = GEN_A) -> list[dict]:
    j = ask_json(prompt.format(n=n), "생성하세요.", 1500, 0.9)
    return [{"question": str(x.get("question", "")), "answer": str(x.get("answer", "")), "gold": [], "hops": 0, "needs_search": False,
             "in_corpus": True, "gates": {"format": bool(x.get("question") and x.get("answer"))}} for x in j.get("items", [])]


# ---------------- noise layer (E) ----------------
CONFLICT = """아래 위키 문단의 사실 하나(연도·수량·이름·장소 중 하나)를 다른 값으로 바꾼, 사용자가 대충 적어 둔 {kind}를 만드세요.
- {style}
- 바꾼 사실의 원래 값과 바뀐 값을 기록하세요. 바꿀 만한 사실이 없으면 {{"skip": true}}.
JSON 한 줄: {{"text": "...", "from": "원래 값", "to": "바뀐 값"}}"""
PARA = """아래 문단과 같은 내용을 다른 문장으로 다시 쓴 {kind}를 만드세요. 사실은 하나도 바꾸지 마세요. {style}
JSON 한 줄: {{"text": "..."}}"""
STYLE = {"memo": "개인 메모처럼 짧고 건조하게 (예: '- 1998년 설립, 본사 서울').",
         "chat": "메신저 대화처럼 구어체로 (예: 'A: 그거 언제였지? B: 아마 1998년?')."}
TIMELINE = """개인 업무 메모리에 쌓일 법한 가상의 사실 하나가 시간이 지나며 바뀐 기록을 만드세요. 기록의 주인공은 '{name}'입니다. 바뀌는 사실은 '{fact}' 종류로 하세요 (실존 인물·회사 금지).
topic은 '{name}'을 그대로 포함한 짧은 제목(예: '{name} {fact}')이고, question에도 '{name}'을 그대로 넣으세요.
기록 2~3개, 각 기록은 날짜(2019-01-01~2026-10-31, 오래된 것부터)와 출처(memo 또는 chat)와 짧은 본문. 마지막 기록의 값이 현재 값입니다.
그리고 현재 값을 묻는 자연스러운 질문과 정답(현재 값, 짧은 구)을 붙이세요.
JSON 한 줄: {{"topic": "...", "records": [{{"date": "YYYY-MM-DD", "source_type": "memo|chat", "text": "..."}}], "question": "...", "answer": "...", "old_answers": ["..."]}}"""


def rand_date(rng: random.Random, lo=2020, hi=2026) -> str:
    return f"{rng.randint(lo, hi)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def make_noise(p: dict, kind: str, rng: random.Random) -> list[dict]:
    src = rng.choice(["memo", "chat"])
    base = {"title": p["title"], "source_type": src, "date": rand_date(rng), "url": None, "layer": "noise",
            "split": p["split"], "license": "synthetic", "doc_id": "noise:" + p["passage_id"]}
    if kind == "paraphrase":
        j = ask_json(PARA.format(kind="메모" if src == "memo" else "대화", style=STYLE[src]), f"제목: {p['title']}\n문단: {p['text']}", 300)
        if not j.get("text"):
            return []
        return [{**base, "passage_id": f"noise:para:{p['passage_id']}", "text": j["text"],
                 "origin": {"derived_from": p["passage_id"], "perturbation": "paraphrase"}}]
    j = ask_json(CONFLICT.format(kind="메모" if src == "memo" else "대화", style=STYLE[src]), f"제목: {p['title']}\n문단: {p['text']}", 300)
    if j.get("skip") or not j.get("text") or not j.get("from") or not j.get("to") or norm(j["from"]) == norm(j["to"]) \
            or norm(j["from"]) not in norm(p["text"]) or norm(j["to"]) not in norm(j["text"]):
        return []
    return [{**base, "passage_id": f"noise:conf:{p['passage_id']}", "text": j["text"],
             "origin": {"derived_from": p["passage_id"], "perturbation": "conflict", "changed_fact": {"from": j["from"], "to": j["to"]}}}]


# v1 timelines collapsed onto a few generic topics ("신규 프로젝트 서버 주소" x66), so almost every question was ambiguous.
# Each timeline now gets a code-assigned unique owner name; the question must carry it.
TL_A = ("별빛", "온새미", "해오름", "달무리", "푸른솔", "은하수", "새벽녘", "꽃샘", "물푸레", "산들", "나래", "가람", "누리", "한빛",
        "초롱", "미르", "다온", "도담", "라온", "아라", "윤슬", "여울", "솔바람", "노을", "코랄", "모카", "자몽", "청귤", "호두", "보리")
TL_B = ("독서모임", "풋살팀", "사진 동아리", "코딩 스터디", "밴드", "도예 공방", "연구실", "봉사단", "등산 모임", "보드게임 모임",
        "캠핑 동호회", "합창단", "스타트업", "디자인 팀", "영어 회화 모임", "러닝 크루", "요리 교실", "웹툰 작업실", "주말농장", "탁구 동호회")
TL_FACT = ("정기 모임 장소", "정기 모임 요일", "회비 금액", "공용 와이파이 비밀번호", "단체 대화방 이름", "담당자 연락처", "서버 주소",
           "공유 폴더 위치", "마감일", "대표 메일 주소", "회의 시간", "창고 비밀번호", "주차 위치", "총무 이름", "발표 순서")


def tl_name(i: int) -> tuple[str, str]:
    rng = random.Random(1000 + i)
    a, b = TL_A[i % len(TL_A)], TL_B[(i // len(TL_A)) % len(TL_B)]  # 600 unique owners before a pair repeats
    return f"{a} {b}", rng.choice(TL_FACT)


def make_timeline(i: int, split: str) -> tuple[list[dict], dict | None]:
    name, fact = tl_name(i)
    j = ask_json(TIMELINE.format(name=name, fact=fact), "생성하세요.", 600, 0.9)
    recs, q, a = j.get("records") or [], str(j.get("question", "")), str(j.get("answer", ""))
    if len(recs) < 2 or not q or not a or norm(a) not in norm(recs[-1].get("text", "")) \
            or name not in q or name not in str(j.get("topic", "")):
        return [], None
    dates = [r.get("date", "") for r in recs]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        return [], None
    rows = [{"passage_id": f"noise:tl2-{i}#{k}", "title": j.get("topic", "메모"), "text": r["text"],
             "source_type": r.get("source_type") if r.get("source_type") in ("memo", "chat") else "memo", "date": r["date"],
             "url": None, "doc_id": f"noise:tl2-{i}", "layer": "noise", "split": split, "license": "synthetic",
             "origin": {"derived_from": None, "perturbation": "outdated" if k < len(recs) - 1 else "current"}} for k, r in enumerate(recs)]
    item = {"question": q, "answer": a, "old_answers": j.get("old_answers") or [], "gold": [rows[-1]["passage_id"]], "hops": 1, "owner": name,
            "needs_search": True, "in_corpus": True, "e_kind": "timeline", "gates": {"format": True, "current_in_last": True}}
    return rows, item


def run_noise(n_para: int, n_conf: int, n_tl: int, workers: int) -> None:
    rng = random.Random(21)
    local = [p for p in load_layer("local") if len(p["text"]) >= 80]
    pick = {s: [p for p in local if p["split"] == s] for s in ("train", "heldout", "dev")}
    jobs = []
    for s, frac in (("train", 0.8), ("heldout", 0.12), ("dev", 0.08)):
        ps = rng.sample(pick[s], int((n_para + n_conf) * frac))
        jobs += [(p, "paraphrase") for p in ps[:int(n_para * frac)]] + [(p, "conflict") for p in ps[int(n_para * frac):]]
    out_rows, e_items = [], []
    with ThreadPoolExecutor(workers) as ex:
        for rows in ex.map(lambda pk: make_noise(pk[0], pk[1], random.Random(zlib.crc32(pk[0]["passage_id"].encode()))), jobs):
            out_rows += rows
        splits = ["train"] * int(n_tl * 0.8) + ["heldout"] * int(n_tl * 0.12) + ["dev"] * (n_tl - int(n_tl * 0.8) - int(n_tl * 0.12))
        for rows, item in ex.map(lambda a: make_timeline(*a), list(enumerate(splits))):
            if item:
                item["split"] = rows[0]["split"]; out_rows += rows; e_items.append(item)
    with open(ROOT / "data/search/corpus_v2/noise.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ROOT / "data/search/gen/e_timeline_items.jsonl", "w", encoding="utf-8") as f:
        for it in e_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"noise_rows": len(out_rows), "by_kind": Counter(r["origin"]["perturbation"] for r in out_rows),
                      "timeline_items": len(e_items), "teacher": dict(STATS)}, ensure_ascii=False), flush=True)


def run_timeline(n_tl: int, workers: int) -> None:
    """Regenerate only the timeline records (unique owners); paraphrase/conflict noise is kept. Rebuild the local index after."""
    path = ROOT / "data/search/corpus_v2/noise.jsonl"
    kept = [r for r in map(json.loads, open(path, encoding="utf-8")) if not r["doc_id"].startswith("noise:tl")]
    splits = ["train"] * int(n_tl * 0.8) + ["heldout"] * int(n_tl * 0.12) + ["dev"] * (n_tl - int(n_tl * 0.8) - int(n_tl * 0.12))
    new, e_items = [], []
    with ThreadPoolExecutor(workers) as ex:
        for rows, item in ex.map(lambda a: make_timeline(*a), list(enumerate(splits))):
            if item:
                item["split"] = rows[0]["split"]; new += rows; e_items.append(item)
    with open(path, "w", encoding="utf-8") as f:
        for r in kept + new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ROOT / "data/search/gen/e_timeline_items.jsonl", "w", encoding="utf-8") as f:
        for it in e_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"timeline_rows": len(new), "timeline_items": len(e_items), "by_split": Counter(i["split"] for i in e_items),
                      "teacher": dict(STATS)}, ensure_ascii=False), flush=True)


def timeline_items(split: str, used: set) -> list[dict]:
    out = []
    for line in open(ROOT / "data/search/gen/e_timeline_items.jsonl", encoding="utf-8"):
        it = json.loads(line)
        if it.get("split") == split and it["gold"][0] not in used:
            # drop the question if another timeline that could answer it also comes up: with unique owners (v2) only a
            # timeline of the same owner counts ("꽃샘 독서모임" records are no answer to a "별빛 독서모임" question)
            doc, owner = it["gold"][0].split("#")[0], it.get("owner")
            others = {h["doc_id"] for h in search("local", it["question"]) if h["doc_id"].startswith("noise:tl") and h["doc_id"] != doc
                      and (owner is None or owner in h["title"])}
            it["gates"]["unambiguous"] = not others
            out.append(dict(it, type="E"))
    return out


def run_tl_items(split: str, out: Path) -> None:
    results = timeline_items(split, set())
    for i, it in enumerate(results):
        it["id"] = f"{out.stem.replace('items_', '')}-{it['type']}-{i:05d}"; it["split"] = split; it["valid"] = all(it["gates"].values())
    with open(out, "w", encoding="utf-8") as f:
        for it in results:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(json.dumps({"split": split, "timeline_items": len(results), "valid": sum(r["valid"] for r in results)}), flush=True)


def e_conflict_items(split: str) -> list[dict]:
    """E items from conflict noise: ask the original fact; the low-reliability memo/chat says otherwise."""
    loc = {p["passage_id"]: p for p in load_layer("local")}
    items = []
    for line in open(ROOT / "data/search/corpus_v2/noise.jsonl", encoding="utf-8"):
        r = json.loads(line)
        o = r["origin"]
        if o["perturbation"] != "conflict" or r["split"] != split:
            continue
        p = loc[o["derived_from"]]
        items.append({"_passage": p, "_noise": r})
    return items


def gen_E_conflict(x) -> dict | None:
    p, r = x["_passage"], x["_noise"]
    f = r["origin"]["changed_fact"]
    j = ask_json("문단에서 아래 사실값이 정답이 되는 자연스러운 검색 질문 1개를 만드세요. 질문에 정답을 넣지 마세요. '이 문서' 같은 표현 금지. JSON 한 줄: {\"question\": \"...\"}",
                 f"제목: {p['title']}\n문단: {p['text']}\n정답이 될 사실값: {f['from']}", 150)
    q = str(j.get("question", ""))
    gates = {"format": bool(q), "answer_not_in_question": norm(f["from"]) not in norm(q) and norm(f["to"]) not in norm(q),
             "no_meta_phrase": not any(b in q for b in BAD)}
    return {"question": q, "answer": f["from"], "conflict_value": f["to"], "gold": [p["passage_id"]], "noise_ids": [r["passage_id"]],
            "hops": 1, "needs_search": True, "in_corpus": True, "e_kind": "conflict", "gates": gates}


# ---------------- item stage ----------------
JOSA = re.compile(r"(으로부터|에서는|에게서|으로서|으로써|이라는|라는|에서|에게|으로|로서|로써|까지|부터|처럼|보다|은|는|이|가|을|를|의|에|와|과|도|로|만|란)$")


def bridge_pairs(local: list[dict], split: str, n: int, rng: random.Random, avoid_docs: set | None = None) -> list:
    """(A, lead passage of doc Y, Y) where passage A mentions the title Y of another local document.
    avoid_docs: for dev/heldout, Y must be a document no train/dev item used as gold: an unrestricted Y reused popular
    train passages (57 of 69 heldout C items shared their second-hop gold passage with train). Y may come from any split
    (same-split-only ran out of bridges after ~40 items)."""
    lead: dict[str, dict] = {}
    for p in local:
        if avoid_docs is None or p["doc_id"] not in avoid_docs:
            lead.setdefault(p["title"], p)
    titles = {t for t in lead if len(t) >= 3 and not re.fullmatch(r"[\d\s]+", t)}
    cand = [p for p in local if p["split"] == split and len(p["text"]) > 60]
    rng.shuffle(cand)
    out, used = [], Counter()
    for p in cand:
        if len(out) >= n:
            break
        words = [JOSA.sub("", w) for w in re.findall(r"[^\s,.()\[\]「」\"'“”]+", p["text"])]
        grams = {" ".join(words[i:i + k]) for k in (3, 2, 1) for i in range(len(words) - k + 1)}
        hits = sorted((g for g in grams if g in titles and g != p["title"] and used[g] < 3), key=len, reverse=True)
        if hits:
            y = hits[0]; used[y] += 1; out.append((p, lead[y], y))
    return out


def run_items(split: str, n: int, out: Path, workers: int, seed: int, only: str = "ABCDE", a_translate: bool = False) -> None:
    """only: generate just these types, n items of each (v4 targeted top-ups); otherwise n is split by TYPE_RATIO."""
    rng = random.Random(seed)
    local, web = load_layer("local"), load_layer("web")
    want = {t: int(n * r) for t, r in TYPE_RATIO.items()} if only == "ABCDE" else {t: n if t in only else 0 for t in TYPE_RATIO}
    over = 1.3  # generate extra; gates reject a share (smoke test: most items pass)
    used: set[str] = set()  # passages already behind an item of this split in earlier item files
    for prev in (ROOT / "data/search/gen").glob(f"items_{split}*.jsonl"):
        if prev.resolve() != out.resolve():
            for l in open(prev, encoding="utf-8"):
                r = json.loads(l); used.update(r.get("gold", [])); used.update(r.get("noise_ids", []))
    local = [p for p in local if p["passage_id"] not in used]
    web = [p for p in web if p["passage_id"] not in used]
    lp = [p for p in local if p["split"] == split and len(p["text"]) >= 80]
    wsplit = "heldout" if split == "heldout" else "train"
    wp = [p for p in web if p["split"] == wsplit and len(p["text"]) >= 150]
    if split == "dev":  # web has no dev split: dev D items come from a disjoint slice of web train docs
        wp = [p for p in wp if zlib.crc32(p["doc_id"].encode()) % 10 == 0]
    elif split == "train":
        wp = [p for p in wp if zlib.crc32(p["doc_id"].encode()) % 10 != 0]
    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(workers) as ex:
        if want["B"]:
            bs = list(ex.map(gen_B, rng.sample(lp, int(want["B"] * over))))
            results += [dict(it, type="B") for it in bs if it]
            print(f"[items] B {len(results)} {time.time()-t0:.0f}s", flush=True)
        if want["C"]:
            avoid = None
            if split != "train":
                avoid = {g.split("#")[0] for f in [*(ROOT / "data/search/gen").glob("items_train*.jsonl"), *(ROOT / "data/search/gen").glob("items_dev*.jsonl")]
                         for l in open(f, encoding="utf-8") for g in json.loads(l).get("gold", [])}
            pairs = bridge_pairs(local, split, int(want["C"] * over * 1.3), rng, avoid)
            results += [dict(it, type="C") for it in ex.map(gen_C, pairs) if it]
            print(f"[items] C done {time.time()-t0:.0f}s", flush=True)
        if want["D"]:
            results += [dict(it, type="D") for it in ex.map(gen_D, rng.sample(wp, int(want["D"] * over * 1.2))) if it]
            print(f"[items] D done {time.time()-t0:.0f}s", flush=True)
        if want["A"]:
            gen = (lambda k: gen_A(k, GEN_A_TR)) if a_translate else gen_A
            a_items = [x for batch in ex.map(gen, [20] * max(1, int(want["A"] * over / 20) + 1)) for x in batch]
            results += [dict(it, type="A") for it in a_items]
        if want["E"]:
            ec = [x for x in e_conflict_items(split) if x["_noise"]["passage_id"] not in used]
            rng.shuffle(ec)
            results += [dict(it, type="E") for it in ex.map(gen_E_conflict, ec[:int(want["E"] * 0.6 * over)]) if it]
    if want["E"]:
        results += timeline_items(split, used)
    for i, it in enumerate(results):
        it["id"] = f"{out.stem.replace('items_', '')}-{it['type']}-{i:05d}"; it["split"] = split; it["valid"] = all(it["gates"].values())
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in results:
            f.write(json.dumps({k: v for k, v in it.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    summ = {t: {"generated": sum(r["type"] == t for r in results), "valid": sum(r["type"] == t and r["valid"] for r in results)} for t in "ABCDE"}
    gate_fail = Counter(g for r in results for g, ok in r["gates"].items() if not ok)
    print(json.dumps({"split": split, "types": summ, "gate_failures": gate_fail, "secs": round(time.time() - t0), "teacher": dict(STATS)}, ensure_ascii=False), flush=True)


# ---------------- trajectory stage ----------------
AGENT = f"""오늘은 {TODAY}입니다. 당신은 한국어 검색 에이전트입니다. 매 턴 JSON 한 줄로 행동 하나만 출력합니다.
행동:
- {{"think": "...", "action": "search", "query": "..."}} : 로컬 메모리 검색 (위키 문서, 사용자 메모·대화 기록).
- {{"think": "...", "action": "web_search", "query": "..."}} : 웹 검색.
- {{"think": "...", "action": "answer", "evidence": [번호, ...], "answer": "..."}} : 최종 답.
검색 결과는 [번호] (출처 · 날짜) 제목: 본문 형식이며, 번호는 대화 전체에서 이어집니다.
같은 주제의 메모·대화 기록이 여럿이면 날짜에 '최신' 또는 '이전 기록'이 붙고, 거의 같은 내용이면 '[n]번과 중복'이 붙습니다.
규칙:
- 계산, 번역, 인사·대화, 초등 수준 상식만 검색 없이 바로 answer (evidence는 []).
- 그 밖의 사실 질문은 기억으로 답하지 말고 search부터 합니다. 필요하면 검색어를 바꿔 다시 검색합니다.
- 검색어는 한국어로 짧게 쓰되, 질문의 고유명사·연도·핵심어를 빠뜨리지 않습니다.
- 질문이 두 단계 정보를 요구하면(예: 무엇인가를 먼저 알아내야 그다음을 찾을 수 있으면) 단계마다 따로 검색합니다.
- web_search는 로컬 검색으로 근거를 못 찾았거나, 실시간·최신 정보일 때만 씁니다.
- 같은 사실에 대해 기록이 엇갈리면 날짜가 최신이고 출처가 믿을 만한 것(위키 > 메모 > 대화)을 따르고, 엇갈린다는 점을 답에 짧게 밝힙니다. 개인 메모의 사실이 바뀐 경우에는 가장 최신 기록을 따릅니다.
- 답은 핵심 구를 포함한 1~2문장. 근거 문단 번호를 evidence에 넣습니다.
- 근거를 찾지 못하면 "찾을 수 없습니다"라고 답하고 지어내지 않습니다.
- 검색은 최대 {MAX_SEARCH}번, 꼭 필요한 만큼만 합니다. think는 1~2문장."""


CURATE = True  # spec v0.2: memory-curation marks in the result block (False = raw display, the v1 trajectories)
CURATE_MODE = "title"  # "title": notes sharing a title form a timeline (v1); "content": same title OR the question's subject phrase (v2)


def subject_phrase(question: str) -> str:
    """Korean questions open with their topic: the first two content words ("현재 별빛 독서모임의 주차 위치는?" -> "별빛 독서모임")."""
    words = [w for w in (JOSA.sub("", x) for x in re.findall(r"[가-힣A-Za-z0-9]+", question)) if w and w not in QWORDS]
    return " ".join(words[:2])


def content_groups(notes: list[dict], question: str) -> list[list[dict]]:
    """Timeline groups without relying on titles (v2, 11/11): two memo/chat records belong together if they share a title
    OR both mention the question's subject phrase. On 198 train timelines with rewritten (all different) titles this fully
    grouped the asked-about records in 76% and mixed in another subject's record in 1 case; text-only grouping (e5 + rare
    words) mixed 22%, because same-kind records of other subjects ("별빛 스타트업 주차") read almost the same."""
    subj = re.sub(r"\s+", "", subject_phrase(question))
    has = [bool(subj) and subj in re.sub(r"\s+", "", n["text"]) for n in notes]
    parent = list(range(len(notes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            if notes[i]["title"] == notes[j]["title"] or (has[i] and has[j]):
                parent[find(i)] = find(j)
    out: dict[int, list[dict]] = {}
    for i, n in enumerate(notes):
        out.setdefault(find(i), []).append(n)
    return list(out.values())


def fmt(n: int, h: dict, mark: str = "") -> str:
    return f"[{n}] ({h['source_type']} · {h.get('date') or '날짜 미상'}{' · ' + mark if mark else ''}) {h['title']}: {h['text'][:600]}"


def _bigrams(t: str) -> set:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def curate_marks(block: list[dict], seen: dict[str, dict], num: dict[str, int], mode: str | None = None, question: str = "") -> dict[str, str]:
    """Rule-based curation from visible fields only (title, source_type, date, text), so the runtime can do the same.
    - near-duplicate of an earlier-numbered passage (char-bigram Jaccard; same title: 0.25 if a memo/chat is involved, else 0.5;
      different titles: 0.85) -> "[n]번과 중복"
    - timeline groups -> newest "최신", others "이전 기록". mode "title" (v1): memo/chat records sharing a title with no
      wiki page of that title in view. mode "content" (v2): content_groups(), same title OR both mention the question's subject
    """
    mode = mode or CURATE_MODE
    marks: dict[str, str] = {}
    wiki_titles = {o["title"] for o in seen.values() if o["source_type"] == "wiki"}
    is_note = lambda o: o["source_type"] in ("memo", "chat")
    gid: dict[str, int] = {}
    cgroups: list[list[dict]] = []
    if mode == "content":
        cgroups = [g for g in content_groups([o for o in seen.values() if is_note(o) and o.get("date")], question) if len(g) >= 2]
        gid = {o["passage_id"]: k for k, g in enumerate(cgroups) for o in g}
    for h in block:
        b = _bigrams(h["text"])
        for pid, o in seen.items():
            # same-title notes with no wiki page in view are a timeline (최신/이전 기록 below), not duplicates:
            # successive records of a changed fact share 42% of pairs at Jaccard >= 0.25 (corpus v2 tl2), which hid '최신' (v3 bug)
            if mode == "title" and is_note(o) and is_note(h) and o["title"] == h["title"] and h["title"] not in wiki_titles:
                continue
            if mode == "content" and pid in gid and gid.get(h["passage_id"]) == gid[pid]:
                continue
            if pid != h["passage_id"] and num[pid] < num[h["passage_id"]]:
                ob = _bigrams(o["text"])
                # thresholds measured on corpus v2: same-title memo/chat paraphrases have median Jaccard 0.26 (0.25 catches 55%,
                # 0.6% false positives on different passages of one wiki doc); templated pages ("OO항은 ... 지정") need 0.85
                note = "memo" in (o["source_type"], h["source_type"]) or "chat" in (o["source_type"], h["source_type"])
                need = (0.25 if note else 0.5) if o["title"] == h["title"] else 0.85
                if b and ob and len(b & ob) / len(b | ob) >= need:
                    marks[h["passage_id"]] = f"[{num[pid]}]번과 중복"; break
    groups: dict[str, list[dict]] = {}
    for o in seen.values():
        groups.setdefault(o["title"], []).append(o)
    timelines = cgroups if mode == "content" else []
    if mode == "title":
        for title, g in groups.items():
            notes = [o for o in g if o["source_type"] in ("memo", "chat") and o.get("date")]
            if len(notes) >= 2 and not any(o["source_type"] == "wiki" for o in g):
                timelines.append(notes)
    for notes in timelines:
        newest = max(notes, key=lambda o: o["date"])
        for o in notes:
            if o["passage_id"] in {h["passage_id"] for h in block} and o["passage_id"] not in marks:
                marks[o["passage_id"]] = "최신" if o is newest else "이전 기록"
    return marks


HINT_C = ("\n\n(데이터 생성용 비공개 안내) 이 질문은 두 단계입니다. 먼저 \"{q1}\"의 답을 검색으로 알아내고, 그 답을 넣어 \"{q2}\"를 따로 검색하세요. "
          "두 번째 검색 결과를 근거로 답하고 기억으로 답하지 마세요. think에 이 안내를 언급하지 마세요.")


# ---------------- runtime harness (rule-only; uses what the deployed system has: the question, shown passages, our reranker) ----------------
# KURE MaxSim(question, passage), 0..64. Calibrated on 240 train items (items_train2), never dev: gold passages median 47-50
# (5th pct B 43.3, D 41.9; multi-hop C last hop only 39), best non-gold hit median 40.8, 25th pct 36.7
REL_WEB = 42.0  # below: try the web before answering (cheap if wrong: one extra search)
REL_NF = 36.0   # below even after the web: answer "찾을 수 없습니다" (costly if wrong, so a lower bar that spares C)
REL_NF_WEB = 44.0  # Harness² r1 "nfweb": once the web was searched, a weaker best passage means the answer is made up. Tune set
# (v4): answers after a web search, best shown passage median 47.9 when correct vs 42.2 when fabricated on "nowhere" items
NOT_FOUND = ("찾을 수 없", "확인할 수 없", "알 수 없", "정보가 없")
H2: set[str] = set()  # Harness² candidate rules switched on by name (outputs/harness2/round_N.json records each one)


def fix_query(question: str, query, first: bool = True) -> str:
    """Repair a query that dropped or garbled the question's words (the 0.6B misspells names: 플라센부르크 -> 플라스부르크).
    The "must keep a question word" fallback applies to the first query only: a multi-hop second query is about the
    bridge entity found in step one and legitimately shares no word with the question."""
    import difflib
    q = str(query or "")
    words = [w for w in (JOSA.sub("", w) for w in re.findall(r"[가-힣A-Za-z0-9]+", question)) if len(w) >= 2 and w not in QWORDS]
    for t in set(re.findall(r"[가-힣A-Za-z0-9]+", q)):
        if len(t) >= 2 and t not in question:
            best = max(words, key=lambda w: difflib.SequenceMatcher(None, t, w).ratio(), default=None)
            if best and abs(len(best) - len(t)) <= 1 and difflib.SequenceMatcher(None, t, best).ratio() >= 0.6:
                q = q.replace(t, best)
    return q if not first or query_ok(question, q) else " ".join(dict.fromkeys(words)) or question


def fix_first(question: str, text) -> str:
    """fstfix: restore leading syllables the 0.6B drops when a question word starts a JSON string (순신 -> 이순신, 찌개 -> 김치찌개).
    Only the first token (the drop happens right after the opening quote; mid-sentence short forms like 비디오 are fine),
    and only if it is not itself a question word and is a 1-2 syllable shorter suffix of exactly one."""
    s = str(text or "")
    m = re.match(r"\s*([가-힣A-Za-z0-9]+)", s)
    if not m:
        return s
    qw = {JOSA.sub("", w) for w in re.findall(r"[가-힣A-Za-z0-9]+", question)} | set(re.findall(r"[가-힣A-Za-z0-9]+", question))
    t = m.group(1); core = JOSA.sub("", t)
    if len(core) < 2 or core in qw or t in qw:
        return s
    full = {w for w in qw if w.endswith(core) and 1 <= len(w) - len(core) <= 2}
    return s[:m.start(1)] + full.pop() + s[m.start(1) + len(core):] if len(full) == 1 else s


# harness v2 (11/10 plan P2/P3)
NO_SEARCH_OK = re.compile(r"번역|영어로|한국어로|일본어|중국어|영어 표현|고쳐|다듬|맞춤법|안녕|고마|반가|계산|더하기|빼기|곱하기|나누기|\d\s*[-+*/×÷]\s*\d")
FACTUAL = re.compile(r"무엇|누구|어디|언제|몇|어느|얼마|무슨|(은|는)\?\s*$")  # fact questions only: chit-chat and rewrites stay search-free
KO_OPS = {"더하기": "+", "빼기": "-", "곱하기": "*", "나누기": "/", "×": "*", "÷": "/", "x": "*"}
CHECK_TOK = re.compile(r"\d[\d,.]*\d|\d|[월화수목금토일]요일")


def calc(question: str) -> str | None:
    """Arithmetic questions answered by rule, not by the 0.6B ("123 곱하기 45는?" -> 5535). ast-walked, never eval()."""
    import ast
    import operator as op
    q = question
    for k, v in KO_OPS.items():
        q = q.replace(k, f" {v} ")
    m = re.search(r"[\d.()\s+\-*/^%]{3,}", q)
    if not m or not re.search(r"\d\s*[-+*/^%]\s*[\d(]", m.group()):
        return None
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops and not (isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100):
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        v = ev(ast.parse(m.group().replace("^", "**").strip(), mode="eval").body)
    except Exception:
        return None
    v = int(v) if float(v).is_integer() else round(v, 4)
    expr = re.sub(r"\s+", " ", m.group()).strip()
    return f"{expr} = {v:,}" if isinstance(v, int) else f"{expr} = {v}"


def unsupported(answer: str, question: str, passages: list[dict]) -> list[str]:
    """Numbers / weekdays in the answer that appear neither in the question nor in any shown passage (copy errors: 1942 -> 1945)."""
    hay = (question + " " + TODAY + " " + " ".join(p["title"] + " " + p["text"] for p in passages)).replace(",", "")
    return [t for t in CHECK_TOK.findall(answer) if t.replace(",", "") not in hay]


def relevance_each(question: str, passages: list[dict]) -> list[float]:
    with _rlock:
        return [float(x) for x in retriever("local").kure.rerank(question, [p["title"] + " " + p["text"] for p in passages])]


FOCUS_K = 4  # Harness² r2 "focus": re-ask the answer over the K passages most relevant to the question
REL_CONFIDENT = 48.0  # r2 "focus2": a not-found reply is overturned only if the best passage scores at least this
# (median KURE score of gold passages on the 240 train items used to calibrate REL_WEB; not fitted on the tune set)


def focus_answer(item: dict, seen: dict, shown: dict, query: str, web: bool, llm, need_confident: bool = False) -> dict | None:
    """Re-read in the trained one-search shape: question -> one search -> the top-K passages -> answer. On the tune set 16 of
    the misses had the gold passage among web results at positions 9-11, behind 8 local passages; the 0.6B gave up or
    misread. Evidence numbers are mapped back to the original numbering."""
    ps = list(seen.values())
    scores = relevance_each(item["question"], ps)
    top = [p for _, p in sorted(zip(scores, ps), key=lambda x: -x[0])[:FOCUS_K]]
    num = {p["passage_id"]: i + 1 for i, p in enumerate(top)}
    marks = curate_marks(top, {p["passage_id"]: p for p in top}, num, question=item["question"]) if CURATE else {}
    msgs = [{"role": "system", "content": AGENT}, {"role": "user", "content": "질문: " + item["question"]},
            {"role": "assistant", "content": json.dumps({"think": "관련 기록을 검색합니다.", "action": "web_search" if web else "search", "query": query}, ensure_ascii=False)},
            {"role": "user", "content": "[검색 결과]\n" + "\n".join(fmt(num[p["passage_id"]], p, marks.get(p["passage_id"], "")) for p in top)}]
    k = last_json(llm(msgs, 350, 0.0))
    if not (k and k.get("action") == "answer" and str(k.get("answer", "")).strip()):
        return None
    if "focus2" in H2 and need_confident and max(scores) < REL_CONFIDENT:
        return None
    back = {i + 1: shown[p["passage_id"]] for i, p in enumerate(top)}
    ev = [back[int(e)] for e in k.get("evidence", []) if str(e).isdigit() and int(e) in back]
    return {"think": str(k.get("think", "")), "action": "answer", "evidence": ev, "answer": str(k["answer"])}


def relevance(question: str, passages: list[dict]) -> float:
    if not passages:
        return 0.0
    with _rlock:
        return float(max(retriever("local").kure.rerank(question, [p["title"] + " " + p["text"] for p in passages])))


def run_agent(item: dict, temperature: float = 0.7, hint: bool = False, llm=None, harness: int = 0, extra: str = "") -> dict:
    """harness: 0 off, 1 = v1 rules (query fix, repeat block, weak->web / not-found), 2 = v1 + calc router, search for
    non-trivial no-search answers, re-ask on unsupported numbers or empty answers."""
    llm = llm or chat  # the student evaluation passes its own generator
    if harness and "xlate" in H2 and not item.get("_xl"):  # translator around the Korean-only agent (corpus/search_translate.py)
        import search_translate
        r = search_translate.route(item, lambda it: run_agent({**it, "_xl": True}, temperature, hint, llm, harness, extra))
        if r:
            return r
    used, fired, force_answer = set(), [], False  # harness state
    if harness >= 2 and (c := calc(item["question"])):
        fired.append("calc")
        final = {"think": "계산 질문이라 규칙으로 계산합니다.", "action": "answer", "evidence": [], "answer": c}
        return {"messages": [], "steps": [final], "final": final, "format_fails": 0, "shown": {}, "cited": [], "harness": fired}
    sys_prompt = AGENT + (HINT_C.format(q1=item.get("q1"), q2=item.get("q2")) if hint and item.get("q1") and item.get("q2") else "") + extra
    if LIVE_ALL or (harness and "live" in H2):  # live results need the real date, not the training-time one
        import search_live
        sys_prompt = sys_prompt.replace(TODAY, search_live.today())
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": ("질문:\n" if harness and "qnl" in H2 else "질문: ") + item["question"]}]  # qnl: "질문: X" puts X's first word in a space-prefixed token the 0.6B cannot copy after a JSON quote
    shown: dict[str, int] = {}
    seen: dict[str, dict] = {}
    steps, final, fails = [], None, 0
    exclude = set(item["gold"]) if item.get("in_corpus") is False and item.get("d_none") else set()
    if exclude and (item.get("recover") or item.get("exclude_doc")):  # "nowhere" items hide the whole source document: its other passages often hold the answer
        exclude |= {"doc:" + g.split("#")[0] for g in item["gold"]}
    for s in range(MAX_SEARCH + 1):
        if s == MAX_SEARCH or force_answer:
            msgs.append({"role": "user", "content": "검색 횟수를 모두 썼습니다. 이제 반드시 answer 행동을 출력하세요."})
        j = None
        for r in range(3):  # a malformed turn is resampled, never kept in the transcript (it would become SFT data)
            j = last_json(llm(msgs, 350, temperature + 0.15 * r))
            if j and j.get("action") in ("search", "web_search", "answer") and (j["action"] != "answer" or "answer" in j):
                break
            fails += 1; j = None
        if j is None:
            break
        if (s == MAX_SEARCH or force_answer) and j["action"] != "answer":
            break
        j = {k: j[k] for k in ("think", "action", "query", "evidence", "answer") if k in j}
        if harness and j["action"] != "answer":
            q = j.get("query") if "nofix" in H2 else fix_query(item["question"], fix_first(item["question"], j.get("query")) if "fstfix" in H2 else j.get("query"), first=not used)  # nofix: the 0.6B spelling repair can only hurt V2
            if q != j.get("query"):
                fired.append("query_fix"); j["query"] = q
            if (j["action"], q) in used:  # a repeated search returns the same block: try the web once, else stop searching
                if j["action"] == "search" and ("web_search", q) not in used:
                    fired.append("repeat->web"); j["action"] = "web_search"
                else:
                    fired.append("repeat->answer"); force_answer = True; continue
            used.add((j["action"], q))
        if harness >= 2 and j["action"] == "answer" and not seen and s == 0 and FACTUAL.search(item["question"]) and not NO_SEARCH_OK.search(item["question"]):
            fired.append("nosearch->search")  # the 0.6B "knows" wrong facts (국화 -> 오상고절): look it up once
            q = fix_query(item["question"], item["question"])
            j = {"think": "확인을 위해 검색합니다.", "action": "search", "query": q}; used.add(("search", q))
        if harness >= 2 and j["action"] == "answer":
            ans = str(j.get("answer", "")).strip()
            bad = (not ans) or (seen and not any(k in ans for k in NOT_FOUND) and unsupported(ans, item["question"], list(seen.values())))
            for _ in range(2 if bad else 0):  # re-ask the answer turn; keep the first reply whose numbers are all on screen
                k = last_json(llm(msgs, 350, 0.3))
                if k and k.get("action") == "answer" and str(k.get("answer", "")).strip() and not unsupported(str(k["answer"]), item["question"], list(seen.values())):
                    fired.append("reask_ok"); j = {x: k[x] for x in ("think", "action", "evidence", "answer") if x in k}; break
            else:
                if bad:
                    fired.append("reask_fail")
        if harness and "fstfix" in H2 and j["action"] == "answer" and j.get("answer"):
            fa = fix_first(item["question"], j["answer"])
            if fa != j["answer"]:
                fired.append("fstfix"); j["answer"] = fa
        if harness and "empty" in H2 and j["action"] == "answer" and not str(j.get("answer", "")).strip():
            fired.append("empty->not_found"); j["answer"] = "찾을 수 없습니다."  # Harness² r1: an empty reply is never right
        if harness and j["action"] == "answer" and seen and ({"focus", "focusall", "focus2"} & H2):
            web_now = any(a == "web_search" for a, _ in used)
            two_hop = sum(a == "search" for a, _ in used) >= 2
            if "focusall" in H2 or (web_now and not ("focus2" in H2 and two_hop)):
                last_q = next((st.get("query") for st in reversed(steps) if st.get("query")), item["question"])
                gave_up = any(k in str(j.get("answer", "")) for k in NOT_FOUND)
                f = focus_answer(item, seen, shown, last_q, web_now, llm, need_confident=gave_up)
                if f:
                    fired.append("focus"); j = f
        if harness and j["action"] == "answer" and seen:
            nf = any(k in str(j.get("answer", "")) for k in NOT_FOUND)
            rel = 1e9 if "nokure" in H2 else relevance(item["question"], list(seen.values())); weak = rel < REL_WEB  # nokure: KURE cannot tell answerable from not (AUROC 0.60)
            web_done = any(a == "web_search" for a, _ in used)
            multi_hop = sum(a == "search" for a, _ in used) >= 2  # r1 "mhskip": a 2-hop chain's final passage is far from the question
            nfweb = "nfweb" in H2 and web_done and rel < REL_NF_WEB and not ("mhskip" in H2 and multi_hop)
            if (nf or weak) and not web_done and s < MAX_SEARCH:
                last = next((st.get("query") for st in reversed(steps) if st.get("query")), item["question"])
                fired.append("weak->web"); j = {"think": "웹에서도 검색해 확인합니다." if "neutral" in H2 else "찾은 결과에 근거가 부족해 웹에서 검색합니다.", "action": "web_search", "query": last}
                used.add(("web_search", last))
            elif (rel < REL_NF or nfweb) and not nf:
                fired.append("weak->not_found"); j = {"think": "검색 결과에서 질문에 대한 근거를 찾지 못했습니다.", "action": "answer", "evidence": [], "answer": "찾을 수 없습니다."}
            elif nf and "찾을 수 없" not in j["answer"]:  # one canonical not-found reply ("관련 정보가 없습니다" -> "찾을 수 없습니다.")
                fired.append("nf_normalize"); j["answer"] = "찾을 수 없습니다."
        msgs.append({"role": "assistant", "content": json.dumps(j, ensure_ascii=False)})
        if j["action"] == "answer":
            steps.append(j); final = j; break
        hits = search("local" if j["action"] == "search" else "web", str(j.get("query", "")), exclude=exclude)
        if harness and "rawweb" in H2 and len(steps) == 0:  # live dev 11/19: the raw question finds what the agent's rewritten query misses; same cache entry as the RAG baseline
            ids = {h["passage_id"] for h in hits}
            extra = [h for h in search("web", item["question"], k=3) if h["passage_id"] not in ids]
            if extra:
                fired.append("rawweb"); hits = hits + extra
        for h in hits:
            shown.setdefault(h["passage_id"], len(shown) + 1); seen[h["passage_id"]] = h
        marks = curate_marks(hits, seen, shown, question=item["question"]) if CURATE else {}
        lines = [fmt(shown[h["passage_id"]], h, marks.get(h["passage_id"], "")) for h in hits]
        steps.append({**j, "results": [h["passage_id"] for h in hits]})
        msgs.append({"role": "user", "content": "[검색 결과]\n" + ("\n".join(lines) or "결과 없음")})
    if harness and "noans" in H2 and final is None:  # 3 malformed turns end the loop with no reply; "empty" never saw this path
        fired.append("noanswer->not_found")
        final = {"think": "답을 만들지 못했습니다.", "action": "answer", "evidence": [], "answer": "찾을 수 없습니다."}
        steps.append(final)
    msgs[0] = {"role": "system", "content": AGENT}  # the privileged hint never reaches the SFT transcript
    num2id = {v: k for k, v in shown.items()}
    cited = [num2id.get(int(e)) for e in (final or {}).get("evidence", []) if str(e).isdigit() and int(e) in num2id]
    return {"messages": msgs, "steps": steps, "final": final, "format_fails": fails, "shown": shown, "cited": cited, "harness": fired}


HINT_LOCAL_FIRST = ("\n\n(데이터 생성용 비공개 안내) 먼저 search로 로컬 메모리를 검색하세요. 로컬 결과에 질문의 답이 없으면 그 결과로 답하지 말고 "
                    "web_search로 검색한 뒤 답하세요. think에 이 안내를 언급하지 마세요.")
HINT_NOWHERE = ("\n\n(데이터 생성용 비공개 안내) 먼저 search로 로컬 메모리를, 그다음 web_search로 웹을 검색하세요. 어느 결과에도 질문의 답이 "
                "직접 나오지 않으면 비슷한 내용으로 추측하지 말고 \"찾을 수 없습니다\"라고 답하세요. think에 이 안내를 언급하지 마세요.")

HINT_REALTIME = ("\n\n(데이터 생성용 비공개 안내) 이 질문은 오늘 기준 최신 정보가 필요합니다. web_search로 검색하고, 결과에 오늘 날짜의 정보가 없으면 "
                 "예전 날짜의 정보로 답하지 말고 \"최신 정보를 찾을 수 없습니다\"라고 답하세요. think에 이 안내를 언급하지 마세요.")
HINTS = {"nowhere": HINT_NOWHERE, "realtime": HINT_REALTIME}


def grade(item: dict, run: dict, texts: dict) -> dict:
    f = run["final"] or {}
    ans = str(f.get("answer", ""))
    n_s = sum(s["action"] == "search" for s in run["steps"]); n_w = sum(s["action"] == "web_search" for s in run["steps"])
    nf = any(k in ans for k in ("찾을 수 없", "확인할 수 없", "알 수 없"))
    g = {"finished": bool(run["final"])}
    t = item["type"]
    if t == "A":
        g |= {"no_search": n_s + n_w == 0, "correct": same(item["question"], item["answer"], ans)}
    elif item.get("d_none"):
        g |= {"used_web": n_w > 0, "says_not_found": nf}
    else:
        g["correct"] = same(item["question"], item["answer"], ans)
        g["gold_cited"] = all(x in run["cited"] for x in item["gold"]) if t != "C" else item["gold"][1] in run["cited"]
        cited_text = " ".join(texts.get(c, "") for c in run["cited"])
        g["grounded"] = norm(item["answer"]) in norm(cited_text) or bool(run["cited"]) and same(item["question"], item["answer"], read_answer(item["question"], [texts.get(c, "") for c in run["cited"]]))
        if t == "C":
            g["multi_search"] = n_s >= 2
            g["hint_not_leaked"] = not any(w in str(st.get("think", "")) for st in run["steps"] for w in ("안내", "힌트", "비공개"))
        if t == "D":
            g["used_web"] = n_w > 0
        if t == "E" and item.get("e_kind") == "conflict":
            g["not_conflict_value"] = not (norm(item["conflict_value"]) in norm(ans) and norm(item["answer"]) not in norm(ans))
        if t == "E" and item.get("e_kind") == "timeline":
            g["not_old_value"] = not any(norm(o) and norm(o) in norm(ans) and norm(item["answer"]) not in norm(ans) for o in item.get("old_answers", []))
    if item.get("hint"):  # v6 items: the private hint must not surface in a think
        g["hint_not_leaked"] = not any(w in str(st.get("think", "")) for st in run["steps"] for w in ("안내", "힌트", "비공개"))
    if item.get("recover"):  # v4 recovery trajectories: local first, then the web; never mention the private hint
        acts = [st["action"] for st in run["steps"] if st["action"] != "answer"]
        g["local_then_web"] = bool(acts) and acts[0] == "search" and "web_search" in acts
        g["hint_not_leaked"] = not any(w in str(st.get("think", "")) for st in run["steps"] for w in ("안내", "힌트", "비공개"))
    if n_s + n_w:
        g["query_keeps_terms"] = query_ok(item["question"], next(st["query"] for st in run["steps"] if st["action"] != "answer"))
    g["n_search"], g["n_web"] = n_s, n_w
    return g


QWORDS = {"무엇", "어디", "언제", "누구", "어떤", "어느", "얼마", "몇", "무슨", "어떻게", "왜", "인가요", "입니까", "있나요", "되나요", "현재"}


def query_ok(question: str, query) -> bool:
    """First query must be Korean when the question is, and keep at least one content word of the question."""
    q = str(query or "")
    if re.search(r"[가-힣]", question) and not re.search(r"[가-힣]", q):
        return False
    words = {JOSA.sub("", w) for w in re.findall(r"[가-힣A-Za-z0-9]+", question)}
    words = {w for w in words if len(w) >= 2 and w not in QWORDS}
    return not words or any(w in q for w in words)


def run_traj(items_path: Path, out: Path, workers: int, attempts: int, limit: int | None, recover: bool = False) -> None:
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    items = [it for it in items if it["valid"]]
    rng = random.Random(5)
    for it in items:  # a third of D items become "nowhere" items: the gold web passage is hidden (half in recovery mode)
        if it["type"] == "D" and rng.random() < (0.5 if recover else 0.33):
            it["d_none"] = True
        if recover and it["type"] == "D":
            it["recover"] = True
    if limit:
        items = items[:limit]
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
    todo = [it for it in items if it["id"] not in done]
    texts = {}
    for name in ("local", "web", "noise"):
        p = ROOT / f"data/search/corpus_v2/{name}.jsonl"
        if p.exists():
            for l in open(p, encoding="utf-8"):
                r = json.loads(l); texts[r["passage_id"]] = r["title"] + ": " + r["text"]
    retriever("local"); retriever("web")
    print(f"[traj] {len(todo)} items to run ({len(done)} done)", flush=True)
    t0 = time.time(); wl = threading.Lock(); kept = Counter(); tried = Counter()

    def work(it):
        best = None
        for a in range(attempts):
            extra = HINTS[it["hint"]] if it.get("hint") else (HINT_NOWHERE if it.get("d_none") else HINT_LOCAL_FIRST) if it.get("recover") else ""
            run = run_agent(it, 0.7 if a == 0 else 0.9, hint=it["type"] == "C", extra=extra)
            g = grade(it, run, texts)
            ok = all(v for k, v in g.items() if not k.startswith("n_"))
            best = {"id": it["id"], "type": it["type"], "item": it, "attempt": a, "accepted": ok, "grades": g,
                    "steps": run["steps"], "messages": run["messages"], "cited": run["cited"]}
            if ok:
                break
        with wl:
            tried[it["type"]] += 1; kept[it["type"]] += best["accepted"]
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(best, ensure_ascii=False) + "\n")
            n = sum(tried.values())
            if n % 100 == 0:
                print(f"[traj] {n}/{len(todo)} {time.time()-t0:.0f}s kept={dict(kept)} tried={dict(tried)} tok={dict(STATS)}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    print(json.dumps({"tried": dict(tried), "kept": dict(kept), "secs": round(time.time() - t0), "teacher": dict(STATS)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("noise"); n.add_argument("--para", type=int, default=800); n.add_argument("--conf", type=int, default=1500)
    n.add_argument("--timeline", type=int, default=700); n.add_argument("--workers", type=int, default=12)
    i = sub.add_parser("items"); i.add_argument("--split", required=True); i.add_argument("--n", type=int, required=True)
    i.add_argument("--out", required=True); i.add_argument("--workers", type=int, default=12); i.add_argument("--seed", type=int, default=1)
    i.add_argument("--only", default="ABCDE"); i.add_argument("--a-translate", action="store_true")
    tl = sub.add_parser("timeline"); tl.add_argument("--n", type=int, default=600); tl.add_argument("--workers", type=int, default=12)
    ti = sub.add_parser("tlitems"); ti.add_argument("--split", required=True); ti.add_argument("--out", required=True)
    t = sub.add_parser("traj"); t.add_argument("--items", required=True); t.add_argument("--out", required=True)
    t.add_argument("--workers", type=int, default=10); t.add_argument("--attempts", type=int, default=2); t.add_argument("--limit", type=int)
    t.add_argument("--recover", action="store_true", help="v4 recovery trajectories for D items (private local-first / not-found hints)")
    a = ap.parse_args()
    (ROOT / "data/search/gen").mkdir(parents=True, exist_ok=True)
    if a.cmd == "noise":
        run_noise(a.para, a.conf, a.timeline, a.workers)
    elif a.cmd == "timeline":
        run_timeline(a.n, a.workers)
    elif a.cmd == "tlitems":
        run_tl_items(a.split, ROOT / a.out)
    elif a.cmd == "items":
        run_items(a.split, a.n, ROOT / a.out, a.workers, a.seed, a.only, a.a_translate)
    else:
        run_traj(ROOT / a.items, ROOT / a.out, a.workers, a.attempts, a.limit, a.recover)
