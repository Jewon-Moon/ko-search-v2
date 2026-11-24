#!/usr/bin/env python3
"""Live web search for the "실사용 모드" (harness flag `live`): Serper (Google, gl=kr) -> fetch result pages -> <=550-char passages.

Only web_search goes live; search (local memory) stays on the fixed index, so personal memo lookups never leave the Mac.
Never run eval items (sealed / tune / dev: some are AI Hub-derived) through it; the fixed index stays the eval setting.
Key: SERPER_API_KEY from the environment or ../.env in the repository root (git-ignored; never printed). Cache: data/search/live_cache.jsonl.
Naver/Kakao search APIs are out: their terms (2026-09) forbid feeding results to AI.
  python corpus/search_live.py "오늘 서울 날씨"      # prints the passages the agent would see
"""
from __future__ import annotations

import datetime
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/search/live_cache.jsonl"
UA = "Mozilla/5.0 (compatible; research-search-agent)"
MAX_PAGE = 1_000_000  # bytes read per page
_lock = threading.Lock()
_cache: dict | None = None
FAILED: list[str] = []  # queries Serper never answered (not cached, so a rerun retries them)


def today() -> str:
    d = datetime.date.today()
    return f"{d.year}년 {d.month}월 {d.day}일"


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


ENV_FILE = ROOT.parent / ".env"  # repository root


def _key() -> str:
    if not os.environ.get("SERPER_API_KEY") and ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            k, _, v = line.strip().removeprefix("export ").partition("=")
            if k == "SERPER_API_KEY":
                os.environ[k] = v.strip().strip("'\"")
    if not os.environ.get("SERPER_API_KEY"):
        raise RuntimeError("live search needs SERPER_API_KEY (env or .env in the repository root)")
    return os.environ["SERPER_API_KEY"]


def serper(q: str, n: int = 10) -> list[dict]:
    """One credit per call: Google organic results for Korea, Korean UI."""
    body = json.dumps({"q": q, "gl": "kr", "hl": "ko", "num": n}).encode()
    req = urllib.request.Request("https://google.serper.dev/search", data=body,
                                 headers={"X-API-KEY": _key(), "Content-Type": "application/json"})
    for attempt in range(4):  # a timeout once killed a 100-item run; back off and retry
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r).get("organic", [])
        except (TimeoutError, urllib.error.URLError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise  # bad key / out of credits: stop, don't burn the run
            time.sleep(2 ** attempt)
    return None  # still failing: the caller shows no results and does not cache them


def _date(s: str | None) -> str | None:
    """Serper dates look like '2024. 5. 3.' or 'May 3, 2024'; relative ones ('3일 전') stay unknown."""
    m = re.match(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", s or "")
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
    try:
        return datetime.datetime.strptime(s or "", "%b %d, %Y").date().isoformat()
    except ValueError:
        return None


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "header", "footer", "nav", "aside", "form"}

    def __init__(self):
        super().__init__()
        self.out, self.depth = [], 0

    def handle_starttag(self, tag, attrs):
        self.depth += tag in self.SKIP

    def handle_endtag(self, tag):
        self.depth -= tag in self.SKIP and self.depth > 0

    def handle_data(self, data):
        if not self.depth and data.strip():
            self.out.append(data.strip())


_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def allowed(url: str) -> bool:
    """robots.txt says we may fetch this page; an unreadable robots.txt counts as no (the snippet still stands in)."""
    host = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(url))
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            with urllib.request.urlopen(urllib.request.Request(host + "/robots.txt", headers={"User-Agent": UA}), timeout=5) as r:
                rp.parse(r.read(200_000).decode("utf-8", "replace").splitlines())
        except urllib.error.HTTPError as e:
            rp.parse([] if e.code == 404 else ["User-agent: *", "Disallow: /"])  # no robots.txt = allowed
        except Exception:
            rp = None
        with _lock:
            _robots[host] = rp
    rp = _robots[host]
    return bool(rp) and rp.can_fetch(UA, url)


def page_text(url: str) -> str:
    """Visible text of a result page robots.txt allows; '' otherwise or on any failure (the snippet still stands in)."""
    if not url.startswith(("http://", "https://")) or not allowed(url):
        return ""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=6) as r:
            if "html" not in r.headers.get("Content-Type", "html"):
                return ""
            raw = r.read(MAX_PAGE)
            enc = r.headers.get_content_charset() or "utf-8"
        p = _Text()
        p.feed(raw.decode(enc, "replace"))
        return " ".join(p.out)
    except Exception:
        return ""


def chunks(text: str, size: int = 550) -> list[str]:
    """Sentence-packed pieces of at most `size` chars (a longer sentence is cut)."""
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?。다요])\s+", text):
        s = s[:size]
        if len(cur) + len(s) + 1 > size and cur:
            out.append(cur); cur = ""
        cur = (cur + " " + s).strip()
    return out + ([cur] if cur else [])


def _bigrams(t: str) -> set:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def rank(q: str, passages: list[dict], k: int, per_doc: int = 2) -> list[dict]:
    """Query-bigram overlap, best first, at most `per_doc` pieces per page; ties keep API order.
    shortcut: lexical only, swap in the e5 encoder if ranking proves weak."""
    qb, seen, out = _bigrams(q), {}, []
    for p in sorted(passages, key=lambda p: -len(qb & _bigrams(p["title"] + p["text"]))):
        d = p.get("doc_id", p["title"])
        if seen.get(d, 0) < per_doc:
            seen[d] = seen.get(d, 0) + 1; out.append(p)
    return out[:k]


def key(q: str) -> str:
    """Cache key: spacing / punctuation / case differences hit the same entry ("AI 기본법 시행일?" == "AI기본법 시행일")."""
    pre, _, body = q.rpartition("::")
    return (pre + "::" if pre else "") + re.sub(r"[\W_]+", "", body).lower()


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = {}
        if CACHE.exists():
            for l in open(CACHE, encoding="utf-8"):
                r = json.loads(l); _cache[key(r["q"])] = r["hits"]
    return _cache


def _store(q: str, hits: list[dict]) -> None:
    with _lock:
        _cache[key(q)] = hits
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"q": q, "hits": hits}, ensure_ascii=False) + "\n")


def _cached(q: str) -> list[dict] | None:
    with _lock:
        return _load_cache().get(key(q))


WIKI = "https://ko.wikipedia.org/w/api.php?"
WIKI_UA = "ko-search-v2-eval/0.1 (research evaluation of a small Korean search model)"  # Wikimedia asks clients to name themselves


def _wiki_get(params: dict) -> dict:
    url = WIKI + urllib.parse.urlencode(params | {"format": "json", "formatversion": 2})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": WIKI_UA}), timeout=15) as r:
                return json.load(r)
        except (TimeoutError, urllib.error.URLError):
            time.sleep(2 ** attempt)
    return {}


def wiki(q: str, k: int = 8, n_pages: int = 5) -> list[dict]:
    """Free, keyless: today's Korean Wikipedia (search API + plain-text article). Stands in for the agent's local `search`."""
    if (hit := _cached("wiki::" + q)) is not None:
        return hit[:k]
    found = _wiki_get({"action": "query", "list": "search", "srsearch": q, "srlimit": n_pages}).get("query", {}).get("search", [])
    def text(title: str) -> str:
        pages = _wiki_get({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title}).get("query", {}).get("pages", [])
        return pages[0].get("extract", "") if pages else ""
    with ThreadPoolExecutor(5) as ex:
        texts = list(ex.map(lambda f: text(f["title"]), found))
    passages = []
    for f, t in zip(found, texts):
        url = "https://ko.wikipedia.org/wiki/" + urllib.parse.quote(f["title"].replace(" ", "_"))
        pieces = chunks(re.sub(r"\n+=+ [^=]+ =+\n+", "\n", t)) or [_strip(f.get("snippet", ""))]
        for i, piece in enumerate(pieces[:60]):
            pid = "wiki:" + hashlib.sha1(f"{url}#{i}".encode()).hexdigest()[:12]
            passages.append({"passage_id": pid, "doc_id": url, "title": f["title"], "text": piece, "source_type": "wiki", "date": None, "url": url})
    hits = rank(q, passages, k)
    if found:  # an empty answer may be a network failure: don't cache it
        _store("wiki::" + q, hits)
    return hits


def search(q: str, k: int = 8) -> list[dict]:
    if (hit := _cached(q)) is not None:
        return hit[:k]
    results = serper(q)
    if results is None:
        FAILED.append(q)
        return []
    with ThreadPoolExecutor(8) as ex:  # pages come in parallel: one slow site no longer stalls the search
        texts = list(ex.map(lambda it: page_text(it.get("link", "")), results))
    passages = []
    for it, page in zip(results, texts):
        url = it.get("link", "")
        title, snippet, date = _strip(it.get("title")), _strip(it.get("snippet")), _date(it.get("date"))
        pieces = chunks(page) or [snippet]
        for i, text in enumerate(pieces[:40]):
            pid = "live:" + hashlib.sha1(f"{url}#{i}".encode()).hexdigest()[:12]
            passages.append({"passage_id": pid, "doc_id": url, "title": title, "text": text, "source_type": "web", "date": date, "url": url})
    hits = rank(q, passages, k)
    _store(q, hits)
    return hits


def _selftest() -> None:
    assert _strip("<b>서울</b> 날씨 &amp; 기온") == "서울 날씨 & 기온"
    c = chunks("가" * 300 + "다. " + "나" * 300 + "다. 짧다.")
    assert all(len(x) <= 550 for x in c) and len(c) == 2, c
    ps = [{"title": "부산", "text": "부산 인구"}, {"title": "서울 날씨", "text": "오늘 서울 날씨는 맑음"}]
    assert rank("서울 날씨", ps, 1)[0]["title"] == "서울 날씨"
    same = [{"doc_id": "a", "title": "서울", "text": f"서울 날씨 {i}"} for i in range(4)] + [{"doc_id": "b", "title": "x", "text": "서울"}]
    assert [h["doc_id"] for h in rank("서울 날씨", same, 5)] == ["a", "a", "b"]
    rp = urllib.robotparser.RobotFileParser(); rp.parse(["User-agent: *", "Disallow: /private"])
    _robots["https://ex.test"] = rp
    assert allowed("https://ex.test/page") and not allowed("https://ex.test/private/x")
    p = _Text(); p.feed("<html><script>x=1</script><p>본문</p><nav>메뉴</nav></html>")
    assert p.out == ["본문"]
    assert key("AI 기본법 시행일?") == key("AI기본법 시행일") and key("wiki::A B") == "wiki::ab"
    assert _date("2024. 5. 3.") == "2024-05-03" and _date("May 3, 2024") == "2024-05-03" and _date("3일 전") is None


if __name__ == "__main__":
    _selftest()
    if len(sys.argv) > 1:
        for h in search(sys.argv[1]):
            print(f"({h['date'] or '날짜 미상'}) {h['title']}: {h['text'][:120]}")
    else:
        print("selftest ok")
