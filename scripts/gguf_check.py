#!/usr/bin/env python3
"""llama.cpp compatibility of a v2 tokenizer: vocab-only GGUF + llama-tokenize ids == HF ids on held-out lines.

The converter identifies the pre-tokenizer by hashing a probe encoding; a new vocab has an unknown hash, so the hash is mapped to
"refact" (digits + GPT-2 regex, the pre-tokenizer v2 was trained with) here instead of editing the llama.cpp submodule.
  python tokenizer/gguf_check.py outputs/tokenizer_v2/ko48k-v2-t100 --llama-tokenize <path to llama-tokenize>
"""
from __future__ import annotations

import argparse
import json
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT.parent / "slm-runtime/external/llama.cpp"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokenizer")
    ap.add_argument("--llama-tokenize", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ROOT / a.tokenizer)
    work = Path(tempfile.mkdtemp(prefix="gguf-v2-"))
    for f in (ROOT / a.tokenizer).iterdir():
        shutil.copy(f, work / f.name)
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B").to_dict()
    cfg.update(vocab_size=len(tok))
    (work / "config.json").write_text(json.dumps(cfg))
    sys.path.insert(0, str(LLAMA))
    import conversion.base as B
    orig = B.TextModel.get_vocab_base_pre

    def pre(self, tokenizer):
        try:
            return orig(self, tokenizer)
        except NotImplementedError:
            return "refact"  # v2 = Digits(individual) + GPT-2 regex, llama.cpp's refact/starcoder pre-type
    B.TextModel.get_vocab_base_pre = pre
    out = work / "vocab.gguf"
    sys.argv = ["convert_hf_to_gguf.py", str(work), "--vocab-only", "--outfile", str(out)]
    runpy.run_path(str(LLAMA / "convert_hf_to_gguf.py"), run_name="__main__")
    sys.path.insert(0, str(Path(__file__).parent))
    from eval_v2 import eval_sets
    sets = eval_sets()
    lines = [x for k in ("ko_web", "ko_wiki", "json_code") for x in sets[k][: a.n // 3]]
    lines += ['{"think": "검색합니다.", "action": "search", "query": "이순신 생년"}', "질문: 2026년 11월 17일 서울 날씨?"]
    bad = []
    for x in lines:
        x = x[:2000]
        r = subprocess.run([a.llama_tokenize, "-m", str(out), "-p", x, "--ids", "--log-disable", "--no-bos", "--no-escape"], capture_output=True, text=True)
        got = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
        want = tok(x, add_special_tokens=False)["input_ids"]
        if got != want:
            bad.append({"text": x[:80], "hf": want[:12], "llama": (got or [])[:12], "err": r.stderr[-200:] if r.returncode else ""})
    res = {"tokenizer": a.tokenizer, "lines": len(lines), "mismatches": len(bad), "examples": bad[:5], "gguf": str(out)}
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
