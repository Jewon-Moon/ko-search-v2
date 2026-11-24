#!/usr/bin/env python3
"""Convert a tokenizer-v2 HF checkpoint to GGUF with the submodule's llama.cpp converter.

The converter recognises a pre-tokenizer by hashing a probe encoding; v2's hash is new, so it is mapped to "refact"
(digits + GPT-2 regex, what v2 was trained with; tokenizer/gguf ids == HF ids was checked on 302 lines) instead of editing llama.cpp.
  python scripts/to_gguf.py models/q35-2b-ko48k-init --outfile models/q35-2b-ko48k-init-q8_0.gguf --outtype q8_0 --no-mtp
--no-mtp is required for our Qwen3.5 checkpoints: transplant.py drops the MTP head, and the converter otherwise expects it.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

LLAMA = Path(__file__).resolve().parents[2] / "slm-runtime/external/llama.cpp"


def main() -> None:
    sys.path[:0] = [str(LLAMA), str(LLAMA / "gguf-py")]
    import conversion.base as B
    orig = B.TextModel.get_vocab_base_pre

    def pre(self, tokenizer):
        try:
            return orig(self, tokenizer)
        except NotImplementedError:
            return "refact"
    B.TextModel.get_vocab_base_pre = pre
    sys.argv = ["convert_hf_to_gguf.py", *sys.argv[1:]]
    runpy.run_path(str(LLAMA / "convert_hf_to_gguf.py"), run_name="__main__")


if __name__ == "__main__":
    main()
