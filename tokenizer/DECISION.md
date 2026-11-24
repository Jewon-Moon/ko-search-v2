# Tokenizer v2 decision (2026-11-17)

Chosen: `outputs/tokenizer_v2/ko48k-v2-t100` (48,000 vocab, twin rate 1.0).
Code: `tokenizer/train_v2.py` (train), `tokenizer/eval_v2.py` (benchmark), `tokenizer/gguf_check.py` (llama.cpp check).
Corpus: AI-Hub-free, 131M characters (FineWeb-2 ko, MIRACL-ko wiki minus doc_id%50 held out, FineWeb ko-en mixed, StarCoder JSON, local code).
Pre-tokenizer: single digits + GPT-2 regex = llama.cpp pre-type `refact` (also EXAONE's). Quotes/brackets never glue onto words.

| tokenizer | vocab | ko_web tok/100ch | ko_wiki tok/100ch | json/code | consistent (space vs quote) |
|---|---|---|---|---|---|
| Qwen3 | 151,669 | 66.2 | 71.3 | 34.7 | 0.423 |
| ko32k (current) | 32,000 | 43.3 | 47.2 | 47.8 | 0.477 |
| v2 32k twin 0 | 32,000 | 43.5 | 46.2 | 48.1 | 0.490 |
| v2 32k twin 0.5 | 32,000 | 44.3 | 47.3 | 48.7 | 0.901 |
| v2 48k twin 0.5 | 48,000 | 42.2 | 44.9 | 46.4 | 0.918 |
| **v2 48k twin 1.0** | 48,000 | **42.8** | **45.7** | 46.7 | **0.968** |

- consistent = share of the 20k most frequent Korean word occurrences segmented identically after a space and after a quote.
- llama.cpp: vocab-only GGUF + llama-tokenize ids == HF ids on 302/302 held-out lines (converter needs the hash -> "refact" mapping, done in gguf_check.py).
- Byte-fragment tokens (partial UTF-8) cannot be 0 in byte-level BPE (they are merge intermediates): 924 vs ko32k 870. Frequency-weighted
  syllables that are one token: 0.9987 (ko32k 0.9943). Mitigation stays runtime UTF-8 validation.
- Not yet shown: that the consistency gain removes the first-syllable drop in a trained model (needs the 1.7B transplant + probe).
