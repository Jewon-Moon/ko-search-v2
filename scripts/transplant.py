#!/usr/bin/env python3
"""Text-only Qwen3.5 base with tokenizer v2: drop the vision tower and MTP head, rebuild the tied embedding for the 48k vocab.

Row init (the ko32k winner, searchslm/outputs/EMBEDDING_RECOVERY_REPORT.md): a v2 token string that exists verbatim in
the base vocab copies that row ("exact"); otherwise the base BPE re-segments the byte-level string and the rows are averaged
("mean"). Both tokenizers are byte-level BPE over the same 256-symbol alphabet. Special tokens map by content; unknown -> mean row.
  python scripts/transplant.py --base models/Qwen3.5-2B-Base --tokenizer tokenizer/ko48k-v2-t100 --out models/q35-2b-ko48k-init
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

ROOT = Path(__file__).resolve().parents[1]


def load_text_model(base: Path, dtype=torch.bfloat16) -> Qwen3_5ForCausalLM:
    cfg = json.loads((base / "config.json").read_text())["text_config"]
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg)).to(dtype)
    sd = {}
    for f in sorted(base.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k.startswith("model.language_model."):
                    sd["model." + k[len("model.language_model."):]] = h.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]  # tied to embed_tokens
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.tie_weights()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="models/Qwen3.5-2B-Base")
    ap.add_argument("--tokenizer", default="tokenizer/ko48k-v2-t100")
    ap.add_argument("--out", default="models/q35-2b-ko48k-init")
    a = ap.parse_args()
    base, out = ROOT / a.base, ROOT / a.out
    old_tok = AutoTokenizer.from_pretrained(base)
    new_tok = AutoTokenizer.from_pretrained(ROOT / a.tokenizer)
    model = load_text_model(base)
    old_emb = model.get_input_embeddings().weight.data.float()
    old_vocab, bpe = old_tok.get_vocab(), old_tok.backend_tokenizer.model
    mean_row = old_emb.mean(0)
    added = {t.content for t in new_tok.added_tokens_decoder.values()}
    n_new = len(new_tok)
    new_emb = torch.empty(n_new, old_emb.shape[1])
    how = Counter()
    for s, i in new_tok.get_vocab().items():
        if s in old_vocab:
            new_emb[i] = old_emb[old_vocab[s]]; how["exact" if s not in added else "special"] += 1
        elif s in added:
            new_emb[i] = mean_row; how["special_unmatched"] += 1
        else:
            ids = [old_vocab[t.value] for t in bpe.tokenize(s)]
            new_emb[i] = old_emb[ids].mean(0); how["mean"] += 1
    assert sum(how.values()) == n_new, (how, n_new)
    model.resize_token_embeddings(n_new, mean_resizing=False)
    model.get_input_embeddings().weight.data.copy_(new_emb.to(model.dtype))
    model.tie_weights()
    c = model.config
    c.vocab_size = n_new
    c.mtp_num_hidden_layers = 0  # MTP head dropped; llama.cpp otherwise expects an extra nextn block
    c.eos_token_id = new_tok.convert_tokens_to_ids("<|endoftext|>")
    c.bos_token_id = None
    c.pad_token_id = new_tok.pad_token_id if new_tok.pad_token_id is not None else c.eos_token_id
    model.save_pretrained(out, safe_serialization=True)
    new_tok.save_pretrained(out)
    params = sum(p.numel() for p in model.parameters())
    rep = {"base": a.base, "tokenizer": a.tokenizer, "rows": dict(how), "vocab": n_new, "parameters": params,
           "embedding_parameters": new_emb.numel(), "dropped": ["model.visual.*", "mtp.*"]}
    (out / "transplant_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
