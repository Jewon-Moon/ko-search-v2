#!/usr/bin/env python3
"""Fetch the continued-pretraining text (2026-11-17 list): random row groups of one FineWeb-2 / FinePDFs Korean shard
(not the shard head, so the sample is spread over the file) and the full Korean Wikipedia 20231101 dump.

Writes data/raw/<source>.jsonl ({"text", "url"}) and data/raw/SOURCES.json (repo, file, license, rows, chars).
  python -I scripts/fetch_cpt.py --fineweb-chars 230e6 --finepdfs-chars 70e6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
SHARDS = {
    "fineweb2": ("HuggingFaceFW/fineweb-2", "data/kor_Hang/train/000_00000.parquet", "ODC-By 1.0 (+ Common Crawl ToU)"),
    "finepdfs": ("HuggingFaceFW/finepdfs", "data/kor_Hang/train/000_00000.parquet", "ODC-By 1.0 (+ Common Crawl ToU)"),
}
WIKI = ("wikimedia/wikipedia", ["20231101.ko/train-0000%d-of-00003.parquet" % i for i in range(3)], "CC BY-SA 3.0/4.0 + GFDL")


def sample_shard(name: str, max_chars: float, seed: int) -> dict:
    repo, path, lic = SHARDS[name]
    f = pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}/{path}"))
    groups = list(range(f.metadata.num_row_groups)); random.Random(seed).shuffle(groups)
    cols = ["text", "url"] + (["ocr_quality_scores", "is_truncated"] if name == "finepdfs" else [])
    rows = chars = 0
    with open(RAW / f"{name}.jsonl", "w", encoding="utf-8") as out:
        for g in groups:
            for r in f.read_row_group(g, columns=cols).to_pylist():
                out.write(json.dumps({k: r[k] for k in cols if k in r}, ensure_ascii=False) + "\n")
                rows += 1; chars += len(r["text"])
            print(f"[{name}] groups read {groups.index(g) + 1}, rows {rows}, chars {chars / 1e6:.0f}M", flush=True)
            if chars >= max_chars:
                break
    return {"repo": repo, "file": path, "license": lic, "row_groups_read": groups.index(g) + 1, "rows": rows, "chars": chars}


def fetch_wiki() -> dict:
    repo, files, lic = WIKI
    rows = chars = 0
    with open(RAW / "wiki_ko.jsonl", "w", encoding="utf-8") as out:
        for fn in files:
            p = hf_hub_download(repo, fn, repo_type="dataset", local_dir=str(RAW / "wiki_parquet"))
            for r in pq.read_table(p, columns=["title", "text", "url"]).to_pylist():
                out.write(json.dumps({"text": r["title"] + "\n" + r["text"], "url": r["url"]}, ensure_ascii=False) + "\n")
                rows += 1; chars += len(r["text"])
    return {"repo": repo, "files": files, "license": lic, "rows": rows, "chars": chars}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fineweb-chars", type=float, default=230e6)
    ap.add_argument("--finepdfs-chars", type=float, default=70e6)
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    srcs = {"fineweb2": sample_shard("fineweb2", a.fineweb_chars, 1), "finepdfs": sample_shard("finepdfs", a.finepdfs_chars, 2),
            "wiki_ko": fetch_wiki()}
    (RAW / "SOURCES.json").write_text(json.dumps(srcs, ensure_ascii=False, indent=1))
    print(json.dumps(srcs, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
