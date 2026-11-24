#!/usr/bin/env python3
"""SFT downloads, 2026-11-17 (2nd list, dolly excluded: gated): MKQA, TyDi QA GoldP, KLUE-MRC train, aya_dataset, oasst2.
Writes data/sft_raw/<name>/... and data/sft_raw/SOURCES.json. Run with python -I (reads downloaded files only after this step).
"""
import json, urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/sft_raw"
HF = [("google-research-datasets/tydiqa", ["secondary_task/train-00000-of-00001.parquet", "secondary_task/validation-00000-of-00001.parquet"], "Apache-2.0"),
      ("klue/klue", ["mrc/train-00000-of-00001.parquet"], "CC BY-SA 4.0"),
      ("CohereLabs/aya_dataset", ["data/train-00000-of-00001.parquet"], "Apache-2.0"),
      ("OpenAssistant/oasst2", None, "Apache-2.0")]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    src = {}
    for repo, files, lic in HF:
        if files is None:
            from huggingface_hub import list_repo_files
            files = [f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet")]
        for f in files:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=str(RAW / repo.split("/")[1]))
        src[repo] = {"files": files, "license": lic}
    url = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
    (RAW / "mkqa").mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, RAW / "mkqa/mkqa.jsonl.gz")
    try:
        lic = urllib.request.urlopen("https://raw.githubusercontent.com/apple/ml-mkqa/main/LICENSE", timeout=20).read().decode()[:300]
    except Exception as e:
        lic = f"LICENSE fetch failed: {e}"
    src["apple/ml-mkqa"] = {"files": [url], "license_file_head": lic}
    (RAW / "SOURCES.json").write_text(json.dumps(src, ensure_ascii=False, indent=1))
    print(json.dumps(src, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
