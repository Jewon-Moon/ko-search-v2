#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# KoSimpleQA dev 50, bare V2, live search (local search -> today's Korean Wikipedia, web_search -> Serper). Harness design input.
cd "$REPO/searchslm"
K="$REPO"
../.venv/bin/python -u model/search_baseline.py --live --items $K/data/kosimpleqa/dev.jsonl \
  --out $K/runs/kosimpleqa/dev.json --preds $K/runs/kosimpleqa/preds_dev --systems "agent:student=hf:$K/runs/sft_v2/final@raw"
../.venv/bin/python -u $K/scripts/simpleqa_grade.py $K/runs/kosimpleqa/preds_dev/*.jsonl
echo "KOSQA_DEV_DONE $(date)"
