#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# Live (Serper) eval, dev set, bare V2: failure analysis for the new harness (user 2026-11-19). Judge = teacher-bench.
cd "$REPO/searchslm"
K="$REPO"
../.venv/bin/python -u model/search_baseline.py --live --items $K/data/live_eval/dev.jsonl \
  --out $K/runs/live_eval/dev_v2.json --preds $K/runs/live_eval/preds_dev --systems "agent:student=hf:$K/runs/sft_v2/final@raw"
echo "LIVE_DEV_DONE $(date)"
