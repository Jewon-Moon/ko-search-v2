#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# KLUE for the new-venv systems (first attempt failed: pandas missing); runs after round 2 so result files are not written concurrently
cd "$REPO"
until grep -q EVAL_ROUND2_DONE runs/eval/round2.log; do sleep 60; done
cd "$REPO/searchslm"; K="$REPO"
export EXT_OUT="$K/runs/eval/external"
echo "== klue redo $(date)"
../.venv/bin/python -u model/search_external.py klue --n 300 --systems "agent:student=hf:$K/runs/sft_v2/final@raw" "agent:student=hf:$K/runs/sft/final@raw" agent:teacher rag:e4b
echo "KLUE_REDO_DONE $(date)"
