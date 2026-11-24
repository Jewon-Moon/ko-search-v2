#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# 5-way comparison (user 2026-11-17): 4B RAG (Gemma 4 E4B), v5 0.6B, teacher (Gemma 4 26B QAT), V1 (SFT v1), V2 (SFT v2).
# Same items, same judge (teacher-bench). v6 rows for v5 / v5+h / V1 / V1+h already exist in runs/eval/preds.
cd "$REPO/searchslm"
NV=../.venv/bin/python; OV=.venv/bin/python; K="$REPO"
V2="agent:student=hf:$K/runs/sft_v2/final@raw"; V1="agent:student=hf:$K/runs/sft/final@raw"
V5H="agent:student=outputs/vast_runs/search_rl_v5/step150@curated2+h#nfweb.mhskip.noans.fstfix"
export EXT_OUT="$K/runs/eval/external"
echo "== v6 $(date)"
$NV -u model/search_baseline.py --items data/search/gen/items_v6_eval_all.jsonl --per-type 999 --out $K/runs/eval/v6eval_round2.json --preds $K/runs/eval/preds \
  --systems "$V2" "$V2+h#nfweb.mhskip.noans" agent:teacher rag:e4b
for task in korquad klue sqa; do
  echo "== $task $(date)"
  $NV -u model/search_external.py $task --n 300 --systems "$V2" "$V1" agent:teacher rag:e4b
  $OV -u model/search_external.py $task --n 300 --systems "$V5H"
done
echo "EVAL_ROUND2_DONE $(date)"
