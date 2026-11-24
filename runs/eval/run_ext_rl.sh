#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# External evals for RL v2 step200 (user 2026-11-19), same items/settings as round 2 (V2 = sft_v2/final@raw, no harness), rule-scored.
cd "$REPO/searchslm"
K="$REPO"; export EXT_OUT="$K/runs/eval/external"
for task in klue korquad sqa; do
  echo "== $task $(date)"
  ../.venv/bin/python -u model/search_external.py $task --n 300 --systems "agent:student=hf:$K/runs/rl_v2/step200@raw"
done
echo "EXT_RL_DONE $(date)"
