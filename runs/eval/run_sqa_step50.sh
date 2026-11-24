#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# Ko-StrategyQA for RL v2 step50 (user 2026-11-19): does the earlier checkpoint over-abstain less than step200?
cd "$REPO/searchslm"
K="$REPO"; export EXT_OUT="$K/runs/eval/external"
../.venv/bin/python -u model/search_external.py sqa --n 300 --systems "agent:student=hf:$K/runs/rl_v2/step50@raw"
echo "SQA50_DONE $(date)"
