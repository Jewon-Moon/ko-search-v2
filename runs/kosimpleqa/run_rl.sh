#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# the teacher must be loaded first: search_baseline also judges with it while running
# KoSimpleQA test 200 for the RL v2 checkpoints (user 2026-11-19), same live search + harness as V2+h in run_test.sh.
cd "$REPO/searchslm"
K="$REPO"; PY=../.venv/bin/python; LMS=~/.lmstudio/bin/lms
H="+h#nokure.noans.empty.nofix.rawweb"
echo "== start $(date)"
$LMS ps | grep -q teacher-bench || $LMS load google/gemma-4-26b-a4b-qat --identifier teacher-bench -c 32768 --parallel 8 --ttl 43200 -y
for s in step150 step200; do
  $PY -u model/search_baseline.py --live --items $K/data/kosimpleqa/test.jsonl --out $K/runs/kosimpleqa/test_rl_$s.json \
    --preds $K/runs/kosimpleqa/preds_test --systems "agent:student=hf:$K/runs/rl_v2/$s@raw$H"
done
echo "== grade $(date)"
$PY -u $K/scripts/simpleqa_grade.py $K/runs/kosimpleqa/preds_test/*rl_v2*[0-9]@raw+h*.rawweb.jsonl | tee $K/runs/kosimpleqa/test_grades_rl.txt
$LMS unload teacher-bench
echo "KOSQA_RL_DONE $(date)"
