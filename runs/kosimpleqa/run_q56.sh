#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# KoSimpleQA test 200 for Q5_K_M and Q6_K GGUFs of RL step200 (user 2026-11-20), via LM Studio; same setup as run_q4.sh / run_q8.sh.
cd "$REPO/searchslm"
K="$REPO"; PY=../.venv/bin/python; LMS=~/.lmstudio/bin/lms
echo "== start $(date)"
$LMS ps | grep -q teacher-bench || $LMS load google/gemma-4-26b-a4b-qat --identifier teacher-bench -c 32768 --parallel 8 --ttl 43200 -y
for v in q5_k_m q6_k; do
  $LMS load ko-search-v2-rl200@$v --identifier rl200-$v -c 8192 --parallel 1 -y
  $PY -u model/search_baseline.py --live --items $K/data/kosimpleqa/test.jsonl --out $K/runs/kosimpleqa/test_$v.json \
    --preds $K/runs/kosimpleqa/preds_test --systems "agent:student=lms:rl200-$v@raw+h#nokure.noans.empty.nofix.rawweb"
  $LMS unload rl200-$v
done
echo "== grade $(date)"
$PY -u $K/scripts/simpleqa_grade.py $K/runs/kosimpleqa/preds_test/agent_student_lms_rl200-q[56]_k*@raw+h*.rawweb.jsonl | tee $K/runs/kosimpleqa/test_grades_q56.txt
$LMS unload teacher-bench
echo "KOSQA_Q56_DONE $(date)"
