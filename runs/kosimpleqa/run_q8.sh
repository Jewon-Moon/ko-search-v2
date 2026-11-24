#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# KoSimpleQA test 200 with the deployment q8_0 GGUF of RL step200 (user 2026-11-20), via LM Studio; same live search + harness as run_rl.sh.
cd "$REPO/searchslm"
K="$REPO"; PY=../.venv/bin/python; LMS=~/.lmstudio/bin/lms
echo "== start $(date)"
$LMS ps | grep -q rl200-q8 || $LMS load ko-search-v2-rl200@q8_0 --identifier rl200-q8 -c 8192 --parallel 1 -y
$LMS ps | grep -q teacher-bench || $LMS load google/gemma-4-26b-a4b-qat --identifier teacher-bench -c 32768 --parallel 8 --ttl 43200 -y
$PY -u model/search_baseline.py --live --items $K/data/kosimpleqa/test.jsonl --out $K/runs/kosimpleqa/test_q8.json \
  --preds $K/runs/kosimpleqa/preds_test --systems "agent:student=lms:rl200-q8@raw+h#nokure.noans.empty.nofix.rawweb"
echo "== grade $(date)"
$PY -u $K/scripts/simpleqa_grade.py $K/runs/kosimpleqa/preds_test/agent_student_lms_rl200-q8@raw+h*.rawweb.jsonl | tee $K/runs/kosimpleqa/test_grades_q8.txt
$LMS unload teacher-bench; $LMS unload rl200-q8
echo "KOSQA_Q8_DONE $(date)"
