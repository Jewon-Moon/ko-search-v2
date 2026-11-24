#!/bin/zsh
REPO="$(cd "$(dirname "$0")/../.." && pwd)"  # repository root
# KoSimpleQA test 200 (user 2026-11-19): live search for every system (local search -> today's Korean Wikipedia, web -> Serper,
# shared cache), plus closed-book runs (the paper's setting). Judge = teacher-bench. Big models one or two at a time.
cd "$REPO/searchslm"
K="$REPO"; PY=../.venv/bin/python; LMS=~/.lmstudio/bin/lms
V2="agent:student=hf:$K/runs/sft_v2/final@raw"; H="+h#nokure.noans.empty.nofix.rawweb"
run() { $PY -u model/search_baseline.py --live --items $K/data/kosimpleqa/test.jsonl --out $K/runs/kosimpleqa/test.json --preds $K/runs/kosimpleqa/preds_test --systems "$@"; }
echo "== start $(date)"
$LMS ps | grep -q teacher-bench || $LMS load google/gemma-4-26b-a4b-qat --identifier teacher-bench -c 32768 --parallel 8 --ttl 43200 -y
run "$V2" "$V2$H" agent:teacher closed:teacher-bench
$LMS ps | grep -q qwen-agent || $LMS load qwen/qwen3.6-35b-a3b --identifier qwen-agent -c 16384 --parallel 6 --ttl 43200 -y
run agent:lm=qwen-agent closed:qwen-agent
$LMS unload qwen-agent
$LMS load google/gemma-4-e4b --identifier baseline-e4b -c 16384 --parallel 4 --ttl 43200 -y
run rag:e4b closed:baseline-e4b
$LMS unload baseline-e4b
echo "== grade $(date)"
$PY -u $K/scripts/simpleqa_grade.py $K/runs/kosimpleqa/preds_test/*.jsonl | tee $K/runs/kosimpleqa/test_grades.txt
$LMS unload teacher-bench
echo "KOSQA_TEST_DONE $(date)"
