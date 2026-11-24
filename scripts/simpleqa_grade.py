#!/usr/bin/env python3
"""SimpleQA-style grading of prediction rows (search_baseline.py --preds): each answer is CORRECT / INCORRECT / NOT_ATTEMPTED,
judged by the teacher (LM Studio "teacher-bench"), then CO / IN / NA %, CGA = CO/(CO+IN), F = harmonic mean of CO and CGA
(the KoSimpleQA / SimpleQA metrics). Writes <preds>.graded.jsonl next to each file and prints one table row per system.
  python scripts/simpleqa_grade.py runs/kosimpleqa/preds_dev/*.jsonl
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "searchslm/corpus"))
import search_teacher as T  # noqa: E402

GRADER = """질문, 정답, 예측 답변을 보고 예측을 셋 중 하나로 판정하세요. 글자 하나만 출력합니다.
A (정답): 예측이 정답의 핵심 내용을 담고 있고 정답과 모순되는 내용이 없다. 표현·어순·띄어쓰기 차이, 정답을 포함한 부연 설명, 약간의 망설임 표현은 괜찮다.
B (오답): 예측이 정답과 다른 답을 내놓았거나 정답과 모순되는 내용이 있다. 여러 후보를 나열해 하나로 정하지 않은 경우도 오답이다.
C (시도 안 함): 예측이 답을 내놓지 않았다 ("찾을 수 없습니다", "확인할 수 없습니다", 빈 답 등). 정답과 모순되는 내용도 없다.
숫자는 정답의 단위·자릿수 수준까지 맞아야 정답이다."""
LABEL = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def grade(r: dict) -> str:
    if not r["pred"].strip():
        return "NOT_ATTEMPTED"
    out = T.chat([{"role": "system", "content": GRADER},
                  {"role": "user", "content": f"질문: {r['question']}\n정답: {r['gold']}\n예측: {r['pred']}"}], 4, 0)
    m = re.search(r"[ABC]", re.sub(r"<think>.*?</think>", "", out, flags=re.S))
    return LABEL[m.group(0)] if m else "INCORRECT"


def metrics(labels: list[str]) -> dict:
    n = len(labels)
    co, inc, na = (labels.count(x) / n for x in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED"))
    cga = co / (co + inc) if co + inc else 0.0
    f = 2 * co * cga / (co + cga) if co + cga else 0.0
    return {"n": n, "CO": round(100 * co, 1), "IN": round(100 * inc, 1), "NA": round(100 * na, 1), "CGA": round(100 * cga, 1), "F": round(100 * f, 1)}


def main() -> None:
    assert metrics(["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "CORRECT"]) == {"n": 4, "CO": 50.0, "IN": 25.0, "NA": 25.0, "CGA": 66.7, "F": 57.1}
    for p in map(Path, sys.argv[1:]):
        if p.name.endswith(".graded.jsonl"):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        with ThreadPoolExecutor(8) as ex:
            labels = list(ex.map(grade, rows))
        with open(p.with_suffix(".graded.jsonl"), "w", encoding="utf-8") as f:
            for r, lab in zip(rows, labels):
                f.write(json.dumps(r | {"sqa_label": lab}, ensure_ascii=False) + "\n")
        print(p.stem[-60:], json.dumps(metrics(labels)))


if __name__ == "__main__":
    main()
