# ko-search-slm — 저사양 PC용 한국어 검색 특화 소형 언어모델

모델 안에 지식을 모두 담는 대신 **검색하는 법**을 학습한 한국어 소형 언어모델입니다. 모델은 매 턴 JSON 행동 하나(`search` / `web_search` / `answer`)를 출력하고, 규칙 하네스가 작은 모델의 약점을 보완합니다. 한양대학교 ERICA 2026 하반기 SW·AI융합연구개발 일반과제 결과물입니다.

이 저장소는 **코드만** 담고 있습니다. 학습·평가 데이터, 검색 결과 캐시, 문항별 예측, 모델 가중치는 포함하지 않습니다.

## 결과 요약

최종 모델: Qwen3.5-2B-Base + 한국어 토크나이저 v2(48,000 어휘) → 1.47B. 추가 사전학습 1억 4,850만 token → 지도학습 → GRPO 강화학습(200 step) + 규칙 하네스.

**KoSimpleQA 200문항, 실제 웹 검색(Serper + 한국어 위키백과), SimpleQA F**

| 시스템 | F |
|---|---|
| 1.47B 모델만 | 69.6 |
| + 하네스 | 76.6 |
| + 강화학습 (최종, bf16) | **78.8** |
| 교사 에이전트 (Gemma 4 26B-A4B) | 78.1 |

최종과 하네스만의 차이는 +2.1입니다(짝 비교 부트스트랩 95% 신뢰구간 −3.2~+7.3, 200문항으로는 유의하지 않음).

**양자화와 저사양 CPU** (llama.cpp GGUF)

| 양자화 | 크기 | F |
|---|---|---|
| Q4_K_M | 884MB | 73.4 |
| Q5_K_M | 1.0GB | 75.9 |
| **Q6_K** | 1.15GB | **78.1** |
| Q8_0 | 1.57GB | 77.8 |

Intel N100(4코어, 8GB RAM, CPU만)에서 Q6_K는 생성 초당 11.2 token, 프롬프트 처리 초당 23 token입니다. 질문당 응답 시간은 중앙값 약 2분 50초이고, 대부분 검색 결과를 읽는 시간입니다.

**알려진 한계**
- 강화학습 후 예/아니오 추론 질문(Ko-StrategyQA)에서 지나치게 "찾을 수 없습니다"로 답합니다.
- 저사양 CPU에서는 응답이 느립니다.

## 구성

```
tokenizer/             토크나이저 v2 (ko48k-v2-t100)와 선택 근거
scripts/               토크나이저 이식(transplant), 사전학습·SFT 데이터 구축과 학습, GGUF 변환, 채점
vast/                  클라우드 GPU 실행 스크립트 (run.sh: cpt / sft / passk / rl)
runs/**/*.sh           평가 실행 스크립트 (KoSimpleQA, 외부 벤치마크, 양자화 비교)
searchslm/             에이전트 공통 코드 (0.6B 1차 모델부터 사용)
  corpus/search_teacher.py   에이전트 루프, 규칙 하네스, 교사 궤적 생성
  corpus/search_live.py      실시간 검색기 (Serper 웹 검색 + 한국어 위키백과) + 결과 캐시
  corpus/search_index.py     로컬 검색기 (BM25 + e5, KURE 재정렬)
  model/search_baseline.py   평가 실행기 (HF 체크포인트 또는 LM Studio GGUF)
  model/search_external.py   공개 벤치마크 평가 (KorQuAD, KLUE-MRC, Ko-StrategyQA)
  model/search_rl_v2.py      GRPO 강화학습 (정답 +1 / 찾을 수 없음 0 / 오답 −0.5)
  prompts/agent_system.txt   에이전트 시스템 프롬프트
```

## 실행 환경

- Python 3.11+, PyTorch, Transformers 5.x (`vast/requirements.txt`)
- 교사 모델과 채점: OpenAI 호환 로컬 서버(LM Studio)에 Gemma 4 26B-A4B를 `teacher-bench`라는 이름으로 올려 사용
- 실시간 검색: 환경 변수 `SERPER_API_KEY` (또는 저장소 루트의 `.env`, git에 올리지 않음)
- 배포: llama.cpp (GGUF 변환은 `scripts/to_gguf.py`, Qwen3.5의 MTP 층 제외)

평가 예시 (`searchslm/`에서 실행, 최종 하네스 설정):

```bash
python model/search_baseline.py --live --items <items.jsonl> --out <out.json> --preds <pred_dir> \
  --systems "agent:student=hf:<checkpoint>@raw+h#nokure.noans.empty.nofix.rawweb"
```

데이터 경로(`data/...`)는 원래 연구 환경 기준이라, 데이터를 직접 받아 같은 위치에 두어야 그대로 돌아갑니다. 스크립트 상단 docstring에 각 데이터의 출처를 적어 두었습니다.

## 데이터 출처 (저장소에 미포함)

사전학습: FineWeb-2 한국어, FinePDFs 한국어, 한국어 위키백과. 지도학습·강화학습: MIRACL-ko, KLUE-MRC, KorQuAD 1.0, TyDi QA, MKQA, 교사 모델 생성 궤적. 평가: KoSimpleQA, KorQuAD, KLUE-MRC, Ko-StrategyQA. 각 데이터는 원 출처의 라이선스를 따릅니다. 1.47B 모델 단계에서는 AI Hub 데이터를 사용하지 않았습니다. 평가 문항과 겹치는 학습 데이터는 n-gram 기준으로 제거했습니다.

## 라이선스

코드는 Apache License 2.0 (`LICENSE`)입니다. 기반 모델(Qwen3.5-2B), 교사 모델(Gemma 4), 검색 모델(e5, KURE), llama.cpp는 이 저장소에 포함되지 않으며, 각 프로젝트의 라이선스를 따릅니다.

관련 저장소: 토크나이저·기본 모델 단계 코드는 [korean-slm-lab](https://github.com/Jewon-Moon/korean-slm-lab).
