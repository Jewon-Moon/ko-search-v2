#!/bin/bash
# ko-search-v2 continued pretraining on one Vast GPU (H100 80GB or A100 80GB), run from /workspace/ko-search-v2.
#   bash vast/run.sh setup    # deps + base download + transplant (~10 min)
#   bash vast/run.sh trial    # 20 stage-1 steps: prints tokens/s -> projected hours and cost (~5 min)
#   bash vast/run.sh stage1   # embedding-only, 10M tokens
#   bash vast/run.sh stage2   # everything, all train blocks (~100M tokens)
#   bash vast/run.sh pack     # tar the final checkpoints for download
#   bash vast/run.sh passk    # RL readiness: pass@8 of V2 on 100 RL items (~20 min); bash vast/run.sh rl  # GRPO, STEPS=300 WRONG=-0.5
#   bash vast/run.sh sft      # full SFT from runs/stage2/final on data/sft (uploaded), 2 epochs, tars runs/sft/final
# Logs: logs/<step>.log ; each step writes logs/<step>.done when it finishes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runs models
step="${1:?setup|trial|stage1|stage2|pack|sft|passk|rl}"
PY="${PY:-python}"  # Vast pytorch images: PY=/venv/main/bin/python
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"  # 64-core hosts: unbounded torch CPU threads made the 30-second transplant crawl for 20+ min
MICRO="${MICRO:-2}"  # blocks per forward pass: 2 fits a 40GB A100 (stage 2 = ~24GB weights+Adam); 8 for 80GB cards. Step size stays 64k tokens.
case "$step" in
  setup)
    pip install -q -r vast/requirements.txt 2>&1 | tail -3
    pip install -q flash-linear-attention 2>&1 | tail -2 || echo "WARN: flash-linear-attention failed; training falls back to slow torch kernels"
    pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -2 || echo "WARN: causal-conv1d failed (optional)"
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B-Base', local_dir='models/Qwen3.5-2B-Base')"
    $PY scripts/transplant.py --base models/Qwen3.5-2B-Base --tokenizer tokenizer/ko48k-v2-t100 --out models/q35-2b-ko48k-init 2>&1 | tail -1 | tee logs/setup.log
    $PY -c "import fla; print('fla ok')" 2>&1 | tee -a logs/setup.log || true
    touch logs/setup.done ;;
  trial)
    $PY -u scripts/train_cpt.py --stage 1 --init models/q35-2b-ko48k-init --out runs/trial --tokens 10e6 --lr 2e-4 --max-steps 20 --eval-every 1000 --micro $MICRO 2>&1 | tee logs/trial.log
    # stage 2 trains every weight: the memory peak. 3 steps catch an OOM before the real run.
    $PY -u scripts/train_cpt.py --stage 2 --init models/q35-2b-ko48k-init --out runs/trial2 --tokens 10e6 --lr 2e-5 --emb-lr 5e-5 --max-steps 3 --eval-every 1000 --micro $MICRO 2>&1 | tee logs/trial2.log || true  # an OOM must still reach the summary
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a logs/trial2.log
    $PY - <<'EOF'
import json, re, numpy as np
rows = [json.loads(l) for l in open("logs/trial.log") if l.startswith('{"step"')]
tps = rows[-1]["tok_s"]
r2 = [json.loads(l) for l in open("logs/trial2.log") if l.startswith('{"step"')]
tps2 = r2[-1]["tok_s"] if r2 else None; n = np.load("data/cpt/train.npy", mmap_mode="r").size
print(json.dumps({"trial_tok_s_stage1": tps, "train_tokens": int(n),
                  "est_hours_stage1_10M": round(10e6 / tps / 3600, 2),
                  "trial_tok_s_stage2_3steps": tps2,
                  "est_hours_stage2_all": round(n / tps2 / 3600, 2) if tps2 else "stage-2 trial failed (OOM?) - see logs/trial2.log"}))
EOF
    rm -rf runs/trial runs/trial2; touch logs/trial.done ;;
  stage1)
    $PY -u scripts/train_cpt.py --stage 1 --init models/q35-2b-ko48k-init --out runs/stage1 --tokens 10e6 --lr 2e-4 --eval-every 50 --micro $MICRO 2>&1 | tee logs/stage1.log
    touch logs/stage1.done ;;
  stage2)
    TOK=$($PY -c "import numpy as np; print(int(np.load('data/cpt/train.npy', mmap_mode='r').size * 0.99))")
    $PY -u scripts/train_cpt.py --stage 2 --init runs/stage1/final --out runs/stage2 --tokens "$TOK" --lr 2e-5 --emb-lr 5e-5 \
      --eval-every 100 --save-every 500 --micro $MICRO 2>&1 | tee logs/stage2.log
    touch logs/stage2.done ;;
  sft)
    # needs data/sft/{train,val}.jsonl uploaded from the Mac (scripts/assemble_sft.py)
    $PY -u scripts/train_sft.py --init runs/stage2/final --data data/sft --out runs/sft --epochs 2 --lr 1e-5 2>&1 | tee logs/sft.log
    tar -cf ko-search-v2-sft.tar runs/sft/final runs/sft/history.json runs/sft/config.json logs/sft.log
    touch logs/sft.done ;;
  pack)
    tar -cf ko-search-v2-cpt.tar runs/stage1/final runs/stage1/history.json runs/stage1/config.json \
      runs/stage2/final runs/stage2/history.json runs/stage2/config.json logs
    ls -lh ko-search-v2-cpt.tar ;;
  passk|rl)  # GRPO on V2 (searchslm/model/search_rl_v2.py); needs searchslm/ with corpus/, model/,
    # data/search/{index,models,corpus_v2} and runs/sft_v2/final + data/rl_v2 here (uploaded). Local index only: no Serper calls.
    cd searchslm
    if [ "$step" = passk ]; then  # 100 train items x 8 samples: is there signal for RL? (pass@8, groups with mixed rewards)
      $PY -u model/search_rl_v2.py --passk 100 --g 8 2>&1 | tee ../logs/passk.log
    else
      $PY -u model/search_rl_v2.py --steps "${STEPS:-300}" --wrong "${WRONG:--0.5}" 2>&1 | tee ../logs/rl.log
      tar -cf ../rl_v2.tar -C .. runs/rl_v2 logs/rl.log
    fi
    touch ../logs/$step.done ;;
  *) echo "unknown step $step"; exit 1 ;;
esac
