#!/usr/bin/env bash
# 제주어↔표준어 KoBART 번역 학습 러너 (양방향 단일 모델, 방향 prefix).
# 사용: bash scripts/run_translation.sh [EPOCHS]
#
# ⚠️ 단일 MPS GPU 경합 방지: 감정 goal run(scripts/run_goal.sh)과 동시에 실행하지 말 것.
#    감정 학습이 끝난 뒤에 실행한다. (마스터 설계 "7. 작업 순서 / GPU 직렬화")
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-5}"
DATA="../2학기_감정분류/merged_jeju_balanced_sorted.xlsx"
OUT="results_translation"
LOG="$OUT/translation_run.log"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$OUT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo "" | tee -a "$LOG"; echo "########## $(ts)  $* ##########" | tee -a "$LOG"; }

banner "START translation run (epochs=$EPOCHS, direction=both, bf16)"

if python3 src/translation/train_translation.py \
      --direction both --epochs "$EPOCHS" \
      --fp16 --data "$DATA" --output-dir "$OUT" 2>&1 \
      | grep -vE "NotOpenSSLWarning|warnings.warn" | tee -a "$LOG"; then
  banner "END translation run (ok)"
else
  banner "END translation run (FAILED rc=${PIPESTATUS[0]})"
fi

# 학습 종료 후 report 출력
REPORT="$OUT/translation_report.txt"
if [ -f "$REPORT" ]; then
  banner "translation_report.txt"
  cat "$REPORT" | tee -a "$LOG"
else
  banner "report 없음 ($REPORT) — 학습 실패 가능"
fi

# 목표 자동 판정: overall chrF >= 80 (verify_translation.py, exit 0/1)
banner "verify_translation (목표 chrF>=80)"
python3 src/translation/verify_translation.py \
    --results-dir "$OUT" --target-chrf 80 --tol 1.0 2>&1 | tee -a "$LOG"
