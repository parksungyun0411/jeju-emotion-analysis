#!/usr/bin/env bash
# 재현 코드 전체 파이프라인 순차 실행 + 발표 자료 수치와 비교
# bf16 mixed precision(MPS) 가속. 사용: DATA=<xlsx 경로> bash scripts/run_goal.sh [EPOCHS]
#
# 실행 순서: dual → koelectra → ensemble → 발표 수치와 비교 → hard-mining
#   (hard-mining(step6)은 최종 앙상블에 안 쓰이는 보조 수치라 맨 마지막에 표만 채움)
# 분할은 행 단위 7:1:2라 test 행의 70.4%가 train에도 있는 문장이다. 여기서 나오는 점수는
# 일반화 성능이 아니다(README 참고).
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-6}"
DATA="${DATA:-merged_jeju_balanced_sorted.xlsx}"
OUT="results_kr_bert"
LOG="$OUT/goal_run.log"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$OUT"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
banner() { echo "" | tee -a "$LOG"; echo "########## $(ts)  $* ##########" | tee -a "$LOG"; }

banner "START goal run (epochs=$EPOCHS, bf16)"

run() {  # run <label> <args...>
  local label="$1"; shift
  banner "BEGIN $label"
  if python3 src/jeju_kr_bert.py "$@" --fp16 --data "$DATA" --output-dir "$OUT" 2>&1 \
        | grep -vE "NotOpenSSLWarning|warnings.warn" | tee -a "$LOG"; then
    banner "END $label (ok)"
  else
    banner "END $label (FAILED rc=${PIPESTATUS[0]})"
  fi
}

verify() {
  banner "VERIFY against presentation targets"
  python3 src/verify_goal.py --results-dir "$OUT" 2>&1 \
      | grep -vE "NotOpenSSLWarning|warnings.warn" | tee -a "$LOG"
  banner "verify exit=${PIPESTATUS[0]}"
}

# ── 크리티컬 패스 (최종 앙상블에 들어가는 두 브랜치) ──
run "step3_dual"        --step dual        --epochs "$EPOCHS" --batch-size 64
run "branch1_koelectra" --step koelectra   --epochs "$EPOCHS" --batch-size 128

# Step 7: 앙상블 (dual + koelectra test probs 결합, α 그리드서치)
banner "BEGIN step7_ensemble"
python3 src/jeju_kr_bert.py --step ensemble \
  --dual-probs   "$OUT/step3_dual_probs.npz" \
  --single-probs "$OUT/branch1_koelectra_probs.npz" \
  --output-dir "$OUT" 2>&1 | grep -vE "NotOpenSSLWarning|warnings.warn" | tee -a "$LOG"
banner "END step7_ensemble"

# 발표 자료의 앙상블 수치(0.8404)와 1차 비교
verify

# ── 보조: Step 6 Hard Mining (앙상블 미사용, 발표본 표 완성용) ──
run "step6_hard_mining" --step hard-mining --epochs "$EPOCHS" --batch-size 128

# 최종 표 (hard-mining 포함) 재검증
verify
banner "GOAL run finished"
