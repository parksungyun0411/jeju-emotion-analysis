#!/usr/bin/env bash
# 발표본 성능 목표(F1-Macro 0.8404) 재현 — 전체 파이프라인 순차 실행 + 자동 검증
# bf16 mixed precision(MPS) 가속. 사용: bash scripts/run_goal.sh [EPOCHS]
#
# 실행 순서: dual → koelectra → ensemble → 목표검증(여기서 0.84 도달 판정) → hard-mining
#   (hard-mining(step6)은 최종 앙상블에 안 쓰이는 보조 수치라 맨 마지막에 표만 채움)
set -uo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${1:-6}"
DATA="../2학기_감정분류/merged_jeju_balanced_sorted.xlsx"
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

# 여기서 목표(0.8404) 도달 여부 1차 판정
verify

# ── 보조: Step 6 Hard Mining (앙상블 미사용, 발표본 표 완성용) ──
run "step6_hard_mining" --step hard-mining --epochs "$EPOCHS" --batch-size 128

# 최종 표 (hard-mining 포함) 재검증
verify
banner "GOAL run finished"
