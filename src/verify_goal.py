"""
발표본 성능 목표 자동 검증
─────────────────────────
results 디렉토리의 각 단계 *_report.txt 에서 Test F1-Macro 를 읽어
최종발표본(최종발표_제주어감정분석.pptx) 단계별 목표치와 비교한다.

판정 기준:
  - 각 단계: 발표본 F1 - TOL 이상이면 PASS (균형 데이터라 초기 단계는 보통 초과)
  - 전체 GOAL: 최종 앙상블(step7)이 발표본 0.8404 - TOL 이상이면 달성

사용:
  python src/verify_goal.py --results-dir results_kr_bert
  (exit code 0 = GOAL 달성, 1 = 미달/결과 누락)
"""
import re
import sys
import argparse
from pathlib import Path

# 발표본 슬라이드 단계별 Test F1-Macro 목표
PRESENTATION_TARGETS = [
    ("step1_baseline",     "Step 1 KR-BERT 베이스라인",      0.4733),
    ("step2_balanced",     "Step 2 데이터 밸런싱",            0.5833),
    ("step3_dual",         "Step 3 Dual-Gated KR-BERT",      0.6158),
    ("step4_tokenizer",    "Step 4 토크나이저 최적화",         0.7184),
    ("step5_dapt",         "Step 5 DAPT",                    0.7456),
    ("step6_hard_mining",  "Step 6 Hard Mining",             0.8003),
    ("branch1_koelectra",  "Branch1 KoELECTRA (단일)",        None),   # 보조 브랜치, 목표 없음
    ("step7_ensemble",     "Step 7 앙상블 (최종 목표)",        0.8404),
]
FINAL_KEY = "step7_ensemble"
FINAL_TARGET = 0.8404
TOL = 0.01  # 허용 오차


def read_f1_macro(report_path: Path):
    if not report_path.exists():
        return None
    txt = report_path.read_text(encoding="utf-8")
    m = re.search(r"Test F1-Macro:\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_kr_bert")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    print(f"{'단계':<32} {'측정 F1':>8} {'발표본':>8} {'판정':>6}")
    print("─" * 60)

    final_f1 = None
    for key, label, target in PRESENTATION_TARGETS:
        f1 = read_f1_macro(rd / f"{key}_report.txt")
        if key == FINAL_KEY:
            final_f1 = f1
        if f1 is None:
            verdict = "미실행"
            tgt_s = f"{target:.4f}" if target is not None else "  —  "
            print(f"{label:<32} {'—':>8} {tgt_s:>8} {verdict:>6}")
            continue
        if target is None:
            print(f"{label:<32} {f1:>8.4f} {'  —  ':>8} {'(보조)':>6}")
            continue
        verdict = "PASS" if f1 >= target - args.tol else "FAIL"
        print(f"{label:<32} {f1:>8.4f} {target:>8.4f} {verdict:>6}")

    print("─" * 60)
    if final_f1 is None:
        print(f"❌ GOAL 미검증 — 최종 앙상블({FINAL_KEY}) 결과가 아직 없습니다.")
        sys.exit(1)
    if final_f1 >= FINAL_TARGET - args.tol:
        print(f"✅ GOAL 달성 — 최종 앙상블 F1-Macro {final_f1:.4f} ≥ 발표본 {FINAL_TARGET:.4f} (tol {args.tol})")
        sys.exit(0)
    print(f"❌ GOAL 미달 — 최종 앙상블 F1-Macro {final_f1:.4f} < 발표본 {FINAL_TARGET:.4f} (tol {args.tol})")
    sys.exit(1)


if __name__ == "__main__":
    main()
