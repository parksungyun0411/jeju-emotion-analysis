"""
번역 성능 목표 자동 검증
─────────────────────────
results_translation/translation_report.txt 를 파싱해 방향별/overall 의
BLEU·chrF 를 표로 출력하고, overall chrF 가 목표(기본 80) 이상인지 판정한다.

목표 정의:
  사용자 요구 "성능 80%" → overall chrF ≥ 80 (문자단위 F-score) 을 1차 목표.
  BLEU 도 함께 보고. tol 만큼 미달은 허용(verify_goal.py 컨벤션과 동일).

판정 기준:
  - overall chrF ≥ target_chrf - tol  이면 ✅ GOAL 달성 (exit 0)
  - 미만/결과 누락                     이면 ❌ 미달          (exit 1)

리포트는 -- Validation -- 과 -- Test -- 두 블록을 담는다. 최종 판정은
Test 블록의 overall(전체) chrF 로 한다(없으면 Validation 으로 폴백).

사용:
  python3 src/translation/verify_translation.py
  python3 src/translation/verify_translation.py --results-dir results_translation --target-chrf 80 --tol 1.0
  (exit code 0 = GOAL 달성, 1 = 미달/결과 누락)
"""
import re
import sys
import argparse
from pathlib import Path

DEFAULT_RESULTS_DIR = "results_translation"
DEFAULT_TARGET_CHRF = 80.0
DEFAULT_TOL = 1.0

# save_report() 가 쓰는 라인 포맷:
#   "  [제주어→표준어] n=1234  BLEU=12.34  chrF=56.78"
_LINE_RE = re.compile(
    r"\[(?P<dir>[^\]]+)\]\s*n=(?P<n>\d+)\s+BLEU=(?P<bleu>[0-9.]+)\s+chrF=(?P<chrf>[0-9.]+)"
)
_VAL_HDR = "-- Validation"
_TEST_HDR = "-- Test"

# 리포트의 한글 방향 라벨 → 내부 키
LABEL2KEY = {"제주어→표준어": "j2s", "표준어→제주어": "s2j", "전체": "overall"}
KEY_ORDER = ["j2s", "s2j", "overall"]
KEY_LABEL = {"j2s": "제주어→표준어", "s2j": "표준어→제주어", "overall": "전체(overall)"}


def parse_report(report_path: Path):
    """리포트를 {'validation': {key: {...}}, 'test': {key: {...}}} 로 파싱."""
    if not report_path.exists():
        return None
    txt = report_path.read_text(encoding="utf-8")
    sections = {"validation": {}, "test": {}}
    current = None
    for line in txt.splitlines():
        if _VAL_HDR in line:
            current = "validation"
            continue
        if _TEST_HDR in line:
            current = "test"
            continue
        if current is None:
            continue
        m = _LINE_RE.search(line)
        if not m:
            continue
        key = LABEL2KEY.get(m.group("dir").strip(), m.group("dir").strip())
        sections[current][key] = {
            "n": int(m.group("n")),
            "bleu": float(m.group("bleu")),
            "chrf": float(m.group("chrf")),
        }
    return sections


def _print_block(title, block):
    print(f"\n{title}")
    print(f"  {'방향':<16} {'n':>8} {'BLEU':>8} {'chrF':>8}")
    print("  " + "─" * 44)
    if not block:
        print("  (결과 없음)")
        return
    keys = [k for k in KEY_ORDER if k in block] + [k for k in block if k not in KEY_ORDER]
    for k in keys:
        m = block[k]
        print(f"  {KEY_LABEL.get(k, k):<16} {m['n']:>8} {m['bleu']:>8.2f} {m['chrf']:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description="번역 chrF 목표 검증")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--target-chrf", type=float, default=DEFAULT_TARGET_CHRF,
                    help="overall chrF 목표 (기본 80)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="허용 오차 (기본 1.0)")
    args = ap.parse_args()

    report_path = Path(args.results_dir) / "translation_report.txt"
    sections = parse_report(report_path)

    print("=" * 50)
    print(f"번역 성능 목표 검증 — overall chrF ≥ {args.target_chrf:.1f} (tol {args.tol})")
    print(f"리포트: {report_path}")
    print("=" * 50)

    if sections is None:
        print(f"\n❌ GOAL 미검증 — 리포트가 없습니다: {report_path}")
        print("   먼저 학습을 실행하세요: bash scripts/run_translation.sh")
        sys.exit(1)

    _print_block("-- Validation (beam search, sacrebleu) --", sections["validation"])
    _print_block("-- Test (beam search, sacrebleu) --", sections["test"])

    # 최종 판정: Test overall 우선, 없으면 Validation overall 폴백
    final_block, final_src = None, None
    if sections["test"].get("overall"):
        final_block, final_src = sections["test"]["overall"], "Test"
    elif sections["validation"].get("overall"):
        final_block, final_src = sections["validation"]["overall"], "Validation"

    print("\n" + "─" * 50)
    if final_block is None:
        print("❌ GOAL 미검증 — overall(전체) chrF 결과가 리포트에 없습니다.")
        sys.exit(1)

    chrf = final_block["chrf"]
    bleu = final_block["bleu"]
    if chrf >= args.target_chrf - args.tol:
        print(f"✅ GOAL 달성 — {final_src} overall chrF {chrf:.2f} ≥ 목표 {args.target_chrf:.1f} "
              f"(tol {args.tol})  [BLEU {bleu:.2f}]")
        sys.exit(0)
    print(f"❌ GOAL 미달 — {final_src} overall chrF {chrf:.2f} < 목표 {args.target_chrf:.1f} "
          f"(tol {args.tol})  [BLEU {bleu:.2f}]")
    sys.exit(1)


if __name__ == "__main__":
    main()
