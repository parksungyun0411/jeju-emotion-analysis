"""
제주어 ↔ 표준어 사전(lexicon) 추출 — 번역 폴백/후처리용.

GPU/모델 불필요(pandas + collections 만). train_translation.py 와 동일한 병렬쌍
데이터(merged_jeju_balanced_sorted.xlsx)에서 어절 단위 치환쌍을 마이닝한다.

마이닝 방법 (외부 의존 없음):
  - 각 (제주어, 표준어) 문장쌍을 공백 어절로 분리.
  - 두 문장의 어절 수가 같은 쌍만 사용(위치 정합이 신뢰 가능) → 같은 위치 어절끼리 비교.
  - 표면형이 다른 (제주어어절 → 표준어어절) 치환을 후보로 수집하고 빈도 집계.
    같은 어절(번역 불변)은 제외.
  - 동일 제주어 어절에 여러 표준어 후보가 있으면 가장 빈번한 1개를 채택(빈도 우위).
  - 빈도 임계값(--min-freq) 이상이고 상위 N(--top-n) 개만 채택.
  - 양방향(j2s, s2j) 사전 생성.

산출: src/translation/data/jeju_std_dict.json
  {
    "meta": {... 추출 통계 ...},
    "j2s": {제주어어절: 표준어어절, ...},
    "s2j": {표준어어절: 제주어어절, ...},
    "j2s_freq": {제주어어절: 빈도, ...},
    "s2j_freq": {표준어어절: 빈도, ...}
  }

사용:
  python3 src/translation/build_dictionary.py
  python3 src/translation/build_dictionary.py --min-freq 3 --top-n 2000
"""
import re
import json
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

JEJU_COL = "제주어 문장"
STD_COL = "표준어 문장"

DEFAULT_DATA = "../2학기_감정분류/merged_jeju_balanced_sorted.xlsx"
DEFAULT_OUT = "src/translation/data/jeju_std_dict.json"

# 어절 끝 구두점만 분리하기 위한 정규화: 양끝의 . , ? ! 등을 떼어 비교 노이즈를 줄인다.
_PUNCT = ".,?!\"'·…~“”’‘()[]{}"


def _norm_token(tok):
    """어절 양끝 구두점 제거(가운데는 보존). 비교용 표면형 반환."""
    return tok.strip(_PUNCT)


def mine_substitutions(df, min_freq, top_n):
    """어절 위치정합 치환쌍 마이닝. (j2s_dict, s2j_dict, freq..., stats) 반환."""
    # 방향별 치환 후보 빈도: src_tok -> Counter({tgt_tok: count})
    j2s_cand = defaultdict(Counter)
    s2j_cand = defaultdict(Counter)

    n_pairs = len(df)
    n_aligned = 0          # 어절 수가 일치해 사용한 문장쌍 수
    n_sub_tokens = 0       # 수집한 치환 토큰 인스턴스 수

    jeju = df[JEJU_COL].astype(str).tolist()
    std = df[STD_COL].astype(str).tolist()

    for j_sent, s_sent in zip(jeju, std):
        j_toks = j_sent.split()
        s_toks = s_sent.split()
        if not j_toks or not s_toks or len(j_toks) != len(s_toks):
            continue
        n_aligned += 1
        for jt, st in zip(j_toks, s_toks):
            jn, sn = _norm_token(jt), _norm_token(st)
            if not jn or not sn:
                continue
            if jn == sn:          # 번역 불변 어절 → 사전에 넣을 필요 없음
                continue
            j2s_cand[jn][sn] += 1
            s2j_cand[sn][jn] += 1
            n_sub_tokens += 1

    def pick(cand):
        """각 src 어절에 대해 최빈 tgt 1개 채택. (dict, freq, total_candidate_srcs)."""
        chosen = {}
        freq = {}
        for src, ctr in cand.items():
            tgt, cnt = ctr.most_common(1)[0]
            if cnt >= min_freq:
                chosen[src] = tgt
                freq[src] = cnt
        # 빈도 내림차순 정렬 후 상위 N
        items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        chosen = {k: chosen[k] for k, _ in items}
        freq = {k: v for k, v in items}
        return chosen, freq, len(cand)

    j2s_dict, j2s_freq, j2s_total = pick(j2s_cand)
    s2j_dict, s2j_freq, s2j_total = pick(s2j_cand)

    stats = dict(
        n_pairs=n_pairs,
        n_aligned=n_aligned,
        n_sub_token_instances=n_sub_tokens,
        j2s_candidates=j2s_total,
        s2j_candidates=s2j_total,
        j2s_accepted=len(j2s_dict),
        s2j_accepted=len(s2j_dict),
        min_freq=min_freq,
        top_n=top_n,
    )
    return j2s_dict, s2j_dict, j2s_freq, s2j_freq, stats


def build(data_path, out_path, min_freq, top_n):
    logger.info(f"=== 사전 추출 (data={data_path}) ===")
    df = pd.read_excel(data_path)
    df = df.dropna(subset=[JEJU_COL, STD_COL]).reset_index(drop=True)
    df[JEJU_COL] = df[JEJU_COL].astype(str)
    df[STD_COL] = df[STD_COL].astype(str)
    logger.info(f"  loaded {len(df):,} parallel pairs")

    j2s, s2j, j2s_freq, s2j_freq, stats = mine_substitutions(df, min_freq, top_n)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": stats,
        "j2s": j2s,
        "s2j": s2j,
        "j2s_freq": j2s_freq,
        "s2j_freq": s2j_freq,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("  --- 추출 통계 ---")
    logger.info(f"  병렬쌍 총수            : {stats['n_pairs']:,}")
    logger.info(f"  어절수 일치 쌍(사용)   : {stats['n_aligned']:,}")
    logger.info(f"  치환 토큰 인스턴스     : {stats['n_sub_token_instances']:,}")
    logger.info(f"  j2s 후보 어절 수       : {stats['j2s_candidates']:,}")
    logger.info(f"  s2j 후보 어절 수       : {stats['s2j_candidates']:,}")
    logger.info(f"  채택 (min_freq>={min_freq}, top {top_n})  j2s={stats['j2s_accepted']:,}  s2j={stats['s2j_accepted']:,}")

    examples = list(j2s.items())[:5]
    logger.info("  --- j2s 예시 5개 (제주어 → 표준어, 빈도) ---")
    for k, v in examples:
        logger.info(f"    {k!r} → {v!r}  (freq={j2s_freq.get(k)})")

    logger.info(f"  saved → {out}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="제주어↔표준어 사전(폴백) 추출")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--min-freq", type=int, default=3,
                    help="치환쌍 채택 최소 빈도 (기본 3)")
    ap.add_argument("--top-n", type=int, default=2000,
                    help="방향별 채택 상위 빈도 항목 수 (기본 2000)")
    args = ap.parse_args()
    build(args.data, args.out, args.min_freq, args.top_n)


if __name__ == "__main__":
    main()
