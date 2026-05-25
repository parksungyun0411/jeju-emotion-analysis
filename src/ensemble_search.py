"""
다중 모델 가중 앙상블 가중치 탐색 (2~N개)
──────────────────────────────────────────
각 단계가 저장한 {name}_probs.npz (probs[N,7], labels[N]) 를 받아,
가중치 단순체(simplex, 합=1) 위를 그리드 탐색해 F1-Macro 최대 조합을 찾는다.
2-way 그리드의 일반화 — 가중치 0이면 해당 모델 제외이므로 부분조합도 모두 포함한다.

사용:
  python3 src/ensemble_search.py \
    --probs results_kr_bert/step3_dual_probs.npz \
            results_kr_bert/branch1_koelectra_probs.npz \
            results_kr_bert/step6_hard_mining_probs.npz \
    --names dual koelectra hardmining \
    --step 0.05 --output-dir results_kr_bert
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def simplex_weights(n, step):
    """합이 1인 n차원 가중치 그리드(해상도 step)를 생성."""
    k = round(1.0 / step)
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            last = k - sum(combo)
            yield tuple(c / k for c in combo) + (last / k,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True, help="*_probs.npz 파일들")
    ap.add_argument("--names", nargs="+", default=None, help="모델 이름(라벨용)")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--output-dir", default="results_kr_bert")
    ap.add_argument("--name", default="ensemble3", help="리포트 파일명")
    args = ap.parse_args()

    names = args.names or [Path(p).stem.replace("_probs", "") for p in args.probs]
    assert len(names) == len(args.probs)

    probs_list, labels_ref = [], None
    for p in args.probs:
        d = np.load(p)
        probs_list.append(d["probs"])
        if labels_ref is None:
            labels_ref = d["labels"]
        else:
            assert np.array_equal(labels_ref, d["labels"]), \
                f"라벨 불일치: {p} — 동일 seed/test set 인지 확인"
    labels = labels_ref
    n = len(probs_list)

    # 개별 모델 성능
    print("개별 모델 F1-Macro:")
    for nm, pr in zip(names, probs_list):
        f1 = f1_score(labels, pr.argmax(-1), average="macro", zero_division=0)
        print(f"  {nm:<14} {f1:.4f}")

    best = None
    for w in simplex_weights(n, args.step):
        ens = sum(wi * pi for wi, pi in zip(w, probs_list))
        preds = ens.argmax(-1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if best is None or f1 > best[1]:
            best = (w, f1, preds)
    w, f1m, preds = best
    acc = accuracy_score(labels, preds)
    wstr = ", ".join(f"{nm}={wi:.2f}" for nm, wi in zip(names, w))
    print(f"\n최적 가중치: {wstr}")
    print(f"  Ensemble  acc={acc:.4f}  F1-Macro={f1m:.4f}")
    print(f"  발표본 목표 0.8404 {'✅ 돌파' if f1m >= 0.8404 else ('≈ 도달(±1%)' if f1m >= 0.8304 else '미달')}")

    out = Path(args.output_dir) / f"{args.name}_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("== 다중 모델 가중 앙상블 탐색 ==\n")
        f.write(f"models: {names}\n")
        f.write(f"best weights: {wstr}\n")
        f.write(f"Test Accuracy:    {acc:.4f}\n")
        f.write(f"Test F1-Macro:    {f1m:.4f}\n")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
