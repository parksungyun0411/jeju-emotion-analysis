"""
제주어 ↔ 표준어 번역 — KoBART seq2seq 파인튜닝

마스터 설계 "3. 서브시스템 B — 번역 모델" 구현.
- 사전학습 한국어 seq2seq KoBART (gogamza/kobart-base-v2) 파인튜닝.
- 양방향 단일 모델 + 방향 prefix (한 모델로 두 방향 학습):
    제주어→표준어: 입력 "표준어로: <제주어>" → 타깃 <표준어>
    표준어→제주어: 입력 "제주어로: <표준어>" → 타깃 <제주어>
  --direction both (기본) 이면 각 병렬쌍이 양방향 2 샘플이 되어 데이터 2배.
- 평가: beam search 생성(num_beams=4) 후 sacrebleu 로 BLEU + chrF, 방향별 분리 보고.

코드 스타일(get_device / bf16 autocast / 동적 패딩 / CLI)은 src/jeju_kr_bert.py 와 일관.

CLI:
  python3 src/translation/train_translation.py --epochs 5 --fp16
  python3 src/translation/train_translation.py --direction j2s --limit 2000  (스모크)

⚠️ 단일 GPU(MPS)는 감정 goal run 과 직렬. 감정 학습 종료 후 실행.
"""
import os
import time
import json
import argparse
import logging
import random
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "gogamza/kobart-base-v2"
JEJU_COL = "제주어 문장"
STD_COL = "표준어 문장"

# 방향 prefix: 출력 언어를 가리킨다.
#   j2s = 제주어 → 표준어 : 입력 앞에 "표준어로: "
#   s2j = 표준어 → 제주어 : 입력 앞에 "제주어로: "
PREFIX = {"j2s": "표준어로: ", "s2j": "제주어로: "}

DEFAULT_HP = dict(
    lr=5e-5,
    batch_size=32,
    epochs=5,
    max_length=64,      # 동적 패딩이므로 truncation 상한만 의미
    weight_decay=0.01,
    warmup_ratio=0.10,
    grad_clip=1.0,
    num_beams=4,
    seed=42,
    eval_samples=1500,  # 매 epoch val 평가용 부분집합(beam 생성이 느려 전체 대신)
    test_samples=4000,  # 최종 test 평가 부분집합 상한
)

# Mixed precision: main()에서 --fp16 시 torch.bfloat16 으로 설정
AMP_DTYPE = None


def amp_ctx(device):
    """bf16 autocast 컨텍스트 (AMP_DTYPE None이면 no-op)."""
    if AMP_DTYPE is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=AMP_DTYPE)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_811(df, seed=42):
    """8:1:1 → Train/Val/Test (번역은 감정과 분할 정합성 불필요)."""
    train_val, test = train_test_split(df, test_size=0.1, random_state=seed)
    # 8:1 → val is 1/9 of train_val
    train, val = train_test_split(train_val, test_size=1/9, random_state=seed)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def build_pairs(df, direction):
    """방향별 (source_text_with_prefix, target_text, direction) 리스트 생성.

    direction='both' 이면 각 병렬쌍을 j2s, s2j 두 샘플로 확장."""
    dirs = ["j2s", "s2j"] if direction == "both" else [direction]
    sources, targets, sample_dirs = [], [], []
    jeju = df[JEJU_COL].astype(str).tolist()
    std = df[STD_COL].astype(str).tolist()
    for j, s in zip(jeju, std):
        for d in dirs:
            if d == "j2s":          # 제주어 → 표준어
                sources.append(PREFIX["j2s"] + j)
                targets.append(s)
            else:                   # s2j: 표준어 → 제주어
                sources.append(PREFIX["s2j"] + s)
                targets.append(j)
            sample_dirs.append(d)
    return sources, targets, sample_dirs


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """방향 prefix 가 붙은 source/target. 토큰화를 __init__ 에서 1회만 수행(에폭마다
    재토큰화 X), 패딩은 DataCollatorForSeq2Seq 가 배치별 최댓값으로 동적 처리."""

    def __init__(self, sources, targets, sample_dirs, tokenizer, max_length=64):
        model_inputs = tokenizer(
            [str(t) for t in sources], truncation=True, max_length=max_length
        )
        labels = tokenizer(
            text_target=[str(t) for t in targets], truncation=True, max_length=max_length
        )
        self.input_ids = model_inputs["input_ids"]
        self.attention_mask = model_inputs["attention_mask"]
        self.labels = labels["input_ids"]
        self.sample_dirs = list(sample_dirs)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# ─── Training ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip, epoch, total_epochs):
    model.train()
    losses = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp_ctx(device):
            out = model(**batch)
            loss = out.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            logger.info(
                f"  [{device}] epoch {epoch} step {step+1}/{len(loader)} "
                f"loss={loss.item():.4f}"
            )
    elapsed = time.time() - t0
    return float(np.mean(losses)) if losses else 0.0, elapsed


@torch.no_grad()
def generate_predictions(model, dataset, tokenizer, collator, device, hp):
    """Dataset 의 source 들을 beam search 로 생성. 디코드된 문자열 리스트 반환.

    DataLoader(shuffle=False) 순서가 dataset 순서와 일치하므로 sample_dirs 와 1:1 정렬됨."""
    model.eval()
    loader = DataLoader(dataset, batch_size=hp["batch_size"], shuffle=False, collate_fn=collator)
    preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with amp_ctx(device):
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=hp["num_beams"],
                max_length=hp["max_length"],
            )
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return preds


def compute_bleu_chrf(predictions, references):
    """sacrebleu 로 BLEU + chrF. references 는 문장 리스트(단일 참조)."""
    import sacrebleu
    refs = [list(references)]  # sacrebleu: list of reference-streams
    bleu = sacrebleu.corpus_bleu(list(predictions), refs)
    chrf = sacrebleu.corpus_chrf(list(predictions), refs)
    return bleu.score, chrf.score


def evaluate(model, dataset, raw_targets, sample_dirs, tokenizer, collator, device, hp):
    """전체 생성 후 방향별로 BLEU/chrF 분리 계산. {dir: {bleu, chrf, n}} 반환."""
    preds = generate_predictions(model, dataset, tokenizer, collator, device, hp)
    results = {}
    dirs_present = sorted(set(sample_dirs))
    for d in dirs_present:
        idxs = [i for i, sd in enumerate(sample_dirs) if sd == d]
        d_preds = [preds[i] for i in idxs]
        d_refs = [raw_targets[i] for i in idxs]
        bleu, chrf = compute_bleu_chrf(d_preds, d_refs)
        results[d] = {"bleu": bleu, "chrf": chrf, "n": len(idxs)}
    # 전체(방향 통합) 도 함께 보고
    bleu_all, chrf_all = compute_bleu_chrf(preds, raw_targets)
    results["overall"] = {"bleu": bleu_all, "chrf": chrf_all, "n": len(preds)}
    return results, preds


def save_report(out_dir, hp, history, val_results, test_results, direction):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "translation_report.txt"
    dir_name = {"j2s": "제주어→표준어", "s2j": "표준어→제주어", "overall": "전체"}
    with open(path, "w", encoding="utf-8") as f:
        f.write("== KoBART 제주어↔표준어 번역 ==\n")
        f.write(f"Base model: {MODEL_NAME}\n")
        f.write(f"Direction:  {direction}\n")
        f.write(f"Hyperparams: {json.dumps(hp, ensure_ascii=False)}\n\n")
        f.write("Epoch history:\n")
        for h in history:
            f.write("  " + json.dumps(h, ensure_ascii=False) + "\n")
        f.write("\n-- Validation (beam search, sacrebleu) --\n")
        for d, m in val_results.items():
            f.write(f"  [{dir_name.get(d, d)}] n={m['n']}  "
                    f"BLEU={m['bleu']:.2f}  chrF={m['chrf']:.2f}\n")
        f.write("\n-- Test (beam search, sacrebleu) --\n")
        for d, m in test_results.items():
            f.write(f"  [{dir_name.get(d, d)}] n={m['n']}  "
                    f"BLEU={m['bleu']:.2f}  chrF={m['chrf']:.2f}\n")
    logger.info(f"  saved → {path}")
    return path


def run_train(data_path, direction, hp, output_dir, limit=None):
    device = get_device()
    set_seed(hp["seed"])
    logger.info(f"=== KoBART 번역 (direction={direction}, base={MODEL_NAME}, device={device}) ===")

    df = pd.read_excel(data_path)
    df = df.dropna(subset=[JEJU_COL, STD_COL]).reset_index(drop=True)
    df[JEJU_COL] = df[JEJU_COL].astype(str)
    df[STD_COL] = df[STD_COL].astype(str)
    logger.info(f"  loaded {len(df):,} parallel pairs")
    if limit:
        df = df.sample(n=min(int(limit), len(df)), random_state=hp["seed"]).reset_index(drop=True)
        logger.info(f"  [SMOKE] subsampled to {len(df):,} pairs")

    train_df, val_df, test_df = split_811(df, seed=hp["seed"])
    logger.info(f"  pairs  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    tr_src, tr_tgt, tr_dirs = build_pairs(train_df, direction)
    va_src, va_tgt, va_dirs = build_pairs(val_df, direction)
    te_src, te_tgt, te_dirs = build_pairs(test_df, direction)
    logger.info(f"  samples train={len(tr_src):,}  val={len(va_src):,}  test={len(te_src):,} "
                f"(direction={direction})")

    train_ds = TranslationDataset(tr_src, tr_tgt, tr_dirs, tokenizer, hp["max_length"])

    # 평가는 부분집합으로 한다: beam search 생성이 느려 전체 val(~수만 샘플)을 매 epoch
    # 돌리면 학습보다 오래 걸린다. val은 매 epoch eval_samples개, test는 최종 test_samples개.
    # build_pairs가 쌍마다 [j2s, s2j]를 교차 생성하므로 앞에서 자르면 방향 균형이 유지된다.
    n_val = min(hp["eval_samples"], len(va_src))
    n_test = min(hp["test_samples"], len(te_src))
    va_tgt, va_dirs = va_tgt[:n_val], va_dirs[:n_val]
    te_tgt, te_dirs = te_tgt[:n_test], te_dirs[:n_test]
    val_ds = TranslationDataset(va_src[:n_val], va_tgt, va_dirs, tokenizer, hp["max_length"])
    test_ds = TranslationDataset(te_src[:n_test], te_tgt, te_dirs, tokenizer, hp["max_length"])
    logger.info(f"  eval subset  val={n_val:,}  test={n_test:,} (beam 생성 비용 절감)")

    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    total_steps = len(train_loader) * hp["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(hp["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    history = []
    for epoch in range(1, hp["epochs"] + 1):
        train_loss, elapsed = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, hp["grad_clip"],
            epoch, hp["epochs"],
        )
        val_results, _ = evaluate(model, val_ds, va_tgt, va_dirs, tokenizer, collator, device, hp)
        ov = val_results["overall"]
        logger.info(
            f"  epoch {epoch}/{hp['epochs']} train_loss={train_loss:.4f} "
            f"val_BLEU={ov['bleu']:.2f} val_chrF={ov['chrf']:.2f} ({elapsed:.0f}s)"
        )
        history.append(dict(
            epoch=epoch, train_loss=train_loss,
            val_bleu=ov["bleu"], val_chrf=ov["chrf"],
        ))

    logger.info("  === TEST evaluation (beam search) ===")
    test_results, _ = evaluate(model, test_ds, te_tgt, te_dirs, tokenizer, collator, device, hp)
    for d, m in test_results.items():
        logger.info(f"  TEST [{d}] BLEU={m['bleu']:.2f} chrF={m['chrf']:.2f} (n={m['n']})")

    save_report(output_dir, hp, history, val_results, test_results, direction)

    model_dir = Path(output_dir) / "kobart_jeju"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"  saved model → {model_dir}")
    return test_results


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="제주어↔표준어 KoBART 번역 학습")
    parser.add_argument("--data", default="../2학기_감정분류/merged_jeju_balanced_sorted.xlsx")
    parser.add_argument("--output-dir", default="results_translation")
    parser.add_argument("--direction", choices=["both", "j2s", "s2j"], default="both",
                        help="both=각 쌍을 양방향 2 샘플로 (기본)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_HP["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HP["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_HP["lr"])
    parser.add_argument("--max-length", type=int, default=DEFAULT_HP["max_length"])
    parser.add_argument("--num-beams", type=int, default=DEFAULT_HP["num_beams"])
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_HP["eval_samples"],
                        help="매 epoch val 평가 샘플 수(beam 생성 비용 절감)")
    parser.add_argument("--test-samples", type=int, default=DEFAULT_HP["test_samples"],
                        help="최종 test 평가 샘플 수 상한")
    parser.add_argument("--limit", type=int, default=None,
                        help="작은 부분집합만 사용 (스모크 테스트용)")
    parser.add_argument("--fp16", action="store_true",
                        help="bf16 mixed precision autocast (MPS/CUDA 가속)")
    args = parser.parse_args()

    hp = dict(DEFAULT_HP)
    hp.update(dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   max_length=args.max_length, num_beams=args.num_beams,
                   eval_samples=args.eval_samples, test_samples=args.test_samples))

    global AMP_DTYPE
    if args.fp16:
        AMP_DTYPE = torch.bfloat16
        logger.info("mixed precision: bfloat16 autocast 활성화")

    run_train(args.data, args.direction, hp, args.output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
