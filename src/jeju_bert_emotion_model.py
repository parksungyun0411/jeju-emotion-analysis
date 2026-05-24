"""
제주어 감정 분류 — BERT 기반 모델 (MPS 가속)

발표본 명세의 4개 sklearn 베이스라인 위에 BERT fine-tuning을 추가한 모델.
- 모델: klue/bert-base (한국어 BERT, 110M params)
- 가속: Apple M-series GPU (MPS backend)
- 평가: Accuracy, F1-Macro, F1-Weighted (sklearn 베이스라인과 동일 지표)
- 분할: 80:20 stratified (동일)
"""
import os
import re
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

EMOTION_LABELS = {0: "중립", 1: "슬픔", 2: "행복", 3: "분노", 4: "놀람", 5: "공포", 6: "혐오"}
NUM_LABELS = len(EMOTION_LABELS)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'[^\w\s.!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_one_epoch(model, loader, optimizer, scheduler, device, log_every=50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % log_every == 0:
            logger.info(f"  step {step+1}/{len(loader)} loss={loss.item():.4f} acc={correct/total:.4f}")

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, f1_macro, f1_weighted, all_preds, all_labels


def run_dataset(dataset_path, dataset_name, text_column, model_name, device,
                batch_size=16, epochs=3, lr=2e-5, max_length=64, output_dir='results_bert'):
    logger.info(f"=== {dataset_name} ({text_column}) — {model_name} ===")
    df = pd.read_excel(dataset_path)
    logger.info(f"  loaded {len(df):,} rows")

    df = df.copy()
    df['processed_text'] = df[text_column].apply(preprocess_text)
    df = df[df['processed_text'].str.len() > 0]
    logger.info(f"  after preprocess: {len(df):,} rows")

    X = df['processed_text'].tolist()
    y = df['감정번호'].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"  train={len(X_train):,}  test={len(X_test):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = EmotionDataset(X_train, y_train, tokenizer, max_length=max_length)
    test_ds = EmotionDataset(X_test, y_test, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        acc, f1_m, f1_w, _, _ = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        logger.info(
            f"  epoch {epoch}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_acc={acc:.4f} f1_macro={f1_m:.4f} f1_weighted={f1_w:.4f} ({elapsed:.1f}s)"
        )
        history.append((epoch, train_loss, train_acc, acc, f1_m, f1_w))

    # 마지막 평가 (상세 리포트)
    acc, f1_m, f1_w, preds, labels = evaluate(model, test_loader, device)
    report = classification_report(
        labels, preds, target_names=[EMOTION_LABELS[i] for i in range(NUM_LABELS)],
        digits=4, zero_division=0
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    suffix = f"{dataset_name}_{text_column}_BERT".replace(' ', '_')
    with open(f"{output_dir}/{suffix}_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {dataset_name} ({len(df):,} rows)\n")
        f.write(f"Text column: {text_column}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {epochs}  Batch: {batch_size}  LR: {lr}  Max length: {max_length}\n\n")
        f.write(f"Test Accuracy:    {acc:.4f}\n")
        f.write(f"Test F1-Macro:    {f1_m:.4f}\n")
        f.write(f"Test F1-Weighted: {f1_w:.4f}\n\n")
        f.write("Per-class:\n")
        f.write(report)
        f.write("\nEpoch history:\n")
        for ep, tl, ta, va, fm, fw in history:
            f.write(f"  epoch {ep}: train_loss={tl:.4f} train_acc={ta:.4f} test_acc={va:.4f} f1_m={fm:.4f} f1_w={fw:.4f}\n")
    logger.info(f"  saved → {output_dir}/{suffix}_report.txt")

    return {
        'dataset': dataset_name,
        'text_column': text_column,
        'model': model_name,
        'accuracy': acc,
        'f1_macro': f1_m,
        'f1_weighted': f1_w,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='klue/bert-base',
                        help='HuggingFace model name (default: klue/bert-base)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--datasets', nargs='+', default=['최종데이터.xlsx', '최종데이터_균형.xlsx'])
    parser.add_argument('--text-columns', nargs='+', default=['제주어 문장', '표준어 문장'],
                        help='어느 컬럼에 대해 학습할지 (제주어 / 표준어 / 둘 다)')
    parser.add_argument('--output-dir', default='results_bert')
    args = parser.parse_args()

    device = get_device()
    logger.info(f"device = {device}")

    all_results = []
    for path in args.datasets:
        name = os.path.splitext(os.path.basename(path))[0]
        for col in args.text_columns:
            res = run_dataset(
                path, name, col,
                model_name=args.model,
                device=device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                max_length=args.max_length,
                output_dir=args.output_dir,
            )
            all_results.append(res)

    # 요약 보고서
    summary_path = Path(args.output_dir) / 'bert_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"🎯 BERT 감정 분류 — 종합 결과\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}  Max length: {args.max_length}\n\n")
        f.write(f"{'Dataset':<25} {'Text':<10} {'Acc':>8} {'F1-Macro':>10} {'F1-Weighted':>13}\n")
        f.write("-" * 70 + "\n")
        for r in all_results:
            f.write(f"{r['dataset']:<25} {r['text_column']:<10} "
                    f"{r['accuracy']:>8.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>13.4f}\n")
    logger.info(f"summary → {summary_path}")
    print(open(summary_path).read())


if __name__ == "__main__":
    main()
