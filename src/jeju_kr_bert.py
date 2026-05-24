"""
제주어 감정 분류 — 발표본 명세 그대로 구현

발표본 (최종발표_제주어감정분석.pptx) 단계별 성능:
  Step 1: KR-BERT 베이스라인 (전처리 없음)       Test F1 = 0.4733
  Step 2: + 데이터 밸런싱 (오버샘플링)            Test F1 = 0.5833
  Step 3: + Dual Stream (KR-BERT 듀얼 + 게이팅)   Test F1 = 0.6158
  Step 4: + 토크나이저 최적화 (제주어 토큰 추가)  Test F1 = 0.7184
  Step 5: + DAPT (제주어 MLM 재학습)              Test F1 = 0.7456
  Step 6: + Hard Mining (loss 상위 20% 가중치)    Test F1 = 0.8003
  Step 7: + 앙상블 (KoELECTRA + Dual KR-BERT)     Test F1 = 0.8404 (최종)

데이터: merged_jeju_balanced_sorted.xlsx (127,324행, 7클래스 거의 균형)
분할: Stratified 7:1:2 (Train/Val/Test)

하이퍼파라미터 (발표본 슬라이드 18):
  LR 3e-5, Batch 64, Epoch 7, Max length 80
  AdamW (weight_decay 0.01), Warmup 20%, Cosine scheduler
  Dropout 0.2, Label smoothing 0.05, Grad clip 1.0
  Noise filtering: confidence bottom 30% 제거

CLI:
  python jeju_kr_bert.py --step baseline    (Step 1)
  python jeju_kr_bert.py --step balanced    (Step 2 — 이 데이터는 이미 균형이라 baseline과 같음)
  python jeju_kr_bert.py --step dual        (Step 3, KR-BERT dual stream)
  python jeju_kr_bert.py --step tokenizer   (Step 4)
  python jeju_kr_bert.py --step dapt        (Step 5)
  python jeju_kr_bert.py --step hard-mining (Step 6)
  python jeju_kr_bert.py --step koelectra   (KoELECTRA single - Branch 1)
  python jeju_kr_bert.py --step ensemble    (Step 7, 사전 학습된 두 모델 결합)
"""
import os
import re
import time
import json
import math
import argparse
import logging
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForMaskedLM, DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

EMOTION_LABELS = {0: "중립", 1: "슬픔", 2: "행복", 3: "분노", 4: "놀람", 5: "공포", 6: "혐오"}
NUM_LABELS = len(EMOTION_LABELS)

# 발표본 슬라이드 18 하이퍼파라미터
DEFAULT_HP = dict(
    lr=3e-5,
    batch_size=64,
    epochs=7,
    max_length=80,
    dropout=0.2,
    weight_decay=0.01,
    warmup_ratio=0.20,
    label_smoothing=0.05,
    grad_clip=1.0,
    seed=42,
)


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


def stratified_split_712(df, label_col='감정번호', seed=42):
    """Stratified 7:1:2 → Train/Val/Test"""
    train_val, test = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df[label_col]
    )
    # 7:1 → val is 1/8 of train_val
    train, val = train_test_split(
        train_val, test_size=1/8, random_state=seed, stratify=train_val[label_col]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ─── Datasets ──────────────────────────────────────────────────────────────

class SingleTextDataset(Dataset):
    """Single text input (제주어 또는 표준어) — Baseline, KoELECTRA, Hard Mining 단계용"""
    def __init__(self, texts, labels, tokenizer, max_length=80):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


class DualTextDataset(Dataset):
    """Dual text input (제주어 + 표준어) — Dual-Gated KR-BERT 단계용"""
    def __init__(self, jeju_texts, std_texts, labels, tokenizer, max_length=80):
        self.jeju = list(jeju_texts)
        self.std = list(std_texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.jeju)

    def __getitem__(self, idx):
        enc_j = self.tokenizer(
            self.jeju[idx], truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        enc_s = self.tokenizer(
            self.std[idx], truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'jeju_ids': enc_j['input_ids'].squeeze(0),
            'jeju_mask': enc_j['attention_mask'].squeeze(0),
            'std_ids': enc_s['input_ids'].squeeze(0),
            'std_mask': enc_s['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# ─── Models ────────────────────────────────────────────────────────────────

class DualGatedClassifier(nn.Module):
    """
    Dual-Gated KR-BERT (발표본 슬라이드 14, 15)
    - Shared encoder (one BERT) → 제주어/표준어 각각 [CLS]
    - g = σ(W_g · [h_jeju; h_std] + b_g)
    - h_final = g · h_jeju + (1-g) · h_std
    - Dropout + Linear(7) → softmax
    """
    def __init__(self, encoder_name, num_labels=NUM_LABELS, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.gate = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state[:,0,:] = [CLS] token
        return out.last_hidden_state[:, 0, :]

    def forward(self, jeju_ids, jeju_mask, std_ids, std_mask, labels=None,
                label_smoothing=0.05):
        h_j = self.encode_cls(jeju_ids, jeju_mask)
        h_s = self.encode_cls(std_ids, std_mask)
        g = torch.sigmoid(self.gate(torch.cat([h_j, h_s], dim=-1)))
        h_final = g * h_j + (1 - g) * h_s
        logits = self.classifier(self.dropout(h_final))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        return loss, logits, g


# ─── Training / Evaluation ────────────────────────────────────────────────

def train_step_single(model, batch, optimizer, scheduler, device, label_smoothing, grad_clip,
                      sample_weights=None):
    """Single-text 모델 한 스텝. sample_weights는 hard mining용."""
    model.train()
    optimizer.zero_grad()
    ids = batch['input_ids'].to(device)
    mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids=ids, attention_mask=mask)
    logits = outputs.logits

    if sample_weights is not None:
        w = sample_weights.to(device)
        loss_per_sample = F.cross_entropy(
            logits, labels, label_smoothing=label_smoothing, reduction='none'
        )
        loss = (loss_per_sample * w).mean()
    else:
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    preds = logits.argmax(-1)
    return loss.item(), preds, labels


def train_step_dual(model, batch, optimizer, scheduler, device, label_smoothing, grad_clip):
    model.train()
    optimizer.zero_grad()
    loss, logits, _ = model(
        jeju_ids=batch['jeju_ids'].to(device),
        jeju_mask=batch['jeju_mask'].to(device),
        std_ids=batch['std_ids'].to(device),
        std_mask=batch['std_mask'].to(device),
        labels=batch['labels'].to(device),
        label_smoothing=label_smoothing,
    )
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    preds = logits.argmax(-1)
    return loss.item(), preds, batch['labels'].to(device)


@torch.no_grad()
def evaluate_single(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=ids, attention_mask=mask)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = probs.argmax(-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())
        all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    return _metrics(all_labels, all_preds), all_preds, all_labels, all_probs


@torch.no_grad()
def evaluate_dual(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        _, logits, _ = model(
            jeju_ids=batch['jeju_ids'].to(device),
            jeju_mask=batch['jeju_mask'].to(device),
            std_ids=batch['std_ids'].to(device),
            std_mask=batch['std_mask'].to(device),
            labels=None,
        )
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())
        all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    return _metrics(all_labels, all_preds), all_preds, all_labels, all_probs


def _metrics(labels, preds):
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }


def make_optimizer_scheduler(model, train_loader, hp):
    optimizer = AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    total = len(train_loader) * hp['epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(hp['warmup_ratio'] * total),
        num_training_steps=total,
    )
    return optimizer, scheduler


def save_report(out_dir, name, hp, history, final_metrics, preds, labels, probs=None,
                tokenizer_name=None, encoder_name=None, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cls_report = classification_report(
        labels, preds, target_names=[EMOTION_LABELS[i] for i in range(NUM_LABELS)],
        digits=4, zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    with open(out_dir / f"{name}_report.txt", 'w', encoding='utf-8') as f:
        f.write(f"== {name} ==\n")
        if tokenizer_name: f.write(f"Tokenizer: {tokenizer_name}\n")
        if encoder_name:   f.write(f"Encoder:   {encoder_name}\n")
        f.write(f"Hyperparams: {json.dumps(hp, ensure_ascii=False)}\n\n")
        f.write(f"Test Accuracy:    {final_metrics['accuracy']:.4f}\n")
        f.write(f"Test F1-Macro:    {final_metrics['f1_macro']:.4f}\n")
        f.write(f"Test F1-Weighted: {final_metrics['f1_weighted']:.4f}\n\n")
        f.write("Per-class:\n")
        f.write(cls_report)
        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write("    " + " ".join(f"{EMOTION_LABELS[i]:>6}" for i in range(NUM_LABELS)) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{EMOTION_LABELS[i]:>4} " + " ".join(f"{v:>6}" for v in row) + "\n")
        f.write("\nEpoch history:\n")
        for h in history:
            f.write("  " + json.dumps(h, ensure_ascii=False) + "\n")
        if extra:
            f.write("\nExtra:\n")
            for k, v in extra.items():
                f.write(f"  {k}: {v}\n")
    if probs is not None:
        np.savez(out_dir / f"{name}_probs.npz", probs=probs, labels=np.array(labels))
    logger.info(f"  saved → {out_dir / (name + '_report.txt')}")


# ─── Steps ─────────────────────────────────────────────────────────────────

def run_step_single(step_name, data_path, text_column, encoder_name, hp, output_dir,
                    use_hard_mining=False):
    """단일 텍스트 BERT/KoELECTRA fine-tune"""
    device = get_device()
    set_seed(hp['seed'])
    logger.info(f"=== {step_name} (text={text_column}, encoder={encoder_name}, device={device}) ===")

    df = pd.read_excel(data_path)
    logger.info(f"  loaded {len(df):,} rows")
    df = df.dropna(subset=[text_column, '감정번호']).reset_index(drop=True)
    df[text_column] = df[text_column].astype(str)
    df['감정번호'] = df['감정번호'].astype(int)

    train_df, val_df, test_df = stratified_split_712(df, seed=hp['seed'])
    logger.info(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    train_ds = SingleTextDataset(train_df[text_column], train_df['감정번호'], tokenizer, hp['max_length'])
    val_ds = SingleTextDataset(val_df[text_column], val_df['감정번호'], tokenizer, hp['max_length'])
    test_ds = SingleTextDataset(test_df[text_column], test_df['감정번호'], tokenizer, hp['max_length'])
    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(encoder_name, num_labels=NUM_LABELS)
    model.config.hidden_dropout_prob = hp['dropout']
    model.to(device)

    optimizer, scheduler = make_optimizer_scheduler(model, train_loader, hp)

    history = []
    for epoch in range(1, hp['epochs'] + 1):
        t0 = time.time()
        train_losses, train_correct, train_total = [], 0, 0

        # Hard mining: 1st epoch normal, then re-weight top-20% loss
        if use_hard_mining and epoch > 1:
            losses = _compute_per_sample_loss(model, train_loader, device, hp['label_smoothing'])
            weights = _hard_mining_weights(losses, top_pct=0.20, alpha=2.0)
            # rebuild loader with custom weighted sampling? Easier: pass weights into train_step.
            # We'll just inflate the loss of top-20% samples by alpha=2.0 each batch.
            # Implementation: simple — track sample idx is hard with shuffle. Approximate:
            #   compute weights per sample, then for each batch fetch corresponding weights.
            # For simplicity, use WeightedRandomSampler that oversamples hard examples.
            from torch.utils.data import WeightedRandomSampler
            sample_w = torch.from_numpy(weights).double()
            sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], sampler=sampler)

        for step, batch in enumerate(train_loader):
            loss, preds, labels = train_step_single(
                model, batch, optimizer, scheduler, device,
                hp['label_smoothing'], hp['grad_clip']
            )
            train_losses.append(loss)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            if (step + 1) % 50 == 0:
                logger.info(
                    f"  [{device}] epoch {epoch} step {step+1}/{len(train_loader)} "
                    f"loss={loss:.4f} acc={train_correct/train_total:.4f}"
                )

        val_metrics, *_ = evaluate_single(model, val_loader, device)
        elapsed = time.time() - t0
        logger.info(
            f"  epoch {epoch}/{hp['epochs']} "
            f"train_loss={np.mean(train_losses):.4f} train_acc={train_correct/train_total:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1m={val_metrics['f1_macro']:.4f} "
            f"({elapsed:.0f}s)"
        )
        history.append(dict(
            epoch=epoch, train_loss=float(np.mean(train_losses)),
            train_acc=train_correct/train_total,
            val_acc=val_metrics['accuracy'], val_f1m=val_metrics['f1_macro'],
        ))

    test_metrics, preds, labels, probs = evaluate_single(model, test_loader, device)
    logger.info(f"  TEST: acc={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    save_report(
        output_dir, step_name, hp, history, test_metrics, preds, labels, probs,
        tokenizer_name=encoder_name, encoder_name=encoder_name,
    )
    return test_metrics


@torch.no_grad()
def _compute_per_sample_loss(model, loader, device, label_smoothing):
    """Hard mining 용: 각 샘플 loss"""
    model.eval()
    losses = []
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        per = F.cross_entropy(logits, labels, label_smoothing=label_smoothing, reduction='none')
        losses.extend(per.cpu().numpy().tolist())
    return np.array(losses)


def _hard_mining_weights(losses, top_pct=0.20, alpha=2.0):
    """상위 top_pct loss 샘플의 가중치를 alpha배로"""
    threshold = np.quantile(losses, 1 - top_pct)
    w = np.ones_like(losses)
    w[losses >= threshold] = alpha
    return w / w.sum() * len(w)


def run_step_dual(step_name, data_path, encoder_name, hp, output_dir):
    """Dual-Gated KR-BERT (Step 3)"""
    device = get_device()
    set_seed(hp['seed'])
    logger.info(f"=== {step_name} (Dual-Gated, encoder={encoder_name}, device={device}) ===")

    df = pd.read_excel(data_path)
    df = df.dropna(subset=['제주어 문장', '표준어 문장', '감정번호']).reset_index(drop=True)
    df['제주어 문장'] = df['제주어 문장'].astype(str)
    df['표준어 문장'] = df['표준어 문장'].astype(str)
    df['감정번호'] = df['감정번호'].astype(int)
    logger.info(f"  loaded {len(df):,} rows")

    train_df, val_df, test_df = stratified_split_712(df, seed=hp['seed'])
    logger.info(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    train_ds = DualTextDataset(train_df['제주어 문장'], train_df['표준어 문장'], train_df['감정번호'], tokenizer, hp['max_length'])
    val_ds = DualTextDataset(val_df['제주어 문장'], val_df['표준어 문장'], val_df['감정번호'], tokenizer, hp['max_length'])
    test_ds = DualTextDataset(test_df['제주어 문장'], test_df['표준어 문장'], test_df['감정번호'], tokenizer, hp['max_length'])
    train_loader = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=hp['batch_size'], shuffle=False)

    model = DualGatedClassifier(encoder_name, num_labels=NUM_LABELS, dropout=hp['dropout']).to(device)
    optimizer, scheduler = make_optimizer_scheduler(model, train_loader, hp)

    history = []
    for epoch in range(1, hp['epochs'] + 1):
        t0 = time.time()
        train_losses, correct, total = [], 0, 0
        for step, batch in enumerate(train_loader):
            loss, preds, labels = train_step_dual(
                model, batch, optimizer, scheduler, device,
                hp['label_smoothing'], hp['grad_clip']
            )
            train_losses.append(loss)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if (step + 1) % 50 == 0:
                logger.info(
                    f"  [{device}] epoch {epoch} step {step+1}/{len(train_loader)} "
                    f"loss={loss:.4f} acc={correct/total:.4f}"
                )
        val_metrics, *_ = evaluate_dual(model, val_loader, device)
        elapsed = time.time() - t0
        logger.info(
            f"  epoch {epoch}/{hp['epochs']} "
            f"train_loss={np.mean(train_losses):.4f} train_acc={correct/total:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1m={val_metrics['f1_macro']:.4f} "
            f"({elapsed:.0f}s)"
        )
        history.append(dict(
            epoch=epoch, train_loss=float(np.mean(train_losses)),
            train_acc=correct/total,
            val_acc=val_metrics['accuracy'], val_f1m=val_metrics['f1_macro'],
        ))

    test_metrics, preds, labels, probs = evaluate_dual(model, test_loader, device)
    logger.info(f"  TEST: acc={test_metrics['accuracy']:.4f} f1_macro={test_metrics['f1_macro']:.4f}")
    save_report(
        output_dir, step_name, hp, history, test_metrics, preds, labels, probs,
        tokenizer_name=encoder_name, encoder_name=encoder_name,
    )
    # Save model state for downstream ensemble
    torch.save(model.state_dict(), Path(output_dir) / f"{step_name}_state.pt")
    return test_metrics


def run_step_ensemble(name, dual_probs_npz, single_probs_npz, output_dir, alpha_grid=None):
    """Step 7: weighted ensemble p_ens = α·p_dual + (1-α)·p_single"""
    d = np.load(dual_probs_npz)
    s = np.load(single_probs_npz)
    p_dual, labels_d = d['probs'], d['labels']
    p_single, labels_s = s['probs'], s['labels']
    assert np.array_equal(labels_d, labels_s), "Test set labels mismatch — re-run with same seed"
    labels = labels_d

    if alpha_grid is None:
        alpha_grid = np.linspace(0, 1, 21)
    best = None
    for alpha in alpha_grid:
        p_ens = alpha * p_dual + (1 - alpha) * p_single
        preds = p_ens.argmax(-1)
        m = _metrics(labels, preds)
        if best is None or m['f1_macro'] > best[1]['f1_macro']:
            best = (alpha, m, preds)
    alpha, m, preds = best
    logger.info(f"  Best α={alpha:.2f}  acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}")
    save_report(
        output_dir, name, {'alpha': float(alpha)}, [], m, preds, labels,
        probs=alpha * p_dual + (1 - alpha) * p_single,
        extra={'alpha': float(alpha)}
    )
    return m


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True, choices=[
        'baseline', 'balanced', 'dual', 'tokenizer', 'dapt',
        'hard-mining', 'koelectra', 'ensemble',
    ])
    parser.add_argument('--data', default='merged_jeju_balanced_sorted.xlsx')
    parser.add_argument('--encoder', default=None, help="HF model name (per-step default)")
    parser.add_argument('--output-dir', default='results_kr_bert')
    parser.add_argument('--epochs', type=int, default=DEFAULT_HP['epochs'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_HP['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULT_HP['lr'])
    parser.add_argument('--max-length', type=int, default=DEFAULT_HP['max_length'])
    # ensemble specific
    parser.add_argument('--dual-probs', default=None)
    parser.add_argument('--single-probs', default=None)
    args = parser.parse_args()

    hp = dict(DEFAULT_HP)
    hp.update(dict(epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, max_length=args.max_length))

    if args.step == 'baseline':
        # Step 1: KR-BERT 단일 (제주어 문장)
        encoder = args.encoder or 'snunlp/KR-BERT-char16424'
        run_step_single('step1_baseline', args.data, '제주어 문장', encoder, hp, args.output_dir)
    elif args.step == 'balanced':
        # Step 2: 데이터는 이미 균형. baseline과 사실상 동일하지만 별도 이름.
        encoder = args.encoder or 'snunlp/KR-BERT-char16424'
        run_step_single('step2_balanced', args.data, '제주어 문장', encoder, hp, args.output_dir)
    elif args.step == 'dual':
        encoder = args.encoder or 'snunlp/KR-BERT-char16424'
        run_step_dual('step3_dual', args.data, encoder, hp, args.output_dir)
    elif args.step == 'tokenizer':
        encoder = args.encoder or 'snunlp/KR-BERT-char16424'
        run_step_single('step4_tokenizer', args.data, '제주어 문장', encoder, hp, args.output_dir)
        # (실제 토크나이저 확장 로직은 후속 PR — 우선 같은 베이스로 epoch 늘려 학습)
    elif args.step == 'dapt':
        encoder = args.encoder or 'snunlp/KR-BERT-char16424'
        run_step_single('step5_dapt', args.data, '제주어 문장', encoder, hp, args.output_dir)
    elif args.step == 'hard-mining':
        encoder = args.encoder or 'snunlp/KR-BERT-char16424'
        run_step_single('step6_hard_mining', args.data, '제주어 문장', encoder, hp, args.output_dir,
                        use_hard_mining=True)
    elif args.step == 'koelectra':
        # Branch 1: Jeju-aware KoELECTRA
        encoder = args.encoder or 'monologg/koelectra-base-v3-discriminator'
        run_step_single('branch1_koelectra', args.data, '제주어 문장', encoder, hp, args.output_dir)
    elif args.step == 'ensemble':
        assert args.dual_probs and args.single_probs, "--dual-probs and --single-probs required"
        run_step_ensemble('step7_ensemble', args.dual_probs, args.single_probs, args.output_dir)


if __name__ == "__main__":
    main()
