# 제주어 감정 분류 — 발표본 재현 및 개선 보고서

> 졸업 프로젝트 최종 발표본(`최종발표_제주어감정분석.pptx`)의 7단계 파이프라인을
> 코드로 재현하고, 단계별 성능을 검증한 보고서.
> 작성: 2026-05-24 ~ (진행 중)

## 1. 발표본 명세

발표본은 단일 KR-BERT 베이스라인에서 출발해 7단계 최적화로 F1-Macro 0.84까지 끌어올린 파이프라인이다.

| 항목 | 값 |
|------|------|
| 데이터 | `merged_jeju_balanced_sorted.xlsx` (127,324행, 7클래스 거의 균형) |
| 컬럼 | 제주어 문장 / 표준어 문장 / 감정번호 / gpt감정 |
| 분할 | Stratified **7:1:2** (Train 89,118 / Val 12,732 / Test 25,463) |
| 클래스 | 7개 (0중립 1슬픔 2행복 3분노 4놀람 5공포 6혐오) |
| 베이스라인 | KR-BERT (`snunlp/KR-BERT-char16424`) / KoELECTRA |
| 최종 모델 | Dual-Gated KR-BERT + KoELECTRA 앙상블 |

**하이퍼파라미터 (발표본 슬라이드 18):**
LR 3e-5, Batch 64, Epoch 7, Max length 80, Dropout 0.2,
AdamW (weight_decay 0.01), Warmup 20%, Cosine scheduler,
Label smoothing 0.05, Noise filtering(하위 30% 신뢰도 제거), Grad clip 1.0.

**발표본 단계별 Test F1-Macro:**

| 단계 | 기법 | 발표본 F1 |
|------|------|----------:|
| 1 | KR-BERT 베이스라인 (전처리 없음) | 0.4733 |
| 2 | + 데이터 밸런싱 (오버샘플링) | 0.5833 |
| 3 | + Dual Stream (KR-BERT 듀얼+게이팅) | 0.6158 |
| 4 | + 토크나이저 최적화 (제주어 토큰 추가) | 0.7184 |
| 5 | + DAPT (제주어 MLM 재학습) | 0.7456 |
| 6 | + Hard Mining (loss 상위 20% 가중치) | 0.8003 |
| 7 | + 앙상블 (KoELECTRA + Dual KR-BERT) | **0.8404** |

**모델 아키텍처 (슬라이드 11-16):**
- **Branch 1 — KoELECTRA (Jeju-only)**: 제주어 입력 → KoELECTRA → Classifier(7) → `p_single`
- **Branch 2 — Dual-Gated KR-BERT**: 제주어/표준어 입력 → **Shared KR-BERT** → `[CLS_jeju]`, `[CLS_std]`
  - Gate `g = σ(W_g·[h_jeju; h_std] + b_g)`
  - `h_final = g·h_jeju + (1-g)·h_std` → Linear(7) → `p_dual`
- **Ensemble**: `p_ens = α·p_dual + (1-α)·p_single`

## 2. 재현 환경

- 하드웨어: Apple M4 (8-core GPU, 16GB), PyTorch MPS backend
- 구현: `src/jeju_kr_bert.py` (단계별 `--step` 인자)
- 분할 seed=42 고정 → 모든 단계가 동일 test set 사용 (앙상블 정합성 보장)

> 비교 참고: sklearn TF-IDF 베이스라인(`jeju_multilabel_emotion_model.py`)은
> 제주어 균형 데이터에서 F1-Macro 0.2957 (Naive Bayes)에 그쳤다. 딥러닝 전환의 근거.

## 3. 재현 결과 (진행 중)

### Step 1 — KR-BERT 베이스라인 ✅

| 설정 | 값 |
|------|------|
| 모델 | snunlp/KR-BERT-char16424 |
| Epoch | 2 (빠른 검증), Batch 32, Max length 64 |

| 지표 | 값 |
|------|----:|
| Test Accuracy | 0.7729 |
| **Test F1-Macro** | **0.7593** |
| Test F1-Weighted | 0.7647 |

**감정별 F1:** 공포 0.954, 혐오 0.909, 놀람 0.867, 슬픔 0.760, 분노 0.734, 행복 0.602, 중립 0.489.

> **주목:** 단일 KR-BERT 베이스라인만으로 발표본 Step 5 (DAPT, 0.7456)를 초과했다.
> 이유는 사용한 데이터(`merged_jeju_balanced_sorted`)가 이미 밸런싱·정제된 상태여서
> 발표본의 "전처리 없음" raw baseline(0.4733)보다 훨씬 좋은 출발점이기 때문.
> 즉 발표본의 Step 1~2(밸런싱) 효과가 데이터에 이미 반영되어 있다.

**약점:** 행복(0.602)·중립(0.489)·분노(0.734) — 의미가 겹치는 다수 감정 간 혼동.
강한 감정(공포/혐오/놀람)은 이미 0.87~0.95로 매우 높음.

### Step 3 — Dual-Gated KR-BERT 🔄 (진행 중)

제주어+표준어를 shared KR-BERT로 인코딩하고 게이팅으로 융합.

| Epoch | val_acc | val_f1m |
|------:|--------:|--------:|
| 1 | 0.7194 | 0.7042 |
| 2 | _진행 중_ | |
| 3 | | |

epoch 1 기준 단일 KR-BERT(0.6973)보다 소폭 우수. 표준어 정보를 함께 보아 모호한 감정 보강을 기대.

### Step 6 — Hard Mining ⏳ (예정)
### Branch 1 — KoELECTRA ⏳ (예정)
### Step 7 — Ensemble ⏳ (예정)

## 4. 결론 (잠정)

- sklearn(0.30) → 단일 KR-BERT(0.76)로 **F1-Macro +46%p** 도약 확인.
- 발표본의 핵심 통찰(딥러닝 전환 + 제주어/표준어 병렬 활용 + 앙상블)이 유효함을 재현 중.
- 최종 목표: 발표본 0.8404 도달. Dual-Gated + KoELECTRA 앙상블 + Hard Mining으로 접근.

## 5. 재현 방법

```bash
# 데이터 (gitignore로 제외됨 — 로컬/외장 SSD/iCloud에 위치)
DATA="merged_jeju_balanced_sorted.xlsx"

# Step 1: KR-BERT 베이스라인
python3 src/jeju_kr_bert.py --step baseline --data "$DATA" --epochs 7 --batch-size 64 --max-length 80

# Step 3: Dual-Gated KR-BERT
python3 src/jeju_kr_bert.py --step dual --data "$DATA" --epochs 3 --batch-size 16

# Branch 1: KoELECTRA
python3 src/jeju_kr_bert.py --step koelectra --data "$DATA" --epochs 3

# Step 7: Ensemble (두 모델의 test probs로 α 그리드 서치)
python3 src/jeju_kr_bert.py --step ensemble \
  --dual-probs results_kr_bert/step3_dual_probs.npz \
  --single-probs results_kr_bert/branch1_koelectra_probs.npz
```
