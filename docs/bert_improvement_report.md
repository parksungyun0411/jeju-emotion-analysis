# 제주어 감정 분류 재현 코드 작성 기록

> 졸업 뒤 2026년 5월에 최종 발표 자료(`최종발표_제주어감정분석.pptx`)를 보고 딥러닝 코드
> (`src/jeju_kr_bert.py`)를 다시 짠 기록이다. 발표 때 점수를 만든 원 코드는 남아 있지 않다.
> 2026년 5월에는 Step 1만 끝까지 돌렸고, 끝까지 돌린 재학습은 후속 저장소
> [jeju-emotion-lora](https://github.com/parksungyun0411/jeju-emotion-lora)에서 했다.
> 2026년 10월에 감정 이름과 단계별 상태를 정리했다.

## 1. 발표 자료의 명세

| 항목 | 값 |
|------|------|
| 데이터 | `merged_jeju_balanced_sorted.xlsx` (127,324행, 7클래스 거의 균형) |
| 컬럼 | 제주어 문장 / 표준어 문장 / 감정번호 / gpt감정 |
| 분할 | Stratified 7:1:2 |
| 클래스 | 0 중립, 1 행복, 2 슬픔, 3 분노, 4 놀람, 5 공포, 6 혐오 |
| 인코더 | KR-BERT (`snunlp/KR-BERT-char16424`), KoELECTRA |
| 최종 모델 | Dual-Gated KR-BERT + KoELECTRA 앙상블 |

클래스 번호와 이름은 학습 파일의 감정번호×gpt감정 교차표에서 정했다. 이 문서의 이전 판과 당시 코드는
1=슬픔, 2=행복으로 적고 있었다.

하이퍼파라미터(슬라이드 18): LR 3e-5, Batch 64, Epoch 7, Max length 80, Dropout 0.2,
AdamW(weight_decay 0.01), Warmup 20%, Cosine scheduler, Label smoothing 0.05,
Noise filtering(신뢰도 하위 30% 제거), Grad clip 1.0.

발표 자료의 단계별 Test F1-Macro:

| 단계 | 기법 | 발표 자료 F1 |
|------|------|-------------:|
| 1 | KR-BERT 베이스라인 | 0.4733 |
| 2 | + 데이터 밸런싱 | 0.5833 |
| 3 | + Dual-Gated KR-BERT | 0.6158 |
| 4 | + 토크나이저 최적화 | 0.7184 |
| 5 | + DAPT (MLM 재학습) | 0.7456 |
| 6 | + Hard Mining | 0.8003 |
| 7 | + 앙상블 (KoELECTRA + Dual KR-BERT) | 0.8404 |

학습 파일은 감정별 행 수를 맞추려고 같은 문장을 최대 56번 복제했다. 이 파일을 행 단위로 7:1:2로 나누면
test 행의 70.4%가 train에도 있는 문장이므로, 위 수치는 일반화 성능이 아니다
([누수 진단](https://github.com/parksungyun0411/jeju-emotion-lora/blob/main/reports/leakage.md)).

모델 구조(슬라이드 11~16):

- Branch 1, KoELECTRA: 제주어 문장 → KoELECTRA → Classifier(7) → `p_single`
- Branch 2, Dual-Gated KR-BERT: 제주어 문장과 표준어 문장 → 같은 KR-BERT → `[CLS_jeju]`, `[CLS_std]`
  - Gate `g = σ(W_g·[h_jeju; h_std] + b_g)`
  - `h_final = g·h_jeju + (1-g)·h_std` → Linear(7) → `p_dual`
- 앙상블: `p_ens = α·p_dual + (1-α)·p_single`

## 2. 단계별 상태

| 단계 | 이 저장소의 상태 | 잰 값 |
|------|------------------|------|
| 1 | 구현 (`--step baseline`) | 행 단위 0.7593 (2 epoch 확인용, `results/step1_baseline_report.txt`) |
| 2 | 별도 처리 없음. 학습 파일이 이미 균형이라 baseline과 같은 학습 | 없음 |
| 3 | 구현 (`--step dual`) | 이 저장소에는 epoch 1 검증 F1 0.7042만 기록. jeju-emotion-lora: 행 단위 0.8310, 문장 단위 0.4806 |
| 4 | 미구현. `--step tokenizer`는 baseline과 같은 학습 | 없음 |
| 5 | 미구현. `--step dapt`는 baseline과 같은 학습 | 없음 |
| 6 | 구현 (`--step hard-mining`). 앙상블에는 쓰지 않음 | 기록 없음 |
| KoELECTRA | 구현 (`--step koelectra`) | jeju-emotion-lora: 행 단위 0.8199, 문장 단위 0.4968 |
| 7 | 구현 (`--step ensemble`) | jeju-emotion-lora: 행 단위 0.8393 (α 0.55), 문장 단위 0.5192 (α 0.50) |

- 발표 자료의 'Noise filtering'은 구현하지 않았다.
- 이 저장소의 앙상블은 3단계 Dual-Gated KR-BERT와 KoELECTRA 두 모델만 결합한다. 발표 자료처럼 4~6단계를
  누적한 모델이 아니다.
- α는 test 확률로 그리드 서치해 고르므로 앙상블 값은 그만큼 낙관적이다.
- jeju-emotion-lora의 값은 이 저장소의 `src/jeju_kr_bert.py`를 그대로 불러와 분할 함수만 바꿔 다시 학습한
  결과다([결과 표](https://github.com/parksungyun0411/jeju-emotion-lora/blob/main/reports/RESULTS.md)).

## 3. Step 1 확인용 실행 (행 단위 분할)

| 설정 | 값 |
|------|------|
| 모델 | snunlp/KR-BERT-char16424 |
| Epoch | 2, Batch 32, Max length 64 |

| 지표 | 값 |
|------|----:|
| Test Accuracy | 0.7729 |
| Test F1-Macro | 0.7593 |
| Test F1-Weighted | 0.7647 |

감정별 F1과 test 행 가운데 train에 같은 문장이 있는 비율:

| 감정 | F1 | train과 겹치는 test 행 |
|------|---:|---:|
| 공포 | 0.954 | 100.0% |
| 혐오 | 0.909 | 99.8% |
| 놀람 | 0.867 | 98.7% |
| 행복 | 0.760 | 62.2% |
| 분노 | 0.734 | 82.2% |
| 슬픔 | 0.602 | 31.2% |
| 중립 | 0.489 | 11.2% |

결과 파일은 감정 이름이 1·2가 바뀐 채 저장되어 있어(파일 앞의 정정 메모 참고), 위 표에서는 행복과 슬픔을
바로잡았다. 겹침 비율은 jeju-emotion-lora의 누수 진단에서 옮겼다. 겹침이 많은 감정일수록 F1이 높고, 분노와
행복의 순서만 어긋난다. 감정별 점수 차이를 감정의 난이도로 읽기보다 test 문장을 train에서 이미 봤는지로
읽는 편이 맞다.

이 실행이 발표 자료의 Step 1(0.4733)보다 크게 높은 이유는 확인하지 못했다. 발표 때 Step 1이 어떤 데이터와
분할을 썼는지 남아 있지 않다.

## 4. Step 3 중간 기록

Dual-Gated KR-BERT를 행 단위 분할에서 돌리다 epoch 1 검증 결과까지만 기록하고 멈췄다.

| Epoch | val_acc | val_f1m |
|------:|--------:|--------:|
| 1 | 0.7194 | 0.7042 |

같은 시점 단일 KR-BERT의 검증 F1은 0.6973이었다. 한 epoch, 같은 누수 분할의 값이라 두 구조의 차이를
판단할 근거로 쓰지 않는다. 문장 단위 분할에서는 Dual-Gated KR-BERT(0.4806)가 KoELECTRA(0.4968)보다
낮았다.

## 5. 재현 방법

```bash
# 학습 파일은 저장소에 없다 (AI Hub 데이터에서 만든 파일)
DATA="merged_jeju_balanced_sorted.xlsx"

# Step 1: KR-BERT 베이스라인
python3 src/jeju_kr_bert.py --step baseline --data "$DATA"

# Step 3: Dual-Gated KR-BERT
python3 src/jeju_kr_bert.py --step dual --data "$DATA"

# KoELECTRA
python3 src/jeju_kr_bert.py --step koelectra --data "$DATA"

# Step 7: 앙상블 (두 모델의 test 확률로 α 그리드 서치)
python3 src/jeju_kr_bert.py --step ensemble \
  --dual-probs results_kr_bert/step3_dual_probs.npz \
  --single-probs results_kr_bert/branch1_koelectra_probs.npz

# 일괄 실행 (dual → koelectra → ensemble → 발표 수치와 비교 → hard-mining)
DATA="$DATA" bash scripts/run_goal.sh 6
```

위 명령은 모두 행 단위 분할을 쓴다. 문장 단위 분할로 다시 학습하는 방법은 jeju-emotion-lora의
`python -m jeju_lora.run_original --split sentence`에 있다.
