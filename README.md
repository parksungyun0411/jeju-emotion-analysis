# 제주어 다중감정분류 (Jeju Dialect Multi-class Emotion Classification)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-baseline-F7931E?logo=scikitlearn&logoColor=white)

제주어·표준어 병렬 코퍼스에 GPT-4o로 일곱 감정 라벨을 붙여 학습 데이터를 만들고, 고전 머신러닝 기준선과
KR-BERT·KoELECTRA 인코더 앙상블로 제주어 문장의 감정을 분류한 프로젝트다.

건국대학교 스마트ICT융합공학과 졸업 프로젝트(2025.09~2025.12)로 4인 팀이 진행했고, 그 가운데 기술 파트
(GPT-4o 라벨링 자동화, 인코더 앙상블)를 맡았다. 졸업 뒤의 재현 코드 작성과 재검증은 개인적으로 했다.

## 먼저 읽을 것: 발표 점수 0.84는 누수가 있는 분할의 값이다

- 발표 때의 앙상블 F1-Macro 0.84(0.8404)는 학습 파일을 행 단위로 7:1:2로 나눈 분할에서 잰 값이다.
  발표 자료에 적힌 분할은 학습 파일의 Stratified 7:1:2다. 발표 때의 시드는 남아 있지 않다. 재현 코드의
  절차(seed 42)로 나눈 행 단위 분할에서 다시 학습하면 앙상블이 0.8393으로 발표 값과 거의 같게 나온다.
- 학습 파일은 감정별 행 수를 맞추려고 같은 문장을 최대 56번 복제했다. 그래서 이 절차로 나누면 test
  25,462행 가운데 17,914행(70.4%)은 같은 제주어 문장이 train에도 있다. 공포, 혐오, 놀람은 test 행의
  98.7~100%가 train에 있다.
- 같은 데이터를 고유 문장 기준으로 다시 나누고(test 10,496문장, train과 겹침 0) 이 저장소의 재현 코드로
  다시 학습하면 앙상블 F1-Macro는 0.52(0.5192)다. 일반화 성능은 이 값으로 본다.
- 앙상블 가중치 α는 두 경우 모두 test 분할에서 골랐으므로 두 값 모두 그만큼 낙관적이다.
- 진단, 문장 단위 재학습, 소형 LLM과의 비교는 후속 저장소
  [jeju-emotion-lora](https://github.com/parksungyun0411/jeju-emotion-lora)에 있다
  ([누수 진단](https://github.com/parksungyun0411/jeju-emotion-lora/blob/main/reports/leakage.md),
  [결과 표](https://github.com/parksungyun0411/jeju-emotion-lora/blob/main/reports/RESULTS.md)).

## 감정 번호 1·2의 이름

학습 파일의 감정번호×gpt감정 교차표는 대각선만 채워져 있고, 1=행복, 2=슬픔이다. 졸업 프로젝트 때의 코드와
문서는 1=슬픔, 2=행복으로 적고 있었다. 2026년 10월에 이 저장소의 코드와 문서를 데이터에 맞게 고쳤다.
학습과 채점은 번호로 했으므로 전체 점수는 그대로이고, 감정별 보고와 앱에 나오는 이름만 바뀐다. 예전에 저장한
결과 파일은 원 출력을 고치지 않고 앞에 정정 메모를 붙였다.

## 결과

| 접근 | 데이터와 분할 | Test F1-Macro |
|------|---------------|--------------:|
| TF-IDF + 고전 ML 4종 중 최고 (SVM) | 최종데이터_균형.xlsx 12,950행, 행 단위 8:2 | 0.2968 |
| KR-BERT 단일 (재현 코드, 2 epoch 확인용) | 학습 파일, 행 단위 7:1:2 (누수) | 0.7593 |
| Dual-Gated KR-BERT + KoELECTRA 앙상블 (발표 자료) | 학습 파일, 행 단위 7:1:2 (누수) | 0.8404 |
| 같은 앙상블 (재현 코드) | 학습 파일, 행 단위 7:1:2 (누수) | 0.8393 |
| 같은 앙상블 (재현 코드) | 학습 파일, 문장 단위 (겹침 0) | 0.5192 |

- 0.7593은 [`results/step1_baseline_report.txt`](results/step1_baseline_report.txt), 0.8393과 0.5192는
  jeju-emotion-lora의 [reports/RESULTS.md](https://github.com/parksungyun0411/jeju-emotion-lora/blob/main/reports/RESULTS.md)에서 옮겼다.
- 행 단위 분할 안에서도 train에 같은 문장이 있는 test 행의 앙상블 정확도는 0.970, 없는 행은 0.561이다
  ([진단](https://github.com/parksungyun0411/jeju-emotion-lora/blob/main/reports/diagnostics/row_split_seen_unseen.md)).
  공포, 혐오, 놀람의 감정별 점수가 높게 나온 것은 이 겹침의 영향이 크다.
- 고전 ML과 딥러닝은 데이터 파일과 분할이 다르다. 고전 ML은 감정별 약 1,850행인 최종데이터_균형.xlsx를
  8:2로 나눴고, 딥러닝은 복제로 행 수를 늘린 학습 파일(merged_jeju_balanced_sorted.xlsx)을 썼다. 그래서
  0.30과 0.76·0.84의 차이를 모델 차이만으로 읽을 수 없다.

## 데이터

- 수집·정제: AI Hub 제주어·표준어 병렬 코퍼스(텍스트 파일 4,600여 개)와 한국어 단발성 대화 데이터셋.
- 라벨링: GPT-4o API로 문장마다 일곱 감정(중립, 행복, 슬픔, 분노, 놀람, 공포, 혐오) 가운데 하나를
  붙였다(`src/automation.py`). 이 스크립트의 원래 번호는 1=슬픔, 2=행복으로 최종 학습 파일과 반대였고,
  2026년 10월에 파일에 맞춰 고쳤다. 그래서 최종 파일을 만든 프롬프트와 똑같지는 않다. 후속 저장소에서
  Claude로 test 277문장을 따로 라벨링해 비교했을 때 GPT-4o 라벨과의 일치율은 40.1%였다. 라벨 경계가
  흔들린다는 점을 감안해서 점수를 읽어야 한다.
- 학습 파일: `merged_jeju_balanced_sorted.xlsx` 127,324행(빈 값 제거 후 127,307행). 고유 제주어 문장은
  53,662개다. 같은 문장에 라벨이 둘 이상 붙은 경우도 1,185문장(6,733행) 있다.
- 원본 데이터(AI Hub 라이선스), 학습 파일, 학습한 모델 가중치는 저장소에 없다. GPT-4o 라벨링 결과와
  병합·복제 절차도 남아 있지 않아서, AI Hub 원천만으로는 같은 학습 파일을 다시 만들 수 없다. 코드는 같은
  열 구성(`제주어 문장`, `표준어 문장`, `감정번호`)의 파일이 있을 때만 돌아간다. 고전 ML이 읽는
  최종데이터.xlsx와 최종데이터_균형.xlsx도 남아 있지 않다.

## 고전 ML 기준선

TF-IDF(1~3-gram, 최대 30,000 특성)로 Logistic Regression, SVM, Random Forest, Naive Bayes를 비교했다.
제주어 문장, 균형 데이터 기준 F1-Macro는 0.28~0.30이고 가장 높은 것은 SVM 0.2968이다
([`results/baseline_jeju.txt`](results/baseline_jeju.txt). 이 파일의 '최고 성능 모델' Naive Bayes는 정확도로
고른 것이다). 단어 빈도만 보는 방식으로는 방언 문맥을 읽기 어렵다고 보고 사전학습 인코더로 넘어갔다.

구현: `src/jeju_multilabel_emotion_model.py` (제주어), `src/multilabel_emotion_model.py` (표준어)

## 딥러닝 재현 코드

발표 때 점수를 만든 원 코드는 남아 있지 않다. `src/jeju_kr_bert.py`는 졸업 뒤(2026년 5월) 최종 발표 자료의
구조와 하이퍼파라미터를 보고 다시 짠 코드다. 발표 자료의 7단계와 이 코드의 상태는 다음과 같다.

| 단계 | 기법 | 발표 자료 F1* | 이 저장소의 상태 |
|------|------|--------------:|------------------|
| 1 | KR-BERT 베이스라인 | 0.4733 | 구현 (`--step baseline`) |
| 2 | + 데이터 밸런싱 | 0.5833 | 별도 처리 없음. 학습 파일이 이미 균형이라 baseline과 같은 학습 |
| 3 | + Dual-Gated KR-BERT | 0.6158 | 구현 (`--step dual`) |
| 4 | + 제주어 토크나이저 최적화 | 0.7184 | 미구현. `--step tokenizer`는 baseline과 같은 학습 |
| 5 | + DAPT (MLM 재학습) | 0.7456 | 미구현. `--step dapt`는 baseline과 같은 학습 |
| 6 | + Hard Mining | 0.8003 | 구현 (`--step hard-mining`). 앙상블에는 쓰지 않음 |
| 7 | + 앙상블 (KoELECTRA + Dual KR-BERT) | 0.8404 | 구현 (`--step koelectra`, `--step ensemble`) |

\* 발표 자료에 적힌 값이고, 모두 행 단위 분할(누수)에서 잰 값이다. 이 저장소 코드로 다시 돌렸을 때 발표 값과
비슷하게 나온 것은 앙상블(0.8393)뿐이다. Step 1(0.7593, 2 epoch)과 Step 3(0.8310)은 발표 값보다 크게 높게
나왔다. 이 저장소의 앙상블은 3단계 Dual-Gated KR-BERT와 KoELECTRA 두 모델만 결합하며, 발표 자료처럼
4~6단계를 누적한 모델이 아니다. 발표 자료의 'Noise filtering(신뢰도 하위 30% 제거)'도 구현하지 않았다.

### 앙상블 구조

```
제주어 문장 ──────────────► KoELECTRA ─► Classifier(7) ─► p_single
                                                              │
제주어 문장 ─┐                                                ▼
             ├─► Shared KR-BERT ─► [h_jeju], [h_std]     p_ens = α·p_dual + (1-α)·p_single
표준어 문장 ─┘        │
                      ▼
      Gate g = σ(W·[h_jeju; h_std] + b)
      h = g·h_jeju + (1-g)·h_std ─► Linear(7) ─► p_dual
```

- Branch 1, KoELECTRA: 제주어 문장만 입력한다.
- Branch 2, Dual-Gated KR-BERT: 제주어 문장과 표준어 문장을 같은 인코더로 처리하고, 게이트가 두 표현의
  비중을 학습한다.
- 앙상블: 두 브랜치 확률의 가중 결합. α는 test 분할에서 그리드 서치로 고른다.

## 리포지토리 구조

```
.
├── src/
│   ├── automation.py                     # GPT-4o 감정 라벨링 자동화
│   ├── jeju_multilabel_emotion_model.py  # 고전 ML 기준선 (제주어)
│   ├── multilabel_emotion_model.py       # 고전 ML 기준선 (표준어)
│   ├── jeju_kr_bert.py                   # 딥러닝 재현 코드 (--step으로 단계 선택)
│   ├── ensemble_search.py                # 여러 모델의 가중 앙상블 탐색
│   ├── verify_goal.py                    # 결과를 발표 자료 수치와 비교
│   ├── translation/                      # [졸업 뒤 개인 확장] KoBART 제주어↔표준어 번역
│   └── scripts/                          # 데이터 검증 유틸
├── app/                                  # [졸업 뒤 개인 확장] FastAPI 추론 서비스
├── webapp/                               # [졸업 뒤 개인 확장] Next.js 웹앱(코드상 이름 제주올)
├── docs/                                 # 2025년 실험 문서, 재현 기록, 확장 기획 문서
├── results/                              # 고전 ML 결과와 Step 1 확인용 실행 결과
├── scripts/                              # 학습 파이프라인 실행 스크립트
└── requirements.txt
```

발표 PPT와 최종 보고서는 저장소에 넣지 않았다.

## 실행 방법

```bash
pip install -r requirements.txt
```

```bash
# 고전 ML 기준선 (최종데이터.xlsx, 최종데이터_균형.xlsx가 실행 폴더에 있어야 한다. 두 파일은 남아 있지 않다)
python src/jeju_multilabel_emotion_model.py

# 딥러닝 재현 코드, 단계별 실행 (--data로 학습 파일 경로 지정)
python src/jeju_kr_bert.py --step dual --data <xlsx 경로>
python src/jeju_kr_bert.py --step koelectra --data <xlsx 경로>
python src/jeju_kr_bert.py --step ensemble \
  --dual-probs results_kr_bert/step3_dual_probs.npz \
  --single-probs results_kr_bert/branch1_koelectra_probs.npz

# 일괄 실행 (dual → koelectra → ensemble → 발표 수치와 비교 → hard-mining)
DATA=<xlsx 경로> bash scripts/run_goal.sh 6
```

- 위 명령은 행 단위 분할을 쓴다. 문장 단위 분할로 다시 학습하는 방법은 jeju-emotion-lora에 있다.
- GPT-4o 라벨링(`src/automation.py`)은 환경변수 `OPENAI_API_KEY`가 필요하다. API 호출이나 응답 해석에
  실패한 행은 -1(분류실패)로 남는다. 2026년 10월 이전 판은 실패한 행을 중립(0)으로 저장했고, 그 판으로 만든
  라벨에 이런 행이 얼마나 섞였는지는 확인할 수 없다.
- Apple Silicon(MPS)에서는 `--fp16`으로 bf16 autocast를 켤 수 있다.

## 졸업 뒤 개인 확장: 탐라(Tamna) 앱

졸업 뒤 2026년 5월에 개인적으로 덧붙인 코드이고, 졸업 프로젝트나 팀의 산출물이 아니다. 학습된 번역·감정
모델 가중치가 저장소에 없어서 그대로 띄우면 번역과 감정 기능은 503 안내만 낸다. 앱 이름은 기획서에서 탐라로
정했지만, 웹앱 코드의 제목(webapp/README.md, 브라우저 제목)에는 가제 제주올이 그대로 남아 있다.

- `src/translation/`: KoBART 제주어↔표준어 번역 학습 코드와 사전 폴백. 번역 모델의 학습·평가 결과는
  저장소에 없다. 학습 코드도 행 단위로 나누므로, 돌려서 얻는 BLEU는 같은 누수의 영향을 받는다.
- `app/`: FastAPI 추론 서비스([app/README.md](app/README.md)). Dual KR-BERT는 학습 때 표준어 입력 자리에
  사람이 쓴 표준어 문장을 받았지만, 앱은 번역 결과(없으면 제주어 문장)를 넣는다. 학습 조건과 달라서 위
  점수는 앱의 출력에 적용되지 않는다.
- `webapp/`: Next.js 15 4탭 모바일 웹앱([webapp/README.md](webapp/README.md)).

기획·설계 문서: [기획서](docs/기획서.md), [개발설계서](docs/개발설계서.md). 둘 다 이 확장 단계의 문서다.

## 문서

| 문서 | 내용 |
|------|------|
| [`docs/bert_improvement_report.md`](docs/bert_improvement_report.md) | 재현 코드 작성 기록(2026년 5월)과 단계별 상태 |
| [`docs/experiment_design.txt`](docs/experiment_design.txt) | 고전 ML 실험 설계 (2025년 원 문서, 앞에 정정 메모) |
| [`docs/model_documentation.txt`](docs/model_documentation.txt) | 고전 ML 모델 설명 (2025년 원 문서, 앞에 정정 메모) |
| [`docs/f1_weighted_analysis.txt`](docs/f1_weighted_analysis.txt) | F1 지표 선택 근거 (2025년 원 문서, 앞에 정정 메모) |
| [`docs/tfidf_explanation.txt`](docs/tfidf_explanation.txt) | TF-IDF 원리 정리 (2025년 원 문서, 앞에 정정 메모) |
| [`results/`](results/) | 고전 ML 결과와 Step 1 확인용 실행 결과 원본 (앞에 정정 메모) |

## 기술 스택

Python, PyTorch, HuggingFace Transformers, scikit-learn, KR-BERT, KoELECTRA, OpenAI GPT-4o API, pandas, numpy, seaborn

## 라이선스

학술·포트폴리오 공개 목적. 학습 데이터의 라이선스는 원본 제공처(AI Hub)를 따른다.
