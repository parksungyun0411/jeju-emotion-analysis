# 제주어 다중감정분류 (Jeju Dialect Multi-class Emotion Classification)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-baseline-F7931E?logo=scikitlearn&logoColor=white)
![F1](https://img.shields.io/badge/F1--Macro-0.84-blue)

> 제주어/표준어 병렬 코퍼스를 GPT-4o로 7감정 라벨링해 데이터를 직접 구축하고,
> 고전 ML의 한계(F1-Macro ≈ 0.30)를 규명한 뒤 **Dual-Gated KR-BERT + KoELECTRA 앙상블**의
> 7단계 딥러닝 파이프라인으로 **F1-Macro 0.84**를 달성한 방언 감정 분류 NLP 프로젝트.

건국대학교 스마트ICT융합공학과 졸업 프로젝트 (2025년 가을학기) —
4인 팀 프로젝트 — 데이터 수집·라벨링·모델 설계·실험·성능 분석 담당.

## 핵심 성과 한눈에

| 접근 | 모델 | Test F1-Macro |
|------|------|--------------:|
| 고전 ML 베이스라인 | TF-IDF + Naive Bayes (4개 분류기 중 최고) | ≈ 0.30 (0.2957) |
| 딥러닝 베이스라인 | KR-BERT 단일 모델 | 0.76 (0.7593, Acc 0.7729) |
| **최종** | **Dual-Gated KR-BERT + KoELECTRA 앙상블** | **0.84 (0.8404)** |

강감정(공포·혐오·놀람)에서는 F1 0.87~0.95를 기록.

## 문제 정의

제주어는 유네스코 지정 소멸위기 언어로, **라벨링된 감정 분석 데이터가 존재하지 않는다.**
표준어용 감정 모델은 제주어 특유의 어휘·어미를 처리하지 못하므로,
데이터 구축부터 모델 설계까지 처음부터 해결해야 했다.

## 1. 데이터 구축 — GPT-4o 자동 라벨링 파이프라인

- **수집·정제**: AI Hub 제주도 학습용 데이터(제주어/표준어 병렬 코퍼스, 텍스트 파일 4,600여 개) + 한국어 단발성 대화 데이터셋
- **라벨링**: GPT-4o API로 7가지 감정(중립·기쁨·슬픔·분노·놀람·공포·혐오) 자동 분류 파이프라인 구축 (`src/automation.py`), 수동 검증 병행
- **최종 학습 데이터**: 127,324행, 7클래스 균형, Stratified 7:1:2 분할 (seed 고정 → 모든 실험이 동일 test set 공유)

## 2. 고전 ML 베이스라인 — 한계 규명

TF-IDF + N-gram(1~3) 특성으로 Logistic Regression · SVM · Random Forest · Naive Bayes를 비교.
제주어 균형 데이터 기준 최고 **F1-Macro 0.2957 (Naive Bayes)** 에 그쳐,
표면 통계 기반 전통 ML로는 방언 감정 분류가 어렵다는 것을 정량적으로 확인 → 딥러닝 전환의 근거.

구현: `src/jeju_multilabel_emotion_model.py` (제주어), `src/multilabel_emotion_model.py` (표준어 비교), 결과: `results/`

## 3. 딥러닝 7단계 최적화 파이프라인

단일 KR-BERT 베이스라인에서 출발해 단계별 개선을 누적, F1-Macro 0.84까지 도달했다.

| 단계 | 기법 | Test F1-Macro* |
|------|------|---------------:|
| 1 | KR-BERT 베이스라인 (전처리 없음) | 0.4733 |
| 2 | + 데이터 밸런싱 (오버샘플링) | 0.5833 |
| 3 | + Dual-Gated KR-BERT (제주어/표준어 shared 인코더 + 게이팅 융합) | 0.6158 |
| 4 | + 제주어 토크나이저 최적화 (제주어 토큰 추가) | 0.7184 |
| 5 | + DAPT (제주어 MLM 재학습) | 0.7456 |
| 6 | + Hard Mining (loss 상위 20% 가중치) | 0.8003 |
| 7 | + 앙상블 (KoELECTRA + Dual KR-BERT) | **0.8404** |

\* 최종 발표 실험 기준 수치. 본 리포의 재현 코드(`src/jeju_kr_bert.py`)는 이미 밸런싱·정제된
데이터를 사용하므로 Step 1 재현치가 0.7593으로 더 높게 시작한다 (Step 1~2 효과가 데이터에
선반영). 재현 로그·단계별 분석은 [`docs/bert_improvement_report.md`](docs/bert_improvement_report.md) 참조.

## 최종 아키텍처

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

- **Branch 1 — KoELECTRA**: 제주어 단독 입력
- **Branch 2 — Dual-Gated KR-BERT**: 제주어/표준어 병렬 입력을 shared 인코더로 처리하고, 게이트가 두 표현의 기여도를 학습
- **앙상블**: 두 브랜치 확률의 가중 결합 (α는 그리드 서치로 탐색)

## 리포지토리 구조

```
.
├── src/
│   ├── automation.py                     # GPT-4o 감정 라벨링 자동화
│   ├── jeju_multilabel_emotion_model.py  # 고전 ML 베이스라인 (제주어)
│   ├── multilabel_emotion_model.py       # 고전 ML 베이스라인 (표준어)
│   ├── jeju_kr_bert.py                   # 딥러닝 7단계 파이프라인 (--step 인자로 단계 선택)
│   ├── ensemble_search.py                # 다중 모델 가중 앙상블 탐색
│   ├── verify_goal.py                    # 목표 성능(0.84) 자동 검증
│   ├── translation/                      # [확장] KoBART 제주어↔표준어 번역
│   └── scripts/                          # 데이터 검증 유틸
├── app/                                  # [확장] FastAPI ML 서비스 (감정+번역 추론 API)
├── webapp/                               # [확장] 탐라(Tamna) Next.js 웹앱
├── docs/                                 # 실험 문서·기획서·개발설계서
├── results/                              # 고전 ML·Step 1 실험 결과 리포트
├── scripts/                              # 학습 파이프라인 실행 스크립트
└── requirements.txt
```

**포함 범위 (정직 고지)**: 라벨링 자동화·고전 ML·딥러닝 7단계 전체 코드와 실험 문서는 본 리포에
포함되어 있다. 반면 **원본 데이터(AI Hub 라이선스)와 학습된 모델 가중치(`results_kr_bert/` 등)는
`.gitignore`로 제외**되어 있어, 재현하려면 AI Hub에서 데이터를 내려받아 직접 학습해야 한다.
발표 PPT·최종 보고서는 별도 보관.

## 실행 방법

```bash
pip install -r requirements.txt
```

```bash
# 고전 ML 베이스라인
python src/jeju_multilabel_emotion_model.py

# 딥러닝 파이프라인 — 단계별 실행 (--data 로 학습 데이터 경로 지정)
python src/jeju_kr_bert.py --step baseline --data <xlsx 경로>
python src/jeju_kr_bert.py --step dual --data <xlsx 경로>
python src/jeju_kr_bert.py --step koelectra --data <xlsx 경로>
python src/jeju_kr_bert.py --step ensemble \
  --dual-probs results_kr_bert/step3_dual_probs.npz \
  --single-probs results_kr_bert/branch1_koelectra_probs.npz

# 전체 파이프라인 일괄 실행 (dual → koelectra → ensemble → 목표검증 → hard-mining)
bash scripts/run_goal.sh 6
```

- 학습 데이터는 [AI Hub](https://www.aihub.or.kr/)에서 다운로드 후 로컬 배치 (리포 미포함)
- GPT-4o 라벨링(`src/automation.py`)은 `src/config.py`에 OpenAI API 키 설정 필요
- Apple Silicon(MPS)에서 bf16 mixed precision으로 학습 가속 지원

## 후속 확장 — 탐라(Tamna) 앱

졸업 프로젝트의 감정 분류·번역 모델을 실제 서비스 형태로 확장한 코드가 함께 들어 있다:

- **`src/translation/`** — KoBART 기반 제주어↔표준어 양방향 번역 학습 + 제주어-표준어 사전 폴백
- **`app/`** — FastAPI ML 서비스: 감정 앙상블 + 번역 추론 API (지연 로드, graceful degradation) → [app/README.md](app/README.md)
- **`webapp/`** — Next.js 15 4탭 모바일 웹앱 (제주어 번역·감정 / 맛집 지도 / 명소·AI 여행경로 / 커뮤니티) → [webapp/README.md](webapp/README.md)

기획·설계 문서: [기획서](docs/기획서.md) · [개발설계서](docs/개발설계서.md)

## 문서

| 문서 | 내용 |
|------|------|
| [`docs/bert_improvement_report.md`](docs/bert_improvement_report.md) | 7단계 파이프라인 재현 보고서 (단계별 성능·분석) |
| [`docs/experiment_design.txt`](docs/experiment_design.txt) | 실험 설계 상세 |
| [`docs/model_documentation.txt`](docs/model_documentation.txt) | 모델 설명서 |
| [`docs/f1_weighted_analysis.txt`](docs/f1_weighted_analysis.txt) | F1 지표 선택 근거 분석 |
| [`docs/tfidf_explanation.txt`](docs/tfidf_explanation.txt) | TF-IDF 원리 정리 |
| [`results/`](results/) | 고전 ML 베이스라인·Step 1 실험 결과 원본 |

## 기술 스택

Python · PyTorch · HuggingFace Transformers · scikit-learn · KR-BERT · KoELECTRA · OpenAI GPT-4o API · pandas · numpy · seaborn

## 라이선스

학술/포트폴리오 공개 목적. 학습 데이터의 라이선스는 원본 제공처(AI Hub)를 따름.
