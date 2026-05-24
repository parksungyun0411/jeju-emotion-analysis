# 제주어 다중감정분류 (Jeju Dialect Multi-label Emotion Classification)

건국대학교 스마트ICT융합공학과 졸업 프로젝트 (2025년 가을학기)

GPT-4o로 라벨링한 제주어/표준어 병렬 데이터를 기반으로, TF-IDF + N-gram 특성과 고전 머신러닝 분류기(Logistic Regression, SVM, Random Forest, Naive Bayes)를 사용해 7가지 감정(중립, 기쁨, 슬픔, 분노, 놀람, 공포, 혐오)을 분류하는 모델.

## 프로젝트 개요

- **목표**: 제주어 텍스트의 감정 분류 모델 개발 및 표준어 베이스라인과의 성능 비교
- **데이터**: AI Hub 제주도 학습용 데이터(제주어/표준어 병렬 코퍼스) + 한국어 단발성 대화 데이터셋
- **라벨링**: GPT-4o로 7가지 감정 자동 분류 (수동 검증)
- **모델링**: TF-IDF (단어/2-gram/3-gram) → 4개 분류기 비교
- **평가**: F1-Weighted, Accuracy, 감정별 상세 성능 분석

## 기술 스택

- Python 3.x
- scikit-learn (TF-IDF, LR, SVM, RF, NB)
- pandas, numpy, openpyxl
- OpenAI API (GPT-4o, 라벨링 단계)
- seaborn, matplotlib (시각화)

## 디렉토리 구조

```
.
├── src/
│   ├── automation.py                       # 데이터 라벨링 자동화 (GPT-4o 호출)
│   ├── jeju_multilabel_emotion_model.py    # 제주어 감정분류 메인 모델
│   ├── multilabel_emotion_model.py         # 표준어 베이스라인 모델
│   ├── multi_sentence_handler*.py          # 다중 문장 처리 (간단/완전 버전)
│   ├── emotion_detail_analysis.py          # 감정별 상세 성능 분석
│   ├── baseline_summary.py                 # 베이스라인 종합 결과
│   ├── baseline_integrated.py              # 통합 데이터 베이스라인
│   ├── test_sentence_split.py              # 문장 분리 테스트
│   ├── check_files.py                      # 데이터 파일 검증
│   ├── check_file_structure.py             # 파일 구조 검증
│   └── config.py                           # API 키 설정 템플릿
├── docs/
│   ├── model_documentation.{py,txt}        # 모델 설명서
│   ├── experiment_design.{py,txt}          # 실험 설계 상세
│   ├── f1_weighted_analysis.{py,txt}       # F1 가중치 분석
│   └── tfidf_explanation.{py,txt}          # TF-IDF 원리 설명
├── results/
│   ├── baseline_jeju.txt                   # 제주어 베이스라인 결과
│   ├── baseline_integrated.txt             # 통합 데이터 베이스라인
│   ├── baseline_summary.txt                # 종합 베이스라인
│   └── baseline_20250929.txt               # 9/29자 실험 결과
└── requirements.txt
```

## 설치 및 실행

```bash
pip install -r requirements.txt
```

`src/config.py`에 OpenAI API 키 설정:
```python
OPENAI_API_KEY = "your_api_key_here"
```

데이터 라벨링 자동화:
```bash
python src/automation.py
```

제주어 감정분류 모델 학습/평가:
```bash
python src/jeju_multilabel_emotion_model.py
```

## 감정 분류 라벨

| 번호 | 감정 |
|------|------|
| 0 | 중립 |
| 1 | 기쁨 |
| 2 | 슬픔 |
| 3 | 분노 |
| 4 | 놀람 |
| 5 | 공포 |
| 6 | 혐오 |

## 참고

- 대용량 데이터 파일(`*.xlsx`, 학습용 텍스트 데이터 4,600여 개)은 저장소에서 제외됨
- 학습 데이터는 [AI Hub 제주어 데이터](https://www.aihub.or.kr/)에서 다운로드 후 `제주도_학습용데이터_1/` 위치에 배치
- 실험 결과 PPT, 보고서는 `docs/` 외 별도 보관

## 라이선스

학술/포트폴리오 공개 목적. 학습 데이터의 라이선스는 원본 제공처를 따름.
