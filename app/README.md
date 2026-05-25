# 제주어 챗봇 앱 (서브시스템 C)

제주어↔표준어 **번역**과 제주어 **감정 분석**을 하나의 채팅 UI로 제공하는
FastAPI 백엔드 + 순수 정적 프론트엔드.

설계 근거: `docs/superpowers/specs/2026-05-25-jeju-chatbot-design.md` §4, §5.

## 구조

```
app/
  main.py                     # FastAPI 앱 (엔드포인트 + 정적 서빙)
  services/
    emotion_service.py        # Dual-Gated KR-BERT + KoELECTRA 앙상블 (지연 로드)
    translation_service.py    # KoBART 양방향 번역 래퍼 (지연 로드)
  static/
    index.html  app.js  style.css   # 빌드 불필요 채팅 UI
```

## 실행

```bash
# 프로젝트 루트에서
uvicorn app.main:app --reload
```

기본 주소: http://127.0.0.1:8000/ (채팅 UI), API 문서: http://127.0.0.1:8000/docs

## 의존성

```bash
pip install -r requirements.txt
```

- `fastapi`, `uvicorn[standard]` — 웹 서버 (requirements.txt 에 포함).
- 추론 시점에만 필요(지연 로드): `torch`, `transformers`, 그리고 번역 래퍼의
  의존성. 모델 아티팩트가 없으면 이 라이브러리들은 import 되지 않으며,
  앱은 정상 기동한다.

## 모델 아티팩트 경로

서비스는 **지연 로드**(lazy)다. 앱 시작 시 모델을 올리지 않고, 첫 요청 때 올린다.
아티팩트가 없으면 해당 기능은 `available=false` 가 되고, 호출 시 `503` 과 안내
메시지를 반환한다 (graceful degradation — 앱 자체는 항상 뜬다).

| 서비스 | 기본 경로 | 필요한 산출물 |
|--------|-----------|----------------|
| 감정 (EmotionService) | `results_kr_bert/` | `step3_dual_state.pt` (Dual KR-BERT state_dict), `branch1_koelectra_model/` (KoELECTRA `save_pretrained`), `step7_ensemble_report.txt` (α 파싱, 없으면 0.5) |
| 번역 (TranslationService) | `results_translation/kobart_jeju/` | KoBART 파인튜닝 모델 (+ `src/translation/translate.py` 의 `Translator`) |

경로는 `EmotionService(artifacts_dir=...)` / `TranslationService(artifacts_dir=...)`
인자로 주입할 수 있다 (의존성 주입, 테스트 용이).

### 감정 앙상블

```
p_ens = α · p_dual + (1 - α) · p_single
```

- `p_dual`: Dual-Gated KR-BERT. 추론 시 표준어 번역 입력이 없으므로 제주어 문장을
  제주어/표준어 양쪽 슬롯에 동일하게 넣는다 (게이트가 자체 조정).
- `p_single`: KoELECTRA.
- `α`: `step7_ensemble_report.txt` 에서 파싱. 파일이 없으면 0.5.

`DualGatedClassifier`, `EMOTION_LABELS`, `get_device` 는 `src/jeju_kr_bert.py` 에서
재사용한다 (서비스가 `src/` 를 `sys.path` 에 추가).

## 엔드포인트

| 메서드 | 경로 | 요청 | 응답 |
|--------|------|------|------|
| GET  | `/api/health`    | — | 각 서비스 `available`/상세 상태 |
| POST | `/api/translate` | `{text, direction:"j2s"\|"s2j"}` | `{translation}` (unavailable → 503) |
| POST | `/api/emotion`   | `{text}` | `{label, label_id, scores}` (unavailable → 503) |
| POST | `/api/chat`      | `{text, direction?}` | j2s(기본): `{translation, emotion}` 동시 / s2j: 번역만 |
| GET  | `/`              | — | 채팅 UI (`static/index.html`) |

`/api/chat` 의 감정 분석은 `direction="j2s"` 일 때만 수행하며, 감정 모델이
unavailable 이어도 번역 결과는 반환한다 (감정 필드만 `null`).

## 감정 라벨 / 이모지

`0 중립😐 / 1 슬픔😢 / 2 행복😊 / 3 분노😠 / 4 놀람😮 / 5 공포😱 / 6 혐오🤢`
(순서는 `jeju_kr_bert.EMOTION_LABELS` 와 일치.)
