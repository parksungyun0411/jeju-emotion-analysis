# 제주어 번역·감정분석 챗봇 — 마스터 설계 문서

> 작성 2026-05-25. 졸업프로젝트 확장: 기존 "제주어 감정분류"에 **제주어↔표준어 번역**과
> **대화형 챗봇**을 더해 통합 제품으로 만든다.
>
> ⚠️ **승인 상태:** 사용자가 자리를 비운 상태에서 "네가 효율적으로 결정해 진행" 지시를 받아
> 작성한 자율 설계다. 아래 결정들은 합리적 기본값이며, 사용자 복귀 시 검토·수정 가능하도록
> 모듈식으로 구현한다. (참고: [[autonomy-grant]])

## 1. 제품 개요 (기획서 요약)

**한 줄:** 제주어 문장을 입력하면 **표준어로 번역**해 주고 **감정을 분석**해 주는, 그리고
반대로 **표준어를 제주어로** 바꿔 주는 대화형 챗봇.

**배경/문제:** 제주어는 유네스코 소멸위기언어. 세대 단절로 이해·학습 수단이 부족하다.
기존 1학기(번역기 제안)·2학기(감정분류) 성과를 하나의 사용 가능한 제품으로 통합한다.

**대상 사용자:** 제주어 학습자, 제주 출신 젊은 세대, 콘텐츠/관광 종사자, 언어 연구자.

**핵심 기능 (MVP):**
1. 제주어 → 표준어 번역
2. 표준어 → 제주어 번역 (양방향)
3. 제주어 입력 감정 분석 (7종: 중립/슬픔/행복/분노/놀람/공포/혐오) + 신뢰도
4. 위 기능을 한 화면에서 쓰는 채팅 UI

**비목표(YAGNI):** 음성 입출력(STT/TTS), 로그인/계정, 멀티턴 문맥 기억, 모바일 앱.
모두 향후 과제로 두고 MVP에서 제외.

**성공 지표:**
- 감정 분류: Test F1-Macro **0.84**(발표본 재현, 진행 중)
- 번역: 테스트셋 **BLEU/chrF** 측정·보고 (베이스라인 대비 향상 입증)
- 데모: 제주어 입력 → 번역+감정이 한 화면에서 동작하는 end-to-end 데모

## 2. 시스템 아키텍처

```
[브라우저 채팅 UI]  ──HTTP(JSON)──>  [FastAPI 백엔드]
   정적 HTML/JS                         ├─ EmotionService (앙상블 추론)
   방향 토글, 감정 뱃지                   ├─ TranslationService (KoBART 추론)
                                        └─ /api/translate /api/emotion /api/chat
                                              │
                                   [학습된 모델 아티팩트]
                                     - 감정: Dual-Gated KR-BERT(state) + KoELECTRA
                                     - 번역: KoBART(제주어↔표준어, 방향 prefix)
```

**3개 서브시스템 (각자 독립 개발·테스트):**
- **A. 감정분류** (기존, 진행 중) — `src/jeju_kr_bert.py`. 목표 F1 0.84. 산출물: 모델 가중치 + probs.
- **B. 번역** (신규) — `src/translation/`. KoBART seq2seq.
- **C. 챗봇 앱** (신규) — `app/`. A·B를 추론 서비스로 감싸 REST + UI 제공.

## 3. 서브시스템 B — 번역 모델

**접근:** 사전학습 한국어 seq2seq **KoBART**(`gogamza/kobart-base-v2`)를 병렬 코퍼스로 파인튜닝.
이유: 한국어 BART라 제주어↔표준어처럼 표면형이 비슷한 변환에 적합, 124M로 M4에서 학습 현실적.

**양방향 단일 모델 + 방향 prefix:** 한 모델로 두 방향을 학습한다. 소스 앞에 방향 토큰을 붙인다.
- 제주어→표준어: 입력 `"표준어로: <제주어문장>"` → 타깃 `<표준어문장>`
- 표준어→제주어: 입력 `"제주어로: <표준어문장>"` → 타깃 `<제주어문장>`
각 병렬쌍이 양방향 2개 샘플을 만들어 데이터가 2배가 되고 표현을 공유한다.

**데이터:** `merged_jeju_balanced_sorted.xlsx`의 `제주어 문장`/`표준어 문장` 컬럼(127k쌍).
seed=42 고정, 8:1:1 분할(번역은 감정과 분할 정합성 불필요).

**평가:** `sacrebleu`로 BLEU + chrF. 베이스라인(무번역 copy, 또는 짧은 학습) 대비 보고.

**구현:** `src/translation/train_translation.py` (`--direction both` 기본, `--epochs` 등 CLI),
`src/translation/translate.py` (추론 래퍼: `translate(text, direction)`),
`scripts/run_translation.sh` (학습 러너, **감정 goal run 종료 후** 실행 — 단일 GPU 경합 방지).

## 4. 서브시스템 C — 챗봇 앱

**백엔드:** FastAPI (`app/main.py`). 시작 시 모델 1회 로드(MPS/CPU 자동).
- `POST /api/translate` `{text, direction:"j2s"|"s2j"}` → `{translation}`
- `POST /api/emotion`   `{text}` → `{label, label_id, scores[7]}`
- `POST /api/chat`      `{text, direction?}` → 입력을 제주어로 보고 `{translation(표준어), emotion:{label,scores}}`;
  `direction="s2j"`면 표준어→제주어 번역만.
- `GET /` → 정적 프론트.

**추론 서비스 (의존성 주입, 테스트 가능하게 분리):**
- `app/services/emotion_service.py` — Dual-Gated KR-BERT(state_dict) + KoELECTRA 로드, softmax 평균(학습된 α) 앙상블. 모델 없으면 명확한 에러.
- `app/services/translation_service.py` — KoBART 로드, beam search.
- 모델 아티팩트 경로는 환경변수/설정으로 주입(없을 때 graceful degradation: 가능한 기능만 노출).

**프론트:** 빌드 단계 없는 단일 페이지(`app/static/index.html` + `app.js` + `style.css`).
채팅 말풍선, 방향 토글, 감정 뱃지(감정별 이모지+색), 신뢰도 막대. 가볍고 즉시 실행 가능.

**실행:** `uvicorn app.main:app`. README에 절차 명시.

## 5. 모델 아티팩트 — 학습 코드 보강 필요

현재 `run_step_single`(KoELECTRA/baseline/hard-mining)은 **모델을 저장하지 않고** probs/report만 남긴다.
챗봇 앙상블 서빙에는 KoELECTRA 가중치가 필요하므로, `run_step_single`에 `model.save_pretrained` +
토크나이저 저장을 추가한다. (dual은 이미 state_dict 저장.) KoELECTRA 단계는 dual 학습 뒤
별도 프로세스로 실행되므로 **지금 소스를 고치면 그때 반영**된다. 앙상블 α는 ensemble 단계에서
선택된 값을 report에서 읽거나 별도 저장.

## 6. 저장소 구조 (기존 repo 확장)

```
jeju-emotion-analysis/
  src/                      # 감정 (기존) + translation/ (신규)
  app/                      # FastAPI 백엔드 + 프론트 (신규)
  docs/
    기획서.md               # (신규) 한글 기획 산출물
    개발설계서.md            # (신규) 한글 개발설계 산출물
    superpowers/specs/      # 본 마스터 설계 + 향후 spec
  scripts/                  # run_goal.sh(감정) + run_translation.sh(번역)
```
repo 이름(`jeju-emotion-analysis`)은 범위가 넓어졌으나 유지하고 README에 확장 범위 명시(이름 변경은 추후).

## 7. 작업 순서 / GPU 직렬화

단일 MPS GPU. **GPU 작업은 직렬**: ① 감정 goal run(진행 중, ~6h) → ② 번역 KoBART 학습.
**비-GPU 작업은 지금 병렬**(에이전트): 기획서·개발설계서 문서화, 번역/앱 코드 스캐폴드, 모델저장 코드 보강.

1. (지금) 문서 2종 + 번역 코드 + 앱 스캐폴드 + 모델저장 보강 — 병렬
2. 감정 goal run 완료 → 0.84 검증
3. 번역 학습 실행 → BLEU 평가
4. 두 모델을 앱에 연결 → end-to-end 데모
5. 리포트/문서 갱신, 브랜치 푸시/PR

## 8. 리스크

- KoBART MPS 학습 시간: 127k×2 샘플 → 에폭 길 수 있음. 동적 패딩·적정 batch·필요시 부분집합으로 관리.
- 앙상블 서빙 메모리: KR-BERT+KoELECTRA+KoBART 동시 로드. 16GB 내 관리, 필요시 지연 로드.
- 번역 품질: 짧은 발화 위주라 BLEU 변동 큼 → chrF 병행 보고.
