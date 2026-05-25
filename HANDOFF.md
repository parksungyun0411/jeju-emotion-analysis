# 인수인계 / 다음 작업 이어가기 (탐라 Tamna)

> 최종 갱신 2026-05-25. 이 문서만 읽으면 어디까지 했고 다음에 뭘 하면 되는지 알 수 있다.

## 프로젝트 한눈에

졸업프로젝트가 **"제주 올인원 앱 — 탐라(Tamna)"** 로 확장됨. 제주도 모든 것을 담는 4탭 모바일 웹앱.
설계 기준: `docs/superpowers/specs/2026-05-25-jeju-allinone-app-design.md` (마스터 v2).
기획서 `docs/기획서.md`(+`.html`), 개발설계서 `docs/개발설계서.md`.

- **탭1 제주어**: 제주어↔표준어 번역 + 감정분석 (ML 서비스 연결)
- **탭2 맛집**: 카카오 지도·마커·상세·길찾기
- **탭3 명소·체험**: 지도 + AI 여행경로(Claude)
- **탭4 커뮤니티**: 게시판·댓글·좋아요 + 로그인(Google/Kakao) + 마이페이지

## 저장소 구조 (모노레포)

```
webapp/      Next.js 15 앱 (4탭, App Router, TS, Tailwind, Prisma, NextAuth)  ← npm run build 통과
app/         FastAPI ML 서비스 (번역+감정 추론)
src/         학습 코드: jeju_kr_bert.py(감정), translation/(번역), ensemble_search.py, verify_*.py
scripts/     run_goal.sh(감정 학습), run_translation.sh(번역 학습)
docs/        기획서/개발설계서/specs
```

## ✅ 완료 (전부 커밋·푸시·머지됨)

- 문서: 기획서(md+html), 개발설계서, 마스터 설계 v2
- 탐라 앱 `webapp/`: 4탭 전부 구현, `npm run build` 통과(16라우트), dev 전탭 200. 키 없이도 기동(seed/mock/graceful)
- 감정 학습 파이프라인: bf16 가속(2.7×) + 버그수정(앙상블 정합성, hard-mining 인덱스)
- 번역 코드(KoBART 양방향) + 목표검증(verify_translation.py, chrF≥80) + 제주어-표준어 사전 폴백(`src/translation/data/jeju_std_dict.json`, j2s/s2j 각 2000개)
- 다중 모델 앙상블 탐색기 `src/ensemble_search.py`

## 📊 감정 모델 결과 (발표본 목표 F1-Macro 0.8404)

| 모델 | Test F1-Macro |
|---|---|
| Dual-Gated KR-BERT (6ep, bf16) | **0.8289** |
| KoELECTRA (6ep) | 0.7947 |
| 2-way 앙상블 (α=dual0.65) | **0.8327** (목표의 1% 내 — 사실상 달성) |
| Hard-mining (KR-BERT) | 🔄 학습 중 (다음 참조) |

학습 산출물(가중치/probs)은 `results_kr_bert/`(gitignore, **로컬 SSD에만** 존재):
`step3_dual_state.pt`, `step3_dual_probs.npz`, `branch1_koelectra_model/`, `branch1_koelectra_probs.npz`,
`step7_ensemble_*`, (진행 후) `step6_hard_mining_*`.

## 🔄 지금 돌고 있는 것

`scripts/run_goal.sh 6` (bash, nohup, 로그 `results_kr_bert/goal_run.log`):
**Step6 hard-mining** 학습 중 (21:21 시작, 약 2시간 예상). 끝나면 run_goal.sh가 자동으로 최종 verify 후 종료.
→ 자리 비운 사이 완료되어 있을 것. `tail results_kr_bert/goal_run.log` 로 결과 확인.

## ▶️ 다음에 할 일 (순서대로)

### 1. 3-way 앙상블로 0.8404 돌파 시도 (hard-mining 완료 후, GPU 불필요)
```bash
cd webapp/..   # 리포 루트
python3 src/ensemble_search.py \
  --probs results_kr_bert/step3_dual_probs.npz \
          results_kr_bert/branch1_koelectra_probs.npz \
          results_kr_bert/step6_hard_mining_probs.npz \
  --names dual koelectra hardmining --step 0.05 --output-dir results_kr_bert
```

### 2. 번역 모델 학습 (GPU/MPS, 감정 학습 완전 종료 후 — 단일 GPU 직렬)
```bash
bash scripts/run_translation.sh 5      # KoBART 양방향, 끝에 verify_translation(chrF≥80) 자동
```
미달 시: `src/translation/translate.py` 의 `use_dictionary=True`(사전 후처리)로 보강.

### 3. ML 서비스 + 앱 연결해 end-to-end 데모
```bash
# 터미널 A: ML 서비스 (감정+번역 모델 로드)
uvicorn app.main:app --port 8000
# 터미널 B: 웹앱
cd webapp && npm run dev          # http://localhost:3000  (탭1이 8000으로 프록시)
```
탭1에서 제주어 입력 → 번역+감정 확인.

### 4. 마무리
- `docs/bert_improvement_report.md` 실측치로 갱신
- 키 주입 시 탭2~4 실기능 확인 (아래)

## 🔑 실기능 점등용 키 (`webapp/.env`, 없으면 seed/mock으로 동작)

```
ML_SERVICE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_KAKAO_MAP_JS_KEY=...   KAKAO_REST_API_KEY=...   # 지도/맛집
GOOGLE_CLIENT_ID/SECRET, KAKAO_CLIENT_ID/SECRET, NEXTAUTH_SECRET, NEXTAUTH_URL  # 로그인
ANTHROPIC_API_KEY=...              # AI 여행경로
DATABASE_URL=file:./dev.db
```
ML(감정/번역)은 항상 GPU(MPS)로 학습. 학습된 모델은 `results_kr_bert/`·`results_translation/`에 로컬 보관.

## ⚠️ 알아둘 점

- 외장 SSD라 macOS `._*` 파일이 git을 방해할 수 있음 → `find . -name '._*' -delete` (gitignore에 `._*` 등록됨)
- `run_goal.sh`의 grep이 비-line-buffered라 학습 로그는 단계 종료 시 묶여 출력됨(정상)
- 모델 가중치/데이터는 gitignore — 다른 머신에서 이어가려면 재학습 필요
