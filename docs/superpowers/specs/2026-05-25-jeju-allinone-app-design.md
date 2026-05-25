# 제주 올인원 앱 — 마스터 설계 문서 (v2)

> 작성 2026-05-25. **범위 확대:** "제주어 챗봇"에서 **제주도에 관한 모든 것**을 담는
> 4탭 모바일 웹앱으로 확장한다. 기존 v1(챗봇) 설계는 본 앱의 탭 1로 흡수된다.
>
> ⚠️ **승인 상태:** 사용자가 자리를 비운 상태에서 "네가 효율적으로 결정해 진행" 지시로
> 작성한 자율 설계. 결정은 합리적 기본값이며 복귀 시 검토·수정 가능하도록 모듈식 구현.
> 참고: [[autonomy-grant]], 이전 설계 `2026-05-25-jeju-chatbot-design.md`.

## 1. 제품 개요 (기획서 요약)

**한 줄:** 제주 여행·생활에 필요한 모든 것 — 제주어 번역/감정분석, 맛집, 명소·체험,
AI 여행경로, 커뮤니티 — 을 한 앱에 담은 **제주 올인원 모바일 웹앱**.

**가제:** `탐라(Tamna)` 또는 `제주올`(JejuAll). 기획서에서 확정.

**4개 탭 (하단 네비게이션):**
1. **제주어** — 제주어↔표준어 번역 + 제주어 입력 감정분석(7종). (기존 ML 자산 활용)
2. **맛집** — 카카오 지도에 제주 맛집 표시, 마커 클릭 시 상세정보, 길찾기.
3. **명소·체험** — 가볼 명소/할 수 있는 체험을 지도에 큐레이션, 상세·길찾기, **AI 여행경로 생성**.
4. **커뮤니티** — 제주 관련 Q&A·꿀팁 게시/댓글. **로그인(Google/Kakao) 필요**, **마이페이지**.

**비목표(MVP 제외, 향후):** 결제/예약, 실시간 채팅, 푸시 알림, 네이티브 앱 스토어 배포,
오프라인 모드. (반응형 웹으로 "어플" 경험 제공, 추후 PWA/네이티브 래핑.)

**대상 사용자:** 제주 여행객, 제주 이주/거주자, 제주 문화·언어 관심층.

## 2. 기술 스택 (결정)

| 영역 | 선택 | 이유 |
|------|------|------|
| 프론트/앱 | **Next.js 15 (App Router, TypeScript, Tailwind)** | 모바일 반응형, SSR/API라우트 일체화, 사용자 보유 경험(NerdMath) |
| 지도 | **Kakao Maps JS SDK + Kakao Local REST API** | 국내 장소/맛집/길찾기에 최적 |
| 길찾기 | Kakao Map 길찾기 링크 / (확장) Kakao Mobility API | 키만으로 즉시 |
| 인증 | **NextAuth.js (Google + Kakao provider)** | 마이페이지·세션, OAuth 표준 |
| DB | **Prisma + SQLite(dev) / PostgreSQL(prod)** | 커뮤니티·유저, 마이그레이션 용이 |
| AI 여행경로 | **Anthropic Claude API** (서버 라우트) | 선택 명소 기반 일정 생성 |
| ML(번역/감정) | **기존 FastAPI 서비스** (KoBART + 감정 앙상블) | 이미 구현, Python 자산 재사용 |
| 상태/데이터 | React Server Components + fetch, 필요시 SWR | 단순성 |

**서비스 분리:** Next.js(웹앱·앱로직·커뮤니티·인증·AI라우트)와 Python FastAPI(ML 추론)는
별도 프로세스. Next.js가 `ML_SERVICE_URL`로 ML 서비스를 호출(서버 라우트 프록시).

## 3. 시스템 아키텍처

```
[모바일 웹 브라우저]
   └─ Next.js 15 (webapp/)  ── App Router 페이지 + 하단 탭 네비
        ├─ /(탭1) 제주어   → /api/translate, /api/emotion  ─proxy→ [FastAPI ML]
        ├─ /food (탭2)     → Kakao Maps SDK + /api/places   ─→ [Kakao Local API]
        ├─ /spots (탭3)    → Kakao Maps + /api/route(AI)     ─→ [Claude API]
        ├─ /community(탭4) → /api/posts ...                  ─→ [Prisma/DB]
        ├─ /mypage         → 세션 기반
        └─ /api/auth/*     → NextAuth (Google, Kakao)
[FastAPI ML 서비스]  번역(KoBART) + 감정(Dual KR-BERT + KoELECTRA 앙상블)
```

## 4. 탭별 상세 설계

### 탭 1 — 제주어 (번역 + 감정)
- UI: 입력창 + 방향 토글(제주어→표준어/역) + 결과 카드(번역문 + 감정 뱃지·신뢰도).
- 데이터: Next 서버 라우트 `/api/translate`,`/api/emotion`이 FastAPI(`ML_SERVICE_URL`) 프록시.
- ML 미준비 시: 명확한 "모델 학습 중" 안내(503 graceful).

### 탭 2 — 맛집 (지도)
- Kakao Maps 지도에 맛집 마커. 데이터 출처: (a) Kakao Local `keyword/category` 검색(FD6 음식점),
  (b) 큐레이션 seed JSON(`webapp/data/restaurants.seed.json`) 폴백.
- 마커 클릭 → 바텀시트 상세(이름/카테고리/주소/평점/사진 placeholder) + **길찾기**(Kakao Map 링크).
- 필터: 지역/카테고리. 검색.

### 탭 3 — 명소·체험 (지도 + AI 경로)
- 명소/체험 큐레이션(seed JSON `attractions.seed.json`, 카테고리: 자연/문화/체험/포토).
- 지도 마커 + 상세 + 길찾기. 다중 선택(장바구니).
- **AI 여행경로 생성:** 선택 장소 + 사용자 조건(일수/이동수단/취향)을 서버 라우트 `/api/route`가
  Claude API로 보내 일자별 동선 일정을 생성. 결과를 지도에 폴리라인으로 표시.

### 탭 4 — 커뮤니티 + 마이페이지
- 게시판: 카테고리(질문/꿀팁/후기). 목록·상세·작성·댓글·좋아요. **작성/댓글은 로그인 필요.**
- 인증: NextAuth Google + Kakao. 세션은 DB(Prisma adapter).
- **마이페이지:** 프로필(이름/이메일/아바타), 내 글/댓글, 좋아요/북마크, 로그아웃.

## 5. 데이터 모델 (Prisma)

- `User`(NextAuth 표준: id,name,email,image) + `Account`,`Session`,`VerificationToken`
- `Post`(id, authorId, category, title, body, createdAt, likeCount)
- `Comment`(id, postId, authorId, body, createdAt)
- `Like`(userId, postId) / `Bookmark`(userId, targetType, targetId)
- 맛집/명소는 외부 API + seed JSON(초기엔 DB 미적재).

## 6. API 라우트 (Next.js)

| 메서드·경로 | 설명 |
|---|---|
| `POST /api/translate` | {text,direction} → ML 프록시 |
| `POST /api/emotion` | {text} → ML 프록시 |
| `GET /api/places` | {query,category,lat,lng} → Kakao Local 프록시(+seed 폴백) |
| `POST /api/route` | {spots[],days,transport,prefs} → Claude로 일정 생성 |
| `GET/POST /api/posts`, `/api/posts/[id]`, `/comments`, `/like` | 커뮤니티 CRUD(로그인 가드) |
| `/api/auth/[...nextauth]` | NextAuth (Google, Kakao) |

## 7. 비밀키/환경변수 (`.env.example` 제공, 실제 키는 사용자 보유분 주입)

```
ML_SERVICE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_KAKAO_MAP_JS_KEY=...     # 카카오 지도 JS
KAKAO_REST_API_KEY=...               # 카카오 로컬 검색
GOOGLE_CLIENT_ID=...  GOOGLE_CLIENT_SECRET=...
KAKAO_CLIENT_ID=...   KAKAO_CLIENT_SECRET=...
NEXTAUTH_SECRET=...   NEXTAUTH_URL=http://localhost:3000
ANTHROPIC_API_KEY=...                # AI 여행경로
DATABASE_URL=file:./dev.db           # dev SQLite
```
키가 없으면 해당 기능은 **mock/seed 데이터 또는 안내 메시지**로 graceful degradation.
앱 자체는 키 없이도 기동·탐색 가능해야 한다.

## 8. 저장소 구조 (모노레포)

```
jeju-emotion-analysis/            # (리포 이름은 유지, README에 범위 명시; 추후 jeju-allinone 등으로 변경 검토)
  webapp/            # ★ Next.js 15 앱 (신규) — 4탭 프론트 + API 라우트 + 인증 + 커뮤니티
    app/ (라우트), components/, lib/, data/(seed), prisma/, .env.example
  app/               # FastAPI ML 서비스 (번역+감정) — 기존
  src/               # 학습 코드(감정 jeju_kr_bert.py, 번역 translation/) — 기존
  docs/              # 기획서.md + 기획서.html + 개발설계서.md + specs/
```

## 9. 작업 순서 / 병렬화

- **GPU(직렬):** ① 감정 학습(진행 중) → ② 번역 학습. (앱 개발과 무관하게 계속)
- **비-GPU(지금 병렬, 에이전트):**
  - A. 기획서 재작성 (`docs/기획서.md` + `docs/기획서.html`)
  - B. 개발설계서 재작성 (`docs/개발설계서.md`)
  - C. Next.js 앱 스캐폴드 (`webapp/`) — 단일 에이전트가 프로젝트 전체 소유(충돌 방지)
- **이후:** 탭별 기능 점진 구현(맵·커뮤니티·AI경로), ML 연결, 데모 검증, 푸시/PR.

## 10. MVP 정의 (이번 사이클 목표)

기동 가능한 4탭 앱 셸 + 탭1(ML 연결, 모델 준비 시 동작) + 탭2/3 지도·마커·상세·길찾기(seed 데이터로
동작, Kakao 키 주입 시 실데이터) + 커뮤니티 CRUD(SQLite) + 로그인/마이페이지(키 주입 시) + AI경로
라우트(키 주입 시). 키·모델 없이도 앱이 뜨고 탐색되는 것을 수용 기준으로 한다.

## 11. 리스크

- 외부 키 의존(Kakao/OAuth/Claude): 키 없으면 해당 기능 제한 → seed/mock·안내로 완화, `.env.example`로 명세.
- 범위 과대: 탭별 독립 구현·점진 출시. 각 탭은 다른 탭 없이도 동작.
- ML 서비스 연동: Next↔FastAPI 별도 기동. README에 두 프로세스 실행법 명시.
