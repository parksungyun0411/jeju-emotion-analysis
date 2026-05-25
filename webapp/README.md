# 제주올 (Jeju All-in-One) — Web App

제주 여행·생활 올인원 모바일 웹앱. **Next.js 15 (App Router, TypeScript, Tailwind)**.
4개 탭으로 구성된 모바일 우선 반응형 앱입니다.

| 탭 | 경로 | 기능 |
|----|------|------|
| 제주어 | `/` | 제주어↔표준어 번역 + 감정분석 (ML 서비스 프록시) |
| 맛집 | `/food` | Kakao 지도 + 맛집 마커/상세/길찾기 |
| 명소·체험 | `/spots` | 명소 큐레이션 + AI 여행경로(Claude) |
| 커뮤니티 | `/community` `/mypage` | 게시판/댓글/좋아요 + 로그인/마이페이지 |

키·모델이 없어도 앱은 기동·탐색되며, 각 기능은 seed/mock/안내로 graceful degradation 됩니다.

## 빠른 시작

```bash
cd webapp

# 1) 의존성 설치 (postinstall 에서 prisma generate 자동 실행)
npm install

# 2) 환경변수 파일 생성 후 보유 키 입력 (선택)
cp .env.example .env.local
#  - 키가 비어 있어도 앱은 동작합니다.

# 3) DB 마이그레이션 (커뮤니티/유저 — SQLite)
npx prisma migrate dev --name init

# 4) 개발 서버
npm run dev
#  → http://localhost:3000
```

### ML 서비스(탭1 번역/감정) 별도 기동

탭1의 번역/감정 기능은 리포 루트의 **FastAPI ML 서비스**(`../app`)가 별도 프로세스로
떠 있어야 동작합니다. (없으면 "모델 학습 중" 안내 표시)

```bash
# 리포 루트에서 (webapp 의 상위)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

웹앱은 `ML_SERVICE_URL`(기본 `http://127.0.0.1:8000`)로 `/api/translate`, `/api/emotion` 을 프록시합니다.

## 빌드

```bash
npm run build   # prisma generate + next build (타입체크 포함)
npm start       # 프로덕션 서버
```

## 환경변수 (`.env.example` 참고)

| 키 | 용도 | 없을 때 |
|----|------|---------|
| `ML_SERVICE_URL` | 탭1 번역/감정 FastAPI 베이스 URL | 기본 `http://127.0.0.1:8000`, 미응답 시 "모델 학습 중" |
| `NEXT_PUBLIC_KAKAO_MAP_JS_KEY` | Kakao 지도 JS SDK (브라우저) | 지도 placeholder, 목록은 동작 |
| `KAKAO_REST_API_KEY` | Kakao Local 장소 검색 (서버) | `data/restaurants.seed.json` 폴백 |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | NextAuth Google 로그인 | 해당 provider 비활성 |
| `KAKAO_CLIENT_ID` / `KAKAO_CLIENT_SECRET` | NextAuth Kakao 로그인 | 해당 provider 비활성 |
| `NEXTAUTH_SECRET` / `NEXTAUTH_URL` | NextAuth 세션 | 로그인 불가(안내 표시) |
| `ANTHROPIC_API_KEY` | AI 여행경로(Claude) | mock 일정 반환 |
| `DATABASE_URL` | Prisma (dev: SQLite) | 기본 `file:./dev.db` |

## 디렉토리

```
webapp/
  app/
    page.tsx                # 탭1 제주어
    food/page.tsx           # 탭2 맛집
    spots/page.tsx          # 탭3 명소·체험 + AI 경로
    community/              # 탭4 커뮤니티 (목록/상세/작성)
    mypage/page.tsx         # 마이페이지
    api/
      translate, emotion    # ML 프록시
      places, attractions   # 지도 데이터 (+seed 폴백)
      route                 # AI 여행경로 (Claude)
      posts/...             # 커뮤니티 CRUD
      me/posts              # 내 글
      auth/[...nextauth]    # NextAuth
  components/               # BottomNav, KakaoMap, BottomSheet, AuthButtons ...
  lib/                      # prisma, auth, types
  data/                     # restaurants/attractions seed JSON
  prisma/schema.prisma      # User/Account/Session/Post/Comment/Like/Bookmark
```

## 데이터 모델 (Prisma)

`User`, `Account`, `Session`, `VerificationToken` (NextAuth 표준) +
`Post`, `Comment`, `Like`, `Bookmark` (커뮤니티). 맛집/명소는 외부 API + seed JSON.

## 참고

- 설계 문서: `../docs/superpowers/specs/2026-05-25-jeju-allinone-app-design.md`
- 비밀키는 절대 커밋하지 마세요. `.env*` 는 `.gitignore` 처리되어 있습니다.
