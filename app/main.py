"""제주어 챗봇 FastAPI 백엔드.

설계 문서 §4 (서브시스템 C) 기준. 번역 + 감정 분석을 REST 로 노출하고
정적 프론트를 서빙한다.

graceful degradation:
  - 앱 시작 시 두 서비스를 인스턴스화하지만 **모델은 로드하지 않는다** (지연 로드).
  - 아티팩트가 없으면 해당 서비스는 `available == False`. 앱은 정상 기동하며,
    unavailable 기능 호출 시 503 + 안내 메시지를 반환한다.

실행: `uvicorn app.main:app --reload`
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.emotion_service import EmotionService, EmotionUnavailableError
from app.services.translation_service import (
    TranslationService,
    TranslationUnavailableError,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="제주어 챗봇 API",
    description="제주어↔표준어 번역 + 제주어 감정 분석",
    version="0.1.0",
)

# 데모/로컬 프론트 편의를 위해 CORS 전체 허용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 인스턴스 — 생성만, 모델 로드는 지연.
emotion_service = EmotionService()
translation_service = TranslationService()


# ── Pydantic 모델 ─────────────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    text: str = Field(..., description="번역할 원문")
    direction: str = Field("j2s", description='"j2s"(제주어→표준어) | "s2j"(표준어→제주어)')


class TranslateResponse(BaseModel):
    translation: str


class EmotionRequest(BaseModel):
    text: str = Field(..., description="감정을 분석할 제주어 문장")


class EmotionResponse(BaseModel):
    label: str
    label_id: int
    scores: Dict[str, float]


class ChatRequest(BaseModel):
    text: str = Field(..., description="입력 문장")
    direction: str = Field("j2s", description='기본 "j2s". "s2j"면 번역만 반환')


class ChatResponse(BaseModel):
    translation: str
    emotion: Optional[EmotionResponse] = None


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, dict]


# ── 엔드포인트 ──────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        services={
            "translation": translation_service.status(),
            "emotion": emotion_service.status(),
        },
    )


@app.post("/api/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest) -> TranslateResponse:
    try:
        result = translation_service.translate(req.text, req.direction)
    except TranslationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return TranslateResponse(translation=result)


@app.post("/api/emotion", response_model=EmotionResponse)
def emotion(req: EmotionRequest) -> EmotionResponse:
    try:
        result = emotion_service.predict(req.text)
    except EmotionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return EmotionResponse(**result)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """대화 엔드포인트.

    - direction="j2s" (기본): 제주어 → 표준어 번역 + 감정 분석 동시 반환.
    - direction="s2j": 표준어 → 제주어 번역만 (감정 분석은 제주어 입력 대상이므로 생략).
    """
    direction = req.direction or "j2s"

    # 번역.
    try:
        translation = translation_service.translate(req.text, direction)
    except TranslationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 감정: 제주어 입력(j2s)일 때만. 감정 모델 unavailable 이면 번역만 돌려준다
    # (graceful degradation — 503 으로 전체를 막지 않는다).
    emotion_payload: Optional[EmotionResponse] = None
    if direction == "j2s":
        try:
            emotion_payload = EmotionResponse(**emotion_service.predict(req.text))
        except (EmotionUnavailableError, ValueError):
            emotion_payload = None

    return ChatResponse(translation=translation, emotion=emotion_payload)


# ── 정적 프론트 ──────────────────────────────────────────────────────────────
@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# /static 마운트 (app.js, style.css 등). 디렉토리는 항상 존재하도록 보장.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
