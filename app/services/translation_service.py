"""제주어↔표준어 번역 추론 서비스.

설계 문서 §3 (서브시스템 B) / §4 기준.

KoBART 양방향 단일 모델 (방향 prefix). 추론 래퍼는 `src/translation/translate.py`
의 `Translator` 클래스를 재사용한다 (`translate(text, direction)`).

graceful degradation:
  - `Translator` 와 모델 가중치는 **지연 로드** (첫 translate 호출 시).
  - 아티팩트(또는 translate.py 모듈)가 없으면 `available == False`, 그 상태로 앱이 뜬다.
    호출 시에는 `TranslationUnavailableError` 를 던져 main.py 가 503 처리.

방향(direction):
  - "j2s" : 제주어 → 표준어
  - "s2j" : 표준어 → 제주어
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

VALID_DIRECTIONS = ("j2s", "s2j")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
_TRANSLATE_MODULE_PATH = _SRC_DIR / "translation" / "translate.py"


class TranslationUnavailableError(RuntimeError):
    """번역 모델 아티팩트가 없거나 로드에 실패했을 때 발생. main.py 에서 503 처리."""


class TranslationService:
    """KoBART 양방향 번역 추론 래퍼.

    Parameters
    ----------
    artifacts_dir : KoBART 파인튜닝 모델 디렉토리 (기본 `results_translation/kobart_jeju`).
    """

    def __init__(
        self,
        artifacts_dir: str = "results_translation/kobart_jeju",
    ) -> None:
        base = Path(artifacts_dir)
        if not base.is_absolute():
            base = _PROJECT_ROOT / base
        self.artifacts_dir = base

        self._loaded = False
        self._translator = None

    # ── 가용성 ────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """추론에 필요한 것들이 갖춰졌는지.

        조건: (1) translate.py 모듈 존재, (2) 모델 아티팩트 디렉토리 존재.
        실제 가중치 로드는 지연되므로 여기서는 경로 존재만 확인한다.
        """
        return _TRANSLATE_MODULE_PATH.exists() and self.artifacts_dir.is_dir()

    def status(self) -> Dict[str, object]:
        """health 엔드포인트용 상세 상태."""
        return {
            "available": self.available,
            "loaded": self._loaded,
            "artifacts_dir": str(self.artifacts_dir),
            "artifacts_exist": self.artifacts_dir.is_dir(),
            "translate_module": {
                "path": str(_TRANSLATE_MODULE_PATH),
                "exists": _TRANSLATE_MODULE_PATH.exists(),
            },
        }

    # ── 지연 로드 ─────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.available:
            raise TranslationUnavailableError(
                "번역 모델 아티팩트를 찾을 수 없습니다. "
                f"(모델: {self.artifacts_dir}, 모듈: {_TRANSLATE_MODULE_PATH}). "
                "번역 학습 완료 후 아티팩트를 배치하세요."
            )

        # src/ 를 path 에 추가해 translation.translate.Translator 재사용.
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        try:
            from translation.translate import Translator  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - 환경 의존
            raise TranslationUnavailableError(
                f"translation.translate 모듈 로드 실패: {exc}"
            ) from exc

        try:
            self._translator = Translator(str(self.artifacts_dir))
        except Exception as exc:
            self._translator = None
            raise TranslationUnavailableError(f"번역 모델 로드 실패: {exc}") from exc

        self._loaded = True

    # ── 추론 ──────────────────────────────────────────────────────────────
    def translate(self, text: str, direction: str = "j2s") -> str:
        """문장을 지정한 방향으로 번역.

        Parameters
        ----------
        text : 원문.
        direction : "j2s"(제주어→표준어) | "s2j"(표준어→제주어).
        """
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction 은 {VALID_DIRECTIONS} 중 하나여야 합니다. (받음: {direction!r})"
            )
        if not text or not str(text).strip():
            raise ValueError("빈 문자열은 번역할 수 없습니다.")

        self._ensure_loaded()
        return self._translator.translate(str(text).strip(), direction)
