"""감정 분석 추론 서비스.

설계 문서 §4 (서브시스템 C) / §5 (모델 아티팩트) 기준.

앙상블 구조 (jeju_kr_bert.py 발표본 명세 그대로):
  p_ens = α · p_dual + (1 - α) · p_single
  - p_dual   : Dual-Gated KR-BERT (state_dict `step3_dual_state.pt`)
  - p_single : KoELECTRA (`branch1_koelectra_model/` 디렉토리, save_pretrained)
  - α        : `step7_ensemble_report.txt` 에서 파싱 (없으면 0.5)

graceful degradation:
  - 무거운 라이브러리(torch/transformers)와 모델 가중치는 **지연 로드**.
    모듈 import 시점이나 서비스 생성 시점에는 절대 로드하지 않는다.
  - 아티팩트가 없으면 `available == False` 로 두고, 그 상태로도 앱이 뜬다.
    `predict()` 호출 시에는 `EmotionUnavailableError` 를 던져 main.py 가 503 처리.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional


# 감정 라벨 — jeju_kr_bert.EMOTION_LABELS 와 1:1 일치해야 함 (앙상블 컬럼 순서 동일).
#   0 중립 / 1 슬픔 / 2 행복 / 3 분노 / 4 놀람 / 5 공포 / 6 혐오
EMOTION_LABELS: Dict[int, str] = {
    0: "중립", 1: "슬픔", 2: "행복", 3: "분노", 4: "놀람", 5: "공포", 6: "혐오",
}
NUM_LABELS = len(EMOTION_LABELS)

# Dual / KoELECTRA 모두 KR-BERT/KoELECTRA char 인코더 기반. 추론 입력 길이 상한.
MAX_LENGTH = 64

# 프로젝트 루트 (app/services/emotion_service.py → 2단계 위).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"


class EmotionUnavailableError(RuntimeError):
    """감정 모델 아티팩트가 없거나 로드에 실패했을 때 발생. main.py 에서 503 처리."""


class EmotionService:
    """감정 분석 앙상블 추론 래퍼.

    Parameters
    ----------
    artifacts_dir : 모델 아티팩트 디렉토리 (기본 `results_kr_bert`).
    dual_state_name : Dual KR-BERT state_dict 파일명.
    koelectra_dir_name : KoELECTRA save_pretrained 디렉토리명.
    ensemble_report_name : 앙상블 α 파싱 대상 리포트 파일명.
    encoder_name : Dual KR-BERT 인코더 (state_dict 로드용 백본).
    """

    def __init__(
        self,
        artifacts_dir: str = "results_kr_bert",
        dual_state_name: str = "step3_dual_state.pt",
        koelectra_dir_name: str = "branch1_koelectra_model",
        ensemble_report_name: str = "step7_ensemble_report.txt",
        encoder_name: str = "snunlp/KR-BERT-char16424",
    ) -> None:
        base = Path(artifacts_dir)
        if not base.is_absolute():
            base = _PROJECT_ROOT / base
        self.artifacts_dir = base
        self.dual_state_path = base / dual_state_name
        self.koelectra_dir = base / koelectra_dir_name
        self.ensemble_report_path = base / ensemble_report_name
        self.encoder_name = encoder_name

        # 지연 로드 대상 (첫 predict 호출 시 채워짐).
        self._loaded = False
        self._device = None
        self._dual_model = None
        self._dual_tokenizer = None
        self._koelectra_model = None
        self._koelectra_tokenizer = None
        self._alpha = 0.5

    # ── 가용성 ────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """추론에 필요한 아티팩트가 모두 존재하는지.

        앙상블은 dual + KoELECTRA 둘 다 필요하다.
        (한쪽만 있는 부분 동작은 본 MVP 범위 밖 — 명확히 unavailable 로 둔다.)
        """
        return self.dual_state_path.exists() and self.koelectra_dir.is_dir()

    def status(self) -> Dict[str, object]:
        """health 엔드포인트용 상세 상태."""
        return {
            "available": self.available,
            "loaded": self._loaded,
            "artifacts_dir": str(self.artifacts_dir),
            "dual_state": {
                "path": str(self.dual_state_path),
                "exists": self.dual_state_path.exists(),
            },
            "koelectra": {
                "path": str(self.koelectra_dir),
                "exists": self.koelectra_dir.is_dir(),
            },
            "alpha": self._alpha,
        }

    # ── α 파싱 ────────────────────────────────────────────────────────────
    def _parse_alpha(self) -> float:
        """step7_ensemble_report.txt 에서 α 값을 읽는다. 없으면 0.5.

        리포트에는 `Hyperparams: {"alpha": 0.55}` 와 `alpha: 0.55` (Extra) 형태가
        모두 존재할 수 있어 둘 다 시도한다 (jeju_kr_bert.run_step_ensemble 참고).
        """
        if not self.ensemble_report_path.exists():
            return 0.5
        try:
            text = self.ensemble_report_path.read_text(encoding="utf-8")
        except OSError:
            return 0.5
        # "alpha": 0.55  또는  alpha: 0.55  형태 모두 매칭.
        m = re.search(r'alpha"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
        if m:
            try:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                pass
        return 0.5

    # ── 지연 로드 ─────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.available:
            raise EmotionUnavailableError(
                "감정 모델 아티팩트를 찾을 수 없습니다. "
                f"(dual: {self.dual_state_path}, koelectra: {self.koelectra_dir}). "
                "감정 학습 완료 후 아티팩트를 배치하세요."
            )

        # 무거운 import 는 여기서만 (모듈/생성 시점 X).
        import torch  # noqa: WPS433
        from transformers import (  # noqa: WPS433
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        # src/ 를 path 에 추가해 DualGatedClassifier / get_device 재사용.
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        try:
            from jeju_kr_bert import DualGatedClassifier, get_device  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - 환경 의존
            raise EmotionUnavailableError(
                f"jeju_kr_bert 모듈 로드 실패: {exc}"
            ) from exc

        try:
            self._device = get_device()

            # Dual KR-BERT: 백본 인코더 + state_dict 복원.
            dual = DualGatedClassifier(self.encoder_name, num_labels=NUM_LABELS)
            state = torch.load(self.dual_state_path, map_location="cpu")
            dual.load_state_dict(state)
            dual.to(self._device)
            dual.eval()
            self._dual_model = dual
            self._dual_tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)

            # KoELECTRA: save_pretrained 디렉토리에서 직접 로드.
            koe = AutoModelForSequenceClassification.from_pretrained(self.koelectra_dir)
            koe.to(self._device)
            koe.eval()
            self._koelectra_model = koe
            self._koelectra_tokenizer = AutoTokenizer.from_pretrained(self.koelectra_dir)

            self._alpha = self._parse_alpha()
        except Exception as exc:
            # 로드 실패 시 상태를 깨끗이 두고 명확한 예외.
            self._dual_model = None
            self._koelectra_model = None
            raise EmotionUnavailableError(f"감정 모델 로드 실패: {exc}") from exc

        self._loaded = True

    # ── 추론 ──────────────────────────────────────────────────────────────
    def predict(self, text: str) -> Dict[str, object]:
        """제주어 문장의 감정을 앙상블 추론.

        Returns
        -------
        {"label": str, "label_id": int, "scores": {감정명: float, ...}}
        """
        if not text or not str(text).strip():
            raise ValueError("빈 문자열은 분석할 수 없습니다.")

        self._ensure_loaded()

        import torch  # noqa: WPS433
        import torch.nn.functional as F  # noqa: WPS433

        text = str(text).strip()

        with torch.no_grad():
            # Dual KR-BERT: 제주어/표준어 두 입력을 받지만 추론 시 표준어 번역이
            # 없으므로 동일 텍스트를 양쪽에 넣는다 (게이트가 자체 조정).
            enc = self._dual_tokenizer(
                text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
            )
            ids = enc["input_ids"].to(self._device)
            mask = enc["attention_mask"].to(self._device)
            _, dual_logits, _ = self._dual_model(
                jeju_ids=ids, jeju_mask=mask, std_ids=ids, std_mask=mask, labels=None,
            )
            p_dual = F.softmax(dual_logits.float(), dim=-1)

            # KoELECTRA single.
            enc_k = self._koelectra_tokenizer(
                text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
            )
            koe_out = self._koelectra_model(
                input_ids=enc_k["input_ids"].to(self._device),
                attention_mask=enc_k["attention_mask"].to(self._device),
            )
            p_single = F.softmax(koe_out.logits.float(), dim=-1)

            # 가중 앙상블.
            p_ens = self._alpha * p_dual + (1.0 - self._alpha) * p_single
            probs = p_ens.squeeze(0).cpu().tolist()

        label_id = int(max(range(NUM_LABELS), key=lambda i: probs[i]))
        scores = {EMOTION_LABELS[i]: float(probs[i]) for i in range(NUM_LABELS)}
        return {
            "label": EMOTION_LABELS[label_id],
            "label_id": label_id,
            "scores": scores,
        }
