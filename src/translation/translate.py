"""
제주어 ↔ 표준어 번역 추론 래퍼.

train_translation.py 가 save_pretrained 한 KoBART 모델을 로드해 방향 prefix 를 붙여 생성.
챗봇 백엔드(app/services/translation_service.py)에서 의존성 주입으로 사용.

사용:
    from src.translation.translate import Translator
    tr = Translator("results_translation/kobart_jeju")
    tr.translate("밥 먹었수꽈?", "j2s")           # 제주어 → 표준어
    tr.translate_batch(["...", "..."], "s2j")     # 표준어 → 제주어
"""
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# 방향 prefix (학습 시와 동일해야 함): 출력 언어를 가리킨다.
PREFIX = {"j2s": "표준어로: ", "s2j": "제주어로: "}

DEFAULT_MODEL_DIR = "results_translation/kobart_jeju"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Translator:
    """save_pretrained 된 KoBART 번역 모델 로더 + 양방향 추론."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device=None, num_beams=4, max_length=64):
        model_path = Path(model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"번역 모델을 찾을 수 없습니다: {model_path}\n"
                f"먼저 학습을 실행하세요: "
                f"python3 src/translation/train_translation.py --fp16\n"
                f"(또는 bash scripts/run_translation.sh)"
            )
        self.device = device if device is not None else get_device()
        self.num_beams = num_beams
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()
        logger.info(f"Translator 로드 완료: {model_path} (device={self.device})")

    @staticmethod
    def _check_direction(direction):
        if direction not in PREFIX:
            raise ValueError(f"direction 은 'j2s' 또는 's2j' 여야 합니다 (받음: {direction!r})")

    @torch.no_grad()
    def translate(self, text, direction):
        """단일 문장 번역. direction='j2s'(제주어→표준어) | 's2j'(표준어→제주어)."""
        return self.translate_batch([text], direction)[0]

    @torch.no_grad()
    def translate_batch(self, texts, direction):
        """문장 리스트 일괄 번역. 방향 prefix 를 붙여 beam search 생성 후 디코드."""
        self._check_direction(direction)
        prefix = PREFIX[direction]
        sources = [prefix + str(t) for t in texts]
        enc = self.tokenizer(
            sources, truncation=True, max_length=self.max_length,
            padding=True, return_tensors="pt",
        ).to(self.device)
        gen = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=self.num_beams,
            max_length=self.max_length,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="번역 추론 빠른 테스트")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--direction", choices=["j2s", "s2j"], default="j2s")
    parser.add_argument("text", nargs="+", help="번역할 문장")
    args = parser.parse_args()

    tr = Translator(args.model_dir)
    for src, out in zip(args.text, tr.translate_batch(args.text, args.direction)):
        print(f"[{args.direction}] {src}  →  {out}")
