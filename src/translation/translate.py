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
import json
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# 방향 prefix (학습 시와 동일해야 함): 출력 언어를 가리킨다.
PREFIX = {"j2s": "표준어로: ", "s2j": "제주어로: "}

DEFAULT_MODEL_DIR = "results_translation/kobart_jeju"
DEFAULT_DICT_PATH = str(Path(__file__).resolve().parent / "data" / "jeju_std_dict.json")

# 어절 양끝 구두점 처리(build_dictionary.py 와 동일 규칙).
_PUNCT = ".,?!\"'·…~“”’‘()[]{}"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ─── 사전(lexicon) 폴백/후처리 ────────────────────────────────────────────────

_DICT_CACHE = {}


def load_dictionary(dict_path=DEFAULT_DICT_PATH):
    """build_dictionary.py 가 만든 사전 JSON 로드(캐시). 없으면 None."""
    key = str(dict_path)
    if key in _DICT_CACHE:
        return _DICT_CACHE[key]
    p = Path(dict_path)
    if not p.exists():
        logger.warning(f"사전 파일 없음: {p} — 사전 후처리 비활성(모델 출력 그대로).")
        _DICT_CACHE[key] = None
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    _DICT_CACHE[key] = data
    logger.info(f"사전 로드: {p} (j2s={len(data.get('j2s', {}))}, s2j={len(data.get('s2j', {}))})")
    return data


def _apply_dict_to_text(text, mapping):
    """어절 단위 사전 치환. 어절 양끝 구두점은 보존하고 코어 표면형만 치환."""
    if not mapping:
        return text
    out = []
    for tok in str(text).split():
        # 양끝 구두점 분리
        lead = ""
        trail = ""
        core = tok
        while core and core[0] in _PUNCT:
            lead += core[0]
            core = core[1:]
        while core and core[-1] in _PUNCT:
            trail = core[-1] + trail
            core = core[:-1]
        repl = mapping.get(core, core)
        out.append(lead + repl + trail)
    return " ".join(out)


def dictionary_translate(text, direction, dict_path=DEFAULT_DICT_PATH):
    """순수 사전 기반 번역(모델 없을 때 폴백). 사전 없으면 입력 그대로 반환.

    direction='j2s'(제주어→표준어) | 's2j'(표준어→제주어)."""
    if direction not in PREFIX:
        raise ValueError(f"direction 은 'j2s' 또는 's2j' 여야 합니다 (받음: {direction!r})")
    data = load_dictionary(dict_path)
    if not data:
        return str(text)
    mapping = data.get(direction, {})
    return _apply_dict_to_text(text, mapping)


class Translator:
    """save_pretrained 된 KoBART 번역 모델 로더 + 양방향 추론."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, device=None, num_beams=4, max_length=64,
                 use_dictionary=False, dict_path=DEFAULT_DICT_PATH):
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
        self.use_dictionary = use_dictionary       # 기본 후처리 사용 여부(메서드에서 override 가능)
        self.dict_path = dict_path
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()
        # 사전은 lazy 로드(미존재 시 후처리 자동 skip)
        self._dict = load_dictionary(dict_path) if use_dictionary else None
        logger.info(f"Translator 로드 완료: {model_path} (device={self.device}, "
                    f"use_dictionary={use_dictionary})")

    def _post_dict(self, texts, direction, use_dictionary):
        """모델 출력에 사전 치환 후처리 적용. 사전 없으면 그대로 반환."""
        enabled = self.use_dictionary if use_dictionary is None else use_dictionary
        if not enabled:
            return texts
        data = self._dict if self._dict is not None else load_dictionary(self.dict_path)
        if not data:
            return texts
        mapping = data.get(direction, {})
        return [_apply_dict_to_text(t, mapping) for t in texts]

    @staticmethod
    def _check_direction(direction):
        if direction not in PREFIX:
            raise ValueError(f"direction 은 'j2s' 또는 's2j' 여야 합니다 (받음: {direction!r})")

    @torch.no_grad()
    def translate(self, text, direction, use_dictionary=None):
        """단일 문장 번역. direction='j2s'(제주어→표준어) | 's2j'(표준어→제주어).

        use_dictionary: None(=인스턴스 기본값) | True | False. True 면 모델 출력에
        사전 치환 후처리를 적용한다(사전 미존재 시 자동 skip)."""
        return self.translate_batch([text], direction, use_dictionary=use_dictionary)[0]

    @torch.no_grad()
    def translate_batch(self, texts, direction, use_dictionary=None):
        """문장 리스트 일괄 번역. 방향 prefix 를 붙여 beam search 생성 후 디코드.
        use_dictionary 가 켜지면 사전 치환 후처리를 거친 결과를 반환."""
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
        outputs = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return self._post_dict(outputs, direction, use_dictionary)


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
