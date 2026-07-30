"""
Shared VLM-as-judge model loading and inference helpers for InternVL3.5-38B and
Qwen3-VL-32B-Thinking, following the same AutoModelForImageTextToText loading
pattern already validated in intern_vl_testing.ipynb / qwen_vl_testing.ipynb.
"""
import json
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from prompts.benchmarks import JUDGE_SYSTEM_PROMPT, VALID_SCORES

MODEL_IDS = {
    "internvl": "OpenGVLab/InternVL3_5-38B-HF",
    "qwenvl": "Qwen/Qwen3-VL-32B-Thinking",
}

_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


class JudgeModel:
    """Wraps a single VLM judge (InternVL or Qwen-VL) behind a uniform score() call."""

    def __init__(self, model_key: str, device_map: str = "auto"):
        if model_key not in MODEL_IDS:
            raise ValueError(f"Unknown model_key {model_key!r}; expected one of {list(MODEL_IDS)}")
        self.model_key = model_key
        self.model_id = MODEL_IDS[model_key]
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            quantization_config=_BNB_CONFIG,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # Qwen3-VL-Thinking has no dedicated system-prompt convention in the existing
        # qwen_vl_testing.ipynb; InternVL's testing notebook uses a system role. We fold
        # the judge instructions into the user turn for both models for consistency.
        self.use_system_role = model_key == "internvl"

    def _build_messages(self, image: Image.Image, prompt_text: str) -> list:
        user_turn = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
        if self.use_system_role:
            return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, user_turn]
        # Fold the system instructions into the text turn for models without a system role.
        user_turn["content"][1]["text"] = f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt_text}"
        return [user_turn]

    def generate(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 512) -> str:
        messages = self._build_messages(image, prompt_text)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(
            self.model.device
        )
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        output_text = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return output_text[0]

    def score(self, image: Image.Image, prompt_text: str, criteria: list) -> dict:
        """Runs generation and parses a {criterion: score} dict, retrying once on
        malformed output. Any criterion that can't be parsed/validated is scored -1
        (matches the paper's own use of -1 for 'unable to make the evaluation')."""
        raw = self.generate(image, prompt_text)
        parsed = _parse_scores(raw, criteria)
        if parsed is None:
            raw = self.generate(image, prompt_text)
            parsed = _parse_scores(raw, criteria)
        if parsed is None:
            parsed = {c: -1 for c in criteria}
        return parsed, raw


def _parse_scores(raw_text: str, criteria: list) -> dict | None:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    result = {}
    for criterion in criteria:
        value = obj.get(criterion)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        if value not in VALID_SCORES:
            return None
        result[criterion] = value
    return result
