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

from prompts.benchmarks import JUDGE_SYSTEM_PROMPT, MINIMIZE_THINKING_SUFFIX, VALID_SCORES

MODEL_IDS = {
    "internvl": "OpenGVLab/InternVL3_5-38B-HF",
    "qwenvl": "Qwen/Qwen3-VL-32B-Thinking"
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
        # "Thinking" models emit an extended <think>...</think> reasoning trace before
        # the JSON answer, which can consume the whole token budget on its own. Append
        # an explicit brevity instruction for these so the trace doesn't crowd out the
        # answer; non-thinking models get the grading instructions unchanged.
        self.is_thinking_model = "thinking" in self.model_id.lower()
        self.system_prompt = JUDGE_SYSTEM_PROMPT + (MINIMIZE_THINKING_SUFFIX if self.is_thinking_model else "")

    def _build_messages(self, image: Image.Image, prompt_text: str) -> list:
        user_turn = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
        if self.use_system_role:
            return [{"role": "system", "content": self.system_prompt}, user_turn]
        # Fold the system instructions into the text turn for models without a system role.
        user_turn["content"][1]["text"] = f"{self.system_prompt}\n\n{prompt_text}"
        return [user_turn]

    # If a call is truncated (ran out of max_new_tokens before finishing - e.g.
    # Qwen3-VL-Thinking still inside <think>...</think>), the retry gets this much
    # larger a budget instead of just trying again at the same size.
    TRUNCATION_RETRY_TOKENS = 12288

    def generate(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 8192) -> tuple[str, bool]:
        """Returns (output_text, was_truncated). was_truncated is True when
        generation used the full max_new_tokens budget without emitting an EOS
        token - i.e. the model was still mid-output (e.g. still inside a
        <think> block) when generation was cut off, as opposed to naturally
        finishing its response early."""
        messages = self._build_messages(image, prompt_text)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(
            self.model.device
        )
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        eos_token_ids = self.model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        hit_eos = bool(set(new_tokens[0].tolist()) & set(eos_token_ids))
        was_truncated = (new_tokens.shape[1] >= max_new_tokens) and not hit_eos
        return output_text, was_truncated

    def score(self, image: Image.Image, prompt_text: str, criteria: list) -> dict:
        """Runs generation and parses a {criterion: score} dict. Two distinct
        failure modes get two distinct retries:
        - Truncated generation (ran out of max_new_tokens before finishing):
          retried once with TRUNCATION_RETRY_TOKENS, since the fix is "give it
          more room," not "ask again and hope."
        - Malformed/unparseable output despite finishing normally: retried once
          at the same budget (a different sampling draw might format the JSON
          correctly - though note do_sample=False makes this deterministic
          unless the prompt itself is tweaked, so this mostly guards against
          transient decode issues).
        Any criterion still unparseable after both retries is scored -1 (matches
        the paper's own use of -1 for 'unable to make the evaluation')."""
        raw, truncated = self.generate(image, prompt_text)
        parsed = _parse_scores(raw, criteria)

        if parsed is None and truncated:
            print(
                f"[{self.model_key}] generation truncated at max_new_tokens before finishing "
                f"(criteria={criteria}) - retrying with TRUNCATION_RETRY_TOKENS={self.TRUNCATION_RETRY_TOKENS}"
            )
            raw, truncated = self.generate(image, prompt_text, max_new_tokens=self.TRUNCATION_RETRY_TOKENS)
            parsed = _parse_scores(raw, criteria)
            if parsed is None and truncated:
                print(
                    f"[{self.model_key}] still truncated even at {self.TRUNCATION_RETRY_TOKENS} tokens "
                    f"(criteria={criteria}) - falling back to -1 for this call"
                )
        elif parsed is None:
            raw, truncated = self.generate(image, prompt_text)
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
