"""
Shared VLM-as-judge model loading and inference helpers for InternVL3.5-38B and
Qwen3-VL-32B-Thinking, following the same AutoModelForImageTextToText loading
pattern already validated in intern_vl_testing.ipynb / qwen_vl_testing.ipynb.

Per-model behaviour lives in JUDGE_SPECS rather than being inferred from the model id.
The previous version guessed with `"thinking" in model_id.lower()`, which is silently
wrong for InternVL3.5: its id contains no "thinking", so it would have been graded as a
non-thinking judge with no reasoning prompt - despite reasoning mode being available and
materially changing its scores (see R1_SYSTEM_PROMPT in prompts/benchmarks.py).

Both judges now run unquantized at bf16. 4-bit NF4 was found to corrupt InternVL badly
enough that it described a completely different image than the one supplied (histology
slides read back as "Barbie logos", "a tweet screenshot", "lip gloss") - a vision-tower /
projector failure, not the graceful quality loss quantization is usually assumed to cause.
Running both judges at bf16 also keeps a judge-vs-judge comparison free of a quantization
confound.
"""
import json
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from prompts.benchmarks import (
    JUDGE_SYSTEM_PROMPT,
    MINIMIZE_THINKING_SUFFIX,
    R1_SYSTEM_PROMPT,
    VALID_SCORES,
)

# Retained for reference / fallback only - NOT used. See the module docstring.
_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Everything model-specific, stated explicitly:
#   model_id            HuggingFace repo id
#   use_system_role     True  -> judge instructions go in a dedicated system turn
#                       False -> folded into the user turn (model has no system role)
#   system_suffix       appended to JUDGE_SYSTEM_PROMPT; the grading instructions
#                       themselves stay identical across judges
#   images_in_template  True  -> processor.apply_chat_template(..., tokenize=True) does
#                                tokenization AND image preprocessing in one call
#                       False -> template the text, then call the processor with images
#   trailing_reminder   appended AFTER the per-item prompt (empty string = nothing).
#                       Only used when use_system_role is False, where the instructions
#                       would otherwise sit far from the generation point.
#   load_in_4bit        kept per-model so a fallback is a one-line change
JUDGE_SPECS = {
    "internvl": {
        "model_id": "OpenGVLab/InternVL3_5-38B-HF",
        "use_system_role": True,       # InternVL has a real system role
        "system_suffix": R1_SYSTEM_PROMPT,   # reasoning mode is off without this
        "images_in_template": True,    # the model card's documented calling convention
        # InternVL needs no reminder: measured 0 unparseable calls in 1145 on Benchmark 5,
        # median ~325 output tokens. Adding one would also alter its prompt and break
        # comparability with the InternVL scores already checkpointed.
        "trailing_reminder": "",
        "load_in_4bit": False,
    },
    "qwenvl": {
        "model_id": "Qwen/Qwen3-VL-32B-Thinking",
        "use_system_role": False,      # folded into the user turn, as originally validated
        "system_suffix": MINIMIZE_THINKING_SUFFIX,  # already a thinking model; just cap the trace
        "images_in_template": False,
        "trailing_reminder": (
            "\n\nReminder: decide each criterion once, pick the lower score if torn, "
            "and output the JSON object now. Do not re-examine scores you have already "
            "reasoned about."
        ),
        "load_in_4bit": False,
    },
}

# Backwards compatibility: several notebooks still read MODEL_IDS.
MODEL_IDS = {key: spec["model_id"] for key, spec in JUDGE_SPECS.items()}


class JudgeModel:
    """Wraps a single VLM judge (InternVL or Qwen-VL) behind a uniform score() call."""

    def __init__(self, model_key: str, device_map: str = "auto", max_memory: dict | None = None):
        """`max_memory` (e.g. {0: "42GiB", 1: "42GiB", 2: "42GiB"}) is passed straight to
        Accelerate. Worth setting: at bf16 these models are sharded across several cards,
        and device_map="auto" alone tends to pack the first card tightly enough to OOM
        mid-generation - hours into a run, on the long-reasoning retry path."""
        if model_key not in JUDGE_SPECS:
            raise ValueError(f"Unknown model_key {model_key!r}; expected one of {list(JUDGE_SPECS)}")
        self.model_key = model_key
        self.spec = JUDGE_SPECS[model_key]
        self.model_id = self.spec["model_id"]

        load_kwargs = {
            "dtype": torch.bfloat16,   # `torch_dtype` is deprecated in transformers 5.x
            "device_map": device_map,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        if self.spec["load_in_4bit"]:
            load_kwargs["quantization_config"] = _BNB_CONFIG

        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.use_system_role = self.spec["use_system_role"]
        self.images_in_template = self.spec["images_in_template"]
        self.system_prompt = JUDGE_SYSTEM_PROMPT + self.spec["system_suffix"]
        self.trailing_reminder = self.spec.get("trailing_reminder", "")

        # A sharded model has no single `.device`; inputs must land on whichever device
        # holds the first layer, and Accelerate's hooks move activations onward from there.
        self.input_device = self.model.get_input_embeddings().weight.device

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
        #
        # The reasoning-budget rule is also repeated AFTER the rubric and option text.
        # Without a system role the whole preamble sits ~2000 tokens before the point
        # where generation starts, which is the weakest position for an instruction the
        # model has to obey while decoding. Restating it last costs ~30 tokens and puts
        # it where recency helps most - this matters for Qwen on Benchmark 5, where the
        # trace otherwise runs past the entire budget without emitting JSON.
        user_turn["content"][1]["text"] = (
            f"{self.system_prompt}\n\n{prompt_text}{self.trailing_reminder}"
        )
        return [user_turn]

    def _eos_token_ids(self) -> list:
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            return []
        return [eos] if isinstance(eos, int) else list(eos)

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
        if self.images_in_template:
            # One call does templating, tokenization and image preprocessing - the
            # convention documented on the InternVL3.5 model card.
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.input_device)
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            ).to(self.input_device)

        # Greedy, always. Reproducibility is a hard requirement here: the checkpoint is
        # the source of truth and a resumed run must reproduce pre-crash scores, and
        # judge-vs-judge agreement must not be contaminated by a judge disagreeing with
        # itself. See R1_SYSTEM_PROMPT in prompts/benchmarks.py for the measurements.
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, prompt_len:]
        output_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

        hit_eos = bool(set(new_tokens[0].tolist()) & set(self._eos_token_ids()))
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
    """Finds the model's final answer JSON object among possibly several
    {...}-shaped substrings in the response (a thinking model's reasoning trace
    often mentions an intermediate/draft JSON before the real final answer, e.g.
    "So the JSON would be {...}" followed later by the actual output). A single
    greedy \\{.*\\} regex would span from the FIRST { to the LAST } and swallow
    everything in between (including non-JSON text like a closing </think> tag),
    producing an unparseable blob even though a valid final JSON exists.

    Instead, this scans every non-nested {...} block (re.findall naturally
    matches disjoint, non-overlapping spans) and tries them from LAST to FIRST,
    returning the last one that both parses as JSON and contains valid scores
    for every requested criterion - the model's true final answer is expected to
    be the last such block in the text."""
    candidates = re.findall(r"\{[^{}]*\}", raw_text, re.DOTALL)
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        result = {}
        valid = True
        for criterion in criteria:
            value = obj.get(criterion)
            try:
                value = int(value)
            except (TypeError, ValueError):
                valid = False
                break
            if value not in VALID_SCORES:
                valid = False
                break
            result[criterion] = value
        if valid:
            return result
    return None
