"""Loading and prompting the candidate answerer models.

Six models across four tiers of domain specialization. Unlike the two judges - both large
and both loadable through `AutoModelForImageTextToText` - these span three different
architectures, so each gets an explicit spec rather than relying on auto-detection:

    Qwen2.5-VL family   Qwen2.5-VL-7B, Patho-R1-7B  (Patho-R1 IS a Qwen2.5-VL finetune)
    InternVL            InternVL3.5-8B
    Gemma3              MedGemma-4b
    LLaVA               LLaVA-Med-v1.5 (Mistral), Quilt-LLaVA (Llama)

The LLaVA pair is the awkward one: `LlavaMistralForCausalLM` / `LlavaLlamaForCausalLM`
are the original LLaVA repo's classes, not the `LlavaForConditionalGeneration` that ships
with transformers. They are handled separately, and each is smoke-tested before being
wired into a full run - the same discipline that caught InternVL's 4-bit corruption in
the judge pipeline, where a model appeared to load fine and produced fluent nonsense.

Every model here is 4-8B and fits on ONE A6000, which is deliberate: at this size several
answerers run concurrently on separate GPUs, whereas at 32B you could afford one - and
cross-model comparison is the entire point of sub-pillar 4a.
"""
from __future__ import annotations

import os

import torch
from PIL import Image

# One place to state everything model-specific.
#   loader          which adapter class below knows how to load and prompt it
#   tier            domain-specialization level, for the Pillar 4a ranking
#   trust_remote    some repos ship custom modelling code
ANSWERER_SPECS = {
    "qwen25vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "loader": "qwen25vl",
        "tier": "general",
        "params": "7B",
        "trust_remote": False,
    },
    "internvl8b": {
        "model_id": "OpenGVLab/InternVL3_5-8B-HF",
        "loader": "internvl",
        "tier": "general",
        "params": "8B",
        "trust_remote": False,
    },
    "medgemma": {
        "model_id": "google/medgemma-4b-it",
        "loader": "gemma3",
        "tier": "broad_medical",
        "params": "4B",
        "trust_remote": False,
    },
    "llavamed": {
        "model_id": "microsoft/llava-med-v1.5-mistral-7b",
        "loader": "llava_legacy",
        "tier": "biomedical",
        "params": "7B",
        "trust_remote": True,
    },
    "quiltllava": {
        "model_id": "wisdomik/Quilt-Llava-v1.5-7b",
        "loader": "llava_legacy",
        "tier": "pathology",
        "params": "7B",
        "trust_remote": True,
    },
    "pathor1": {
        # A Qwen2.5-VL-7B finetune, so it reuses that adapter exactly. Reported 69.53% on
        # PathMMU test_tiny - the same split scored here, so it doubles as a harness check:
        # a wildly different number means our prompting or parsing is wrong, not the model.
        "model_id": "WenchuanZhang/Patho-R1-7B",
        "loader": "qwen25vl",
        "tier": "pathology_sota",
        "params": "7B",
        "trust_remote": False,
    },
}

# Asking for a bare letter keeps parsing unambiguous and generation short. `metrics.py`
# still handles chattier replies, but the prompt should not invite them: a model that
# reasons for 500 tokens costs 50x more and is no more correct on a multiple-choice task.
MCQ_INSTRUCTION = (
    "Answer the multiple-choice question about the pathology image. "
    "Respond with ONLY the letter of the correct option and nothing else."
)

# Blind condition (sub-pillar 3b): the image is withheld to test whether the model can
# shortcut from text priors alone. The wording must NOT hint that an image is missing,
# or a well-behaved model refuses instead of guessing - and a refusal is not the same
# measurement as a text-prior guess.
MCQ_INSTRUCTION_BLIND = (
    "Answer the following multiple-choice question. "
    "Respond with ONLY the letter of the correct option and nothing else."
)


def build_mcq_prompt(question: str, lettered_options: list, blind: bool = False) -> str:
    instruction = MCQ_INSTRUCTION_BLIND if blind else MCQ_INSTRUCTION
    options_block = "\n".join(lettered_options)
    return f"{instruction}\n\nQuestion: {question}\n\n{options_block}\n\nAnswer:"


class AnswererModel:
    """Base adapter: load a model, answer one (image, prompt) pair.

    Subclasses implement `_load` and `_generate`. Generation is greedy everywhere -
    reproducibility is a hard requirement, since the checkpoint is the source of truth and
    a resumed run must reproduce pre-crash predictions. It also matters for the shuffle
    diagnostic: accuracy must vary because option ORDER changed, not because sampling did.
    """

    def __init__(self, model_key: str, device: str = "cuda:0", dtype=torch.bfloat16):
        if model_key not in ANSWERER_SPECS:
            raise ValueError(f"unknown model {model_key!r}; known: {list(ANSWERER_SPECS)}")
        self.model_key = model_key
        self.spec = ANSWERER_SPECS[model_key]
        self.model_id = self.spec["model_id"]
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self._load()

    def _load(self):
        raise NotImplementedError

    def _generate(self, image, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

    def answer(self, image_path: str | None, prompt: str, max_new_tokens: int = 16) -> str:
        """One prediction. `image_path=None` is the blind condition.

        16 tokens is plenty for a letter and enough to catch a short preamble like
        "The answer is B." Models that ignore the format are caught by the parser rather
        than by a longer budget - see the Qwen judge, which burned 12288 tokens per call
        on Benchmark 5 without ever emitting an answer."""
        image = Image.open(image_path).convert("RGB") if image_path else None
        with torch.inference_mode():
            return self._generate(image, prompt, max_new_tokens)

    def memory_summary(self) -> str:
        if not torch.cuda.is_available():
            return "cuda unavailable"
        idx = torch.device(self.device).index or 0
        allocated = torch.cuda.memory_allocated(idx) / 1024**3
        reserved = torch.cuda.memory_reserved(idx) / 1024**3
        return f"allocated {allocated:.2f} GiB | reserved {reserved:.2f} GiB"


class Qwen25VLAnswerer(AnswererModel):
    """Qwen2.5-VL-7B and its finetunes (Patho-R1)."""

    def _load(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=self.device,
        )
        self.model.eval()

    def _build_messages(self, image, prompt: str) -> list:
        content = ([{"type": "image", "image": image}] if image is not None else []) + \
                  [{"type": "text", "text": prompt}]
        return [{"role": "user", "content": content}]

    def _generate(self, image, prompt: str, max_new_tokens: int) -> str:
        messages = self._build_messages(image, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image] if image is not None else None,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False)
        # Decode ONLY the new tokens - the prompt echo would otherwise be parsed as the
        # answer, and every option letter appears in it.
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class InternVLAnswerer(AnswererModel):
    """InternVL3.5-8B - same calling convention as the InternVL judge."""

    def _load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=self.device,
        )
        self.model.eval()

    def _generate(self, image, prompt: str, max_new_tokens: int) -> str:
        content = ([{"type": "image", "image": image}] if image is not None else []) + \
                  [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        # InternVL's documented convention: one call does templating, tokenization and
        # image preprocessing together.
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class Gemma3Answerer(AnswererModel):
    """MedGemma-4b (Gemma3 multimodal)."""

    def _load(self):
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=self.dtype, device_map=self.device,
        )
        self.model.eval()

    def _generate(self, image, prompt: str, max_new_tokens: int) -> str:
        content = ([{"type": "image", "image": image}] if image is not None else []) + \
                  [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device, dtype=self.dtype)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


LOADERS = {
    "qwen25vl": Qwen25VLAnswerer,
    "internvl": InternVLAnswerer,
    "gemma3": Gemma3Answerer,
    # "llava_legacy" lives in answerer_llava.py and runs in the `path-llava` env - it
    # cannot be imported here, because this module uses transformers 5.x APIs that do not
    # exist there. See ENVIRONMENT.md.
}

# Per-model generation budgets, measured rather than assumed. The default of 16 tokens is
# right for models that obey "respond with only the letter"; it silently truncates the
# ones that do not, scoring a correct answer as unparsed.
#
#   qwen25vl / internvl8b / medgemma   16   emit a bare letter
#   llavamed / quiltllava              96   answer in prose (see answerer_llava.py)
#   pathor1                           see below - a reasoning model, needs far more
MAX_NEW_TOKENS = {
    "qwen25vl": 16,
    "internvl8b": 16,
    "medgemma": 16,
    "llavamed": 96,
    "quiltllava": 96,
    # Patho-R1 was trained with chain-of-thought + RL and sometimes emits a "<|P>**Step 1:
    # ..." reasoning trace before committing, so a short budget truncates mid-reasoning.
    # Swept on 20 PathOPEN MCQs - parse rate and accuracy both plateau at 256:
    #
    #   maxtok   parsed   correct   s/item
    #      16     12/20     9/20      1.10
    #      64     14/20     8/20      1.58
    #     256     17/20     9/20      2.77   <- plateau
    #     512     17/20     9/20      2.92
    #     768     17/20     9/20      2.92
    #
    # 256 buys +5 parsed answers over 16 tokens for 2.5x the time. Going past it costs
    # more time for nothing, so this is the knee of the curve, not a guess.
    "pathor1": 256,
}


def max_new_tokens_for(model_key: str) -> int:
    return MAX_NEW_TOKENS.get(model_key, 16)


def load_answerer(model_key: str, device: str = "cuda:0", **kwargs) -> AnswererModel:
    spec = ANSWERER_SPECS[model_key]
    loader = spec["loader"]
    if loader == "llava_legacy":
        raise NotImplementedError(
            f"{model_key} runs in the `path-llava` environment. Use "
            "`from answerer_llava import load_llava_answerer` there, not this module."
        )
    if loader not in LOADERS:
        raise NotImplementedError(
            f"{model_key} uses loader {loader!r}, which is not wired up yet. "
            "Run its smoke test first."
        )
    return LOADERS[loader](model_key, device=device, **kwargs)
