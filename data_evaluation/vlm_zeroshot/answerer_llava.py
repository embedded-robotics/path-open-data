"""Adapter for the two original-LLaVA-repo models.

**Runs in the `path-llava` conda environment only** (torch 2.0.1 / transformers 4.36.2).
It cannot be imported from the main environment - see ENVIRONMENT.md for why the split
exists and how the env was built.

    microsoft/llava-med-v1.5-mistral-7b   Mistral backbone -> "mistral_instruct" template
    wisdomik/Quilt-Llava-v1.5-7b          Llama backbone   -> "v1" template

Kept in its own module rather than added to `answerer_models.py` because that module
imports transformers 5.x APIs that do not exist in this environment. The two files share
the `AnswererModel` interface by convention - `answer(image_path, prompt)` returning a
string - so the runner treats all six models identically.
"""
from __future__ import annotations

import os

def _preload_cusparse() -> None:
    """Make libcusparse.so.11 resolvable before `llava` is imported.

    bitsandbytes loads at `llava` package init and hard-fails without this library, even
    though quantization is never used here. This box ships .so.12 only, so the cu11
    library is pip-installed into the env (see ENVIRONMENT.md).

    Setting os.environ["LD_LIBRARY_PATH"] here would NOT work: the dynamic loader reads
    that variable once, when the process starts, so mutating it from inside Python is too
    late. Loading the library explicitly with RTLD_GLOBAL puts its symbols in the global
    namespace, which is what the later dlopen actually needs."""
    import ctypes
    import glob
    import sysconfig

    pattern = os.path.join(sysconfig.get_paths()["purelib"],
                           "nvidia", "cusparse", "lib", "libcusparse.so.11")
    for candidate in glob.glob(pattern):
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue
    # Not fatal on its own - if LD_LIBRARY_PATH was already exported by the caller, the
    # import below still succeeds. The llava import is what will fail loudly if not.


_preload_cusparse()

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

LLAVA_SPECS = {
    "llavamed": {
        "model_id": "microsoft/llava-med-v1.5-mistral-7b",
        "conv_template": "mistral_instruct",
        "tier": "biomedical",
        "params": "7B",
        # 96, not the 16 the other four models use. Measured on 12 PathOPEN MCQs: at 16
        # tokens only 2/12 replies parse, at 96 tokens 7/12 (and 24/30 once the
        # token-overlap rule was added). This model ignores "respond with only the letter"
        # and answers in prose, so a short budget truncates before the answer appears.
        "max_new_tokens": 96,
    },
    "quiltllava": {
        "model_id": "wisdomik/Quilt-Llava-v1.5-7b",
        # Llama backbone, so "v1" (aka "vicuna_v1"), NOT "mistral_instruct". This is not
        # cosmetic: the wrong template produced a different description of the same slide
        # ("eosinophils" vs "histiocytes") - plausible output that is silently wrong,
        # which is the hardest kind of bug to notice.
        "conv_template": "v1",
        "tier": "pathology",
        "params": "7B",
        # Measured 100% parse rate at 96 tokens (27/30 bare letters). It follows the
        # output format, unlike LLaVA-Med. The budget is kept the same for comparability.
        "max_new_tokens": 96,
    },
}


class LlavaAnswerer:
    """Same interface as `answerer_models.AnswererModel`: `answer(image_path, prompt)`."""

    def __init__(self, model_key: str, device: str = "cuda:0"):
        if model_key not in LLAVA_SPECS:
            raise ValueError(f"unknown LLaVA model {model_key!r}; known: {list(LLAVA_SPECS)}")
        self.model_key = model_key
        self.spec = LLAVA_SPECS[model_key]
        self.model_id = self.spec["model_id"]
        self.device = device

        # device_map must be a STRING. llava/model/builder.py:158 passes it straight into
        # `vision_tower.to(device=...)`, which raises TypeError on a dict - so the usual
        # {"": 0} form fails inside the library, not in our code.
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(self.model_id, None,
                                  get_model_name_from_path(self.model_id),
                                  device_map=device)
        self.model.eval()
        # `model.device` is unreliable on these builds; read it off a parameter.
        self.torch_device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype

    def answer(self, image_path: str | None, prompt: str,
               max_new_tokens: int | None = None) -> str:
        """One prediction. `image_path=None` is the blind condition.

        Note both models CONFABULATE when the image is withheld - LLaVA-Med still
        describes "a histological section of a tumor", where MedGemma correctly notices
        the absence. The condition is valid (no image is passed), but blind accuracy here
        measures text priors plus confabulation, which the paper should state."""
        if max_new_tokens is None:
            max_new_tokens = self.spec["max_new_tokens"]

        images = image_sizes = None
        text = prompt
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
            images = process_images([image], self.image_processor, self.model.config) \
                .to(self.torch_device, dtype=self.dtype)
            image_sizes = [image.size]
            # The sentinel must lead the turn; tokenizer_image_token splices the image
            # embeddings in at IMAGE_TOKEN_INDEX (-200), which is not a real vocab id.
            text = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_templates[self.spec["conv_template"]].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)

        input_ids = tokenizer_image_token(
            conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.torch_device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids, images=images, image_sizes=image_sizes,
                do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            )
        # These builds return ONLY the completion, not the prompt echo - unlike the
        # transformers-native models, which need the prompt sliced off.
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def memory_summary(self) -> str:
        if not torch.cuda.is_available():
            return "cuda unavailable"
        index = torch.device(self.torch_device).index or 0
        allocated = torch.cuda.memory_allocated(index) / 1024**3
        return f"allocated {allocated:.2f} GiB"


def load_llava_answerer(model_key: str, device: str = "cuda:0") -> LlavaAnswerer:
    return LlavaAnswerer(model_key, device=device)
