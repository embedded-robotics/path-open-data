# Environments

Two conda environments are required. This is not accidental complexity - the two LLaVA
models cannot run in the main environment, and the reason is worth stating so nobody
tries to "simplify" it later.

## `path-opendata-vlms` (main)

torch 2.11.0+cu128, transformers 5.14.1, python 3.14.

Runs everything: the judge pipeline in `../vlm_judge/`, and four of the six answerers -
Qwen2.5-VL-7B, InternVL3.5-8B, MedGemma-4b, Patho-R1-7B.

## `path-llava` (LLaVA models only)

torch 2.0.1+cu118, transformers 4.36.2, python 3.10.

Runs `microsoft/llava-med-v1.5-mistral-7b` and `wisdomik/Quilt-Llava-v1.5-7b`.

**Why a second environment.** Both are original-LLaVA-repo checkpoints whose model
classes (`LlavaMistralForCausalLM`, `LlavaLlamaForCausalLM`) do not exist in transformers,
and no HF-format conversion has been published for either. The official `llava-torch`
package supplies them but pins `torch==2.0.1` and `transformers==4.36.2` - installing it
into the main environment would downgrade the stack the judges and the other four
answerers depend on.

The alternative was hand-writing the multimodal forward pass (CLIP tower + mlp2x_gelu
projector + splicing image embeddings at the `<image>` sentinel). That was rejected in
favour of the official code path: a hand-rolled splice that is subtly wrong still produces
fluent output, and the judge pipeline has already been burned once by a model that loaded
cleanly and generated confident nonsense.

### Setup as performed

```bash
conda create -n path-llava python=3.10 -y
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2"              # torch 2.0.1 is built against NumPy 1.x
pip install llava-torch==1.2.2.post1
pip install "bitsandbytes==0.42.0" # 0.41.0 cannot find CUDA on this box
pip install nvidia-cusparse-cu11   # supplies libcusparse.so.11
pip install protobuf               # SentencePiece -> fast tokenizer conversion
```

### Required at runtime

```bash
export LD_LIBRARY_PATH=/data/mn27889/miniconda3/envs/path-llava/lib/python3.10/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH
```

bitsandbytes imports at `llava` package init and hard-fails without `libcusparse.so.11`,
even though quantization is never used here. The box has `.so.12` only, so the cu11
library is installed into the env and put on the library path. `answerer_llava.py` sets
this automatically; it only matters if you invoke python directly.

**Do not upgrade bitsandbytes past 0.42.0** in this env - 0.43+ requires torch>=2.4 and
pip will silently pull torch 2.13, which defeats the entire point of the pinned env.
This happened once during setup and had to be reverted.

### Two behavioural differences, measured

**`device_map` must be a string, not a dict.** `llava/model/builder.py:158` passes
`device_map` straight into `vision_tower.to(device=...)`, which raises `TypeError` on a
dict. Pass `device_map="cuda:0"`.

**These models need a far larger token budget.** They ignore "respond with only the
letter" and answer in prose. Measured on 12 PathOPEN MCQs with LLaVA-Med:

| `max_new_tokens` | replies parsed |
|---|---|
| 16 (fine for the other four models) | **2/12** |
| 96 | **7/12** |

So the shared 16-token default silently discards most of this model's answers. The
adapter sets 96 for LLaVA-family models. Parse rate is a per-model property that must be
reported alongside accuracy - a model that will not follow the output format is a real
finding, but it must not be confused with one that does not know the answer.

**Blind-condition caveat.** With no image, LLaVA-Med still describes "a histological
section of a tumor" - unlike MedGemma, which notices the absence and describes an
abstract painting. The blind condition is still valid (the image genuinely is not passed),
but this model confabulates rather than refusing, so its blind accuracy measures text
priors plus confabulation, and the paper should say so.
