# VLM-as-Answerer (Pillars 3b and 4)

Runs candidate VLMs **as answerers** — the model attempts each question and is scored on
whether it got the answer right. This is a different pipeline from `../vlm/`, where a
large VLM acts as a **judge** grading text that already exists.

| | `../vlm/` (judge) | this folder (answerer) |
|---|---|---|
| model's job | grade an existing answer against a rubric | produce an answer |
| output | ordinal score {-1,0,1,2} | predicted option / free text |
| metric | weighted κ, Gwet's AC1 vs. pathologists | accuracy, BLEU, BERTScore |
| models | 2 large judges (32B/38B) | 6 candidates (4–8B) |
| paper | Pillars 1, 2, 3a, 5 | **Pillars 3b, 4a, 4b, 4c** |

## What the paper asks for

**Sub-pillar 3b — shortcut resistance** (Figure 6, 2 panels)
Zero-shot MCQ accuracy on PathOPEN, PathMMU, PatchVQA, plus two diagnostics:
*(i)* **blind accuracy** — the image is removed and only the text is shown. A model
scoring well blind is exploiting text priors rather than reading the slide.
*(ii)* **position-shuffle variance** — option order is randomised across repeated runs.
A model whose accuracy swings with option order is keying on position, not content.

**Sub-pillar 4a — cross-dataset difficulty** (Figure 7, 2 panels)
All models zero-shot across all five datasets, each scored with its native metric:
accuracy for MCQ/close-ended; exact-match, BLEU, and BERTScore for open-ended.

**Sub-pillar 4b — resolution sensitivity** (Figure 8, 1–2 panels)
Accuracy grouped by PathOPEN's magnification metadata (20×–400×), tested with ANOVA or
a non-parametric equivalent.

**Sub-pillar 4c — organ/category stratification** (Figure 9, 1 panel)
Accuracy by organ and categorization, as an organs × models heatmap.

## Model set

Six models spanning four tiers of domain specialization. The tiering is the point:
Pillar 4's claim is that PathOPEN *discriminates* between models, so the set has to
contain models that genuinely differ in pathology competence.

| tier | model | params | notes |
|---|---|---|---|
| general | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | Patho-R1's base model — the pairing isolates what pathology tuning buys |
| general | `OpenGVLab/InternVL3_5-8B-HF` | 8B | same family as the judge, different scale |
| broad medical | `google/medgemma-4b-it` | 4B | medical but not pathology; tests transfer. **Gated** |
| biomedical | `microsoft/llava-med-v1.5-mistral-7b` | 7B | the canonical biomedical baseline |
| pathology | `wisdomik/Quilt-Llava-v1.5-7b` | 7B | trained on Quilt data — one of our comparator datasets |
| pathology SOTA | `WenchuanZhang/Patho-R1-7B` | 7B | reports 69.53% on PathMMU test_tiny. **Gated** |

Every model is 4–8B so each fits on **one** A6000. That is deliberate: at this size six
models run in parallel across the free GPUs; at 32B you could afford one, and
cross-model comparison is the entire point of 4a.

### Two cautions

- **Use `microsoft/llava-med-v1.5-mistral-7b`, not `katielink/llava-med-7b-pathvqa-delta`.**
  The latter is fine-tuned *on PathVQA*, which is one of our comparator datasets — it
  would be contaminated for this purpose.
- **LLaVA-family models need a different loading path.** `LlavaMistralForCausalLM` and
  `LlavaLlamaForCausalLM` do not load through `AutoModelForImageTextToText` the way the
  judges do. Each gets its own smoke-test notebook before being wired in, the same way
  InternVL was validated in `../vlm/intern_vl_testing.ipynb`.

## Layout

```
vlm_zeroshot/
  answerer_models.py        model registry + loading, one adapter per architecture
  tasks.py                  dataset loaders -> a common Task record
  metrics.py                accuracy, exact-match, BLEU, BERTScore
  smoke_tests/              one notebook per model, run BEFORE the full pipeline
  runners/                  one notebook per (model x pillar)
  answerer_output/          per-model predictions
  analysis/                 cross-model tables and figures
```

Checkpointing follows `../vlm/checkpoint.py` exactly: every model call is appended to
`checkpoints/{model_key}_{task}.jsonl` with an `fsync`, keyed by a stable `item_id`, so
a run is always resumable and re-running only scores what is missing. That pattern has
already survived several multi-day interruptions in the judge pipeline.

## Running

```bash
conda activate path-opendata-vlms          # either env works - see below
cd data_evaluation/vlm_zeroshot
python launch_all.py                       # all six models, detached
python launch_all.py --status              # progress
python launch_all.py --dry-run             # preview the commands
tail -f logs/<model>_mcq_*.log             # follow one model
```

**Which conda environment to activate: either.** `launch_all.py` imports only the standard
library, and `gpu_allocation.PYTHON` holds *absolute* interpreter paths, so each model is
started with the right Python regardless of what is active in your shell - verified by
running `--dry-run` from both envs and diffing the output.

`path-opendata-vlms` is still the one to prefer, simply because everything else in this
repo (the analysis notebooks, the judge pipeline in `../vlm/`) needs it.

A single model can also be run directly, which is the easiest way to debug one in
isolation - but then the interpreter DOES matter:

```bash
# main-env model
/data/mn27889/miniconda3/envs/path-opendata-vlms/bin/python run_answerer.py --model qwen25vl
# LLaVA model - must be the path-llava interpreter
/data/mn27889/miniconda3/envs/path-llava/bin/python run_answerer.py --model llavamed
```

Scope: **6 models x 1,428 tasks x 5 conditions = 42,840 calls**, two models per GPU on
cards 3-5, roughly 2-4 hours. GPUs 0-2 are left alone for the judge run in `../vlm/`.
Every call is checkpointed, so a kill and restart costs only the in-flight item.

## Status

Scaffolding in progress. Nothing has been run yet.

Note that `../vlm/` will be renamed to `../vlm_judge/` once the Qwen Benchmark 5 run
finishes — its checkpoint paths are absolute and resolved at process start, so renaming
the directory while it runs would silently orphan ~9 days of GPU time.
