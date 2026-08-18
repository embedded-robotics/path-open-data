# %% [markdown]
# # MCQ Option Validity and Shortcut Resistance — PathOPEN, PathMMU, PatchVQA
# 
# Implements Pillar 3 (Dual MCQ Evaluation) from the paper, applying **Benchmark 5**
# (sub-benchmarks 5a and 5b) to three MCQ datasets. Benchmark 5 evaluates whether MCQ
# **options** - both correct and wrong - function as standalone, contextually complete
# clinical statements independent of the multiple-choice framing, and whether the
# correct option requires genuine visual grounding.
# 
# Benchmark 5 is the only benchmark applied here (unlike `quiltvqa_eval_runner.ipynb`,
# which spans two), so the rubric name is accurate - but the notebook is named for the
# *question* it answers rather than the rubric number, because it now covers three
# datasets rather than PathOPEN alone.
# 
# ## The three datasets
# 
# | dataset | source | scope |
# |---|---|---|
# | **PathOPEN** | this repo | every resolvable case (~229): correct option (5a) + 4 wrong options (5b) |
# | **PathMMU** | `/data/mn27889/pathology-datasets/PathMMU/data.json` | `test_tiny` split, 4 resolvable categories (PubMed, EduContent, PathCLS, Atlas) |
# | **PatchVQA** | the patch-level subset of **PathBench**, `/data/mn27889/pathology-datasets/PathBench/data/PatchVQA.json` | `test_tiny` split, PathCLS excluded as a duplicate of PathMMU's |
# 
# **PatchVQA vs. PathBench**: PathBench is the parent benchmark and ships three subsets -
# PatchVQA (patch-level MCQs), WSICap, and WSIVQA. Only **PatchVQA** is scored here;
# WSICap and WSIVQA are whole-slide-image datasets requiring gigapixel `.svs` files
# downloaded separately from the GDC, which are not available locally. So "PatchVQA" is
# the precise label throughout, and it should be attributed to PathBench in the paper.
# PatchVQA's images come from PathMMU's pool (per PathBench's own README), which is why
# both datasets share one images directory.
# 
# ## No human baseline
# 
# Unlike Benchmarks 1-4, **Benchmark 5 was never scored by human pathologists at all** -
# the paper states explicitly (Section 3.5.4): "Since pathologist recruitment for
# datasets outside PathVQA was not feasible at this scale, Benchmark 5 scores for
# PathMMU and PatchVQA were obtained using an LLM-as-judge protocol". Per agreed scope,
# this notebook reports **judge-vs-judge agreement** (InternVL vs. Qwen-VL) as the only
# reliability signal, treated as an explicit limitation rather than a substitute for
# human validation.
# 
# **Critical methodological detail**: Benchmark 5a/5b score each option **standalone** -
# "Standalone Completeness was scored using option text only, without question-stem
# context" (Section 2.1, Sub-pillar 3a). This notebook's prompts
# (`build_benchmark_5a_prompt` / `build_benchmark_5b_prompt`) never include the question
# text, unlike every other benchmark notebook in this repo.
# 
# ## Cost
# 
# Roughly **7,000 judge calls per judge, ~4 days** with both judges concurrent. Scoring
# all splits instead would be ~78,000 calls and ~39 days, which is why PathMMU and
# PatchVQA are sampled; `BENCHMARK5_SPLITS` controls it.
# 
# **Known data gaps** (pre-existing, not introduced here): the `SocialPath` category is
# 100% unresolvable for both PathMMU and PatchVQA (images were never downloaded - the
# README documents a separate manual acquisition step), and `PathCLS` is ~9%
# unresolvable. Unresolvable items are skipped and counted, not silently dropped.
# 
# ## Checkpointing
# 
# Every individual judge call is checkpointed immediately to
# `checkpoints/{model_key}_benchmark5.jsonl`, keyed by
# `item_id = "{dataset}::{category}::{split}::{No}::{option_slot}"` (`option_slot`
# is `"correct"` or `"wrong_N"`). Re-running any scoring cell after an
# interruption skips everything already checkpointed and only scores what's
# missing. Because `split` is part of the key, **widening `BENCHMARK5_SPLITS`
# later is purely additive** - already-scored `test_tiny` items are never redone.
# 

# %%
import os
import sys

sys.path.insert(0, os.getcwd())  # so gpu_allocation/judge_models resolve when the CWD is this dir
from gpu_allocation import cuda_visible_devices_for, describe_allocation, max_memory_for

# Which judge(s) this kernel will load. Declared HERE, before torch touches CUDA,
# because CUDA_VISIBLE_DEVICES has no effect once CUDA is initialized - if a torch CUDA
# op has already run in this kernel, restart it.
#
# Both judges run unquantized at bf16 (~64 GB Qwen / ~76 GB InternVL), so each is
# sharded over 3 of the 47.4 GiB A6000s. They are placed in different NUMA islands
# (Qwen 0-2, InternVL 4-6) so they can run concurrently without sharing a PCIe switch.
# MODELS_TO_RUN = ["qwenvl", "internvl"]
MODELS_TO_RUN = ["qwenvl"]


os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices_for(*MODELS_TO_RUN)
print(describe_allocation())

# %%
import glob
import json as _json
import os

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from checkpoint import JudgeCheckpoint
from judge_models import JudgeModel
from parallel_judges import run_judges_in_parallel
from prompts.benchmarks import (
    BENCHMARK_5A,
    BENCHMARK_5B,
    build_benchmark_5a_prompt,
    build_benchmark_5b_prompt,
)


# %%
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
SUBSETS_DIR = os.path.join(
    REPO_ROOT, "data_evaluation", "pathologists", "subsets_processing_output", "data"
)
PROCESSED_DATA_JSON = os.path.join(REPO_ROOT, "pathopen_data", "processed", "data.json")

PATHMMU_ROOT = "/data/mn27889/pathology-datasets/PathMMU"
PATHMMU_DATA_JSON = os.path.join(PATHMMU_ROOT, "data.json")
PATHMMU_IMAGES_DIR = os.path.join(PATHMMU_ROOT, "images")

# PatchVQA is PathBench's patch-level subset (PathBench also ships WSICap and WSIVQA,
# both of which need gigapixel .svs files from the GDC and are not scored here).
PATCHVQA_JSON = "/data/mn27889/pathology-datasets/PathBench/data/PatchVQA.json"
# PatchVQA's images are sourced from PathMMU's image pool (documented in PathBench's
# own README: "Access images can from PathMMU"), not a separate directory.
PATCHVQA_IMAGES_DIR = PATHMMU_IMAGES_DIR

# Same layout as every other runner: judge_output/evaluator_{model_key}/{name}.csv,
# with cross-dataset statistics in agreement_output/.
JUDGE_OUTPUT_DIR = os.path.join(os.getcwd(), "judge_output")
AGREEMENT_OUTPUT_DIR = os.path.join(os.getcwd(), "agreement_output")
CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(AGREEMENT_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def judge_output_path(model_key: str, filename: str) -> str:
    """judge_output/evaluator_{model_key}/{filename}, creating the per-judge dir."""
    directory = os.path.join(JUDGE_OUTPUT_DIR, f"evaluator_{model_key}")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)


REPO_ROOT, SUBSETS_DIR, PATHMMU_DATA_JSON, PATCHVQA_JSON, CHECKPOINT_DIR


# %% [markdown]
# ## Load judge models

# %%
# MODELS_TO_RUN is set in the first cell (it has to be, to pick GPUs before CUDA init).
# max_memory_for(...) returns LOGICAL device ids - CUDA_VISIBLE_DEVICES renumbers cards,
# so with "4,5,6" visible torch sees 0,1,2.
judges = {
    key: JudgeModel(key, max_memory=max_memory_for(key, MODELS_TO_RUN))
    for key in MODELS_TO_RUN
}

# Verify each judge loaded as intended BEFORE committing to a multi-hour run.
# device_map="auto" silently offloads to CPU/disk when a model does not fit, which turns
# a long run into a multi-day one - catch it here, not hours in.
import collections

import torch

for key, judge in judges.items():
    devs = collections.Counter(str(p.device) for p in judge.model.parameters())
    offloaded = [d for d in devs if d in ("cpu", "meta", "disk")]
    print(f"{key}: devices={dict(devs)}")
    print(f"    system_role={judge.use_system_role}  images_in_template={judge.images_in_template}"
          f"  thinking={'<think>' in judge.system_prompt}")
    print(f"    OFFLOAD -> {offloaded if offloaded else 'none (good)'}")
    if offloaded:
        raise RuntimeError(f"{key} offloaded to {offloaded} - lower max_memory or add a card")

judges

# %% [markdown]
# ## PathOPEN: MCQ options, standalone
# 
# Reuses the same `pathopen_vqa_part{1-5}.csv` source subset and `processed/data.json`
# image resolver as `judge_runner_pathopen.ipynb`, but scores MCQ options standalone
# (no question text in the prompt) rather than as open-ended Q&A.

# %%
with open(PROCESSED_DATA_JSON) as f:
    _processed_cases = _json.load(f)
_processed_root = os.path.dirname(PROCESSED_DATA_JSON)

PATHOPEN_IMAGE_INDEX = {}
for _case in _processed_cases:
    for _img in _case["Images"]:
        original = _img.get("Original", "")
        if not original:
            continue
        image_id = original.rsplit("_orig.", 1)[0]
        rel_dir = _img["Directory"].split("processed/")[-1]
        abs_dir = os.path.normpath(os.path.join(_processed_root, rel_dir))
        PATHOPEN_IMAGE_INDEX[image_id] = {"dir": abs_dir, "original": original}


def load_pathopen_image(image_id: str) -> Image.Image:
    entry = PATHOPEN_IMAGE_INDEX[image_id]
    return Image.open(os.path.join(entry["dir"], entry["original"])).convert("RGB")


core_parts = sorted(glob.glob(os.path.join(SUBSETS_DIR, "pathopen_vqa_part*.csv")))
pathopen_df = pd.concat([pd.read_csv(p) for p in core_parts], ignore_index=True)
pathopen_df = pathopen_df[pathopen_df["Image_ID"].isin(PATHOPEN_IMAGE_INDEX)]
len(pathopen_df)


# %%
def iter_pathopen_mcq_subtasks(row: pd.Series):
    """Yields (option_slot, prompt, criteria, option_text) for one PathOPEN row's
    correct option (5a) and 4 wrong options (5b)."""
    yield (
        "correct",
        build_benchmark_5a_prompt(row["MCQ_OE_Correct_Answer"]),
        list(BENCHMARK_5A["criteria"].keys()),
        row["MCQ_OE_Correct_Answer"],
    )
    for i in (1, 2, 3, 4):
        wrong_col = f"MCQ_OE_Wrong_Answer_{i}"
        yield (
            f"wrong_{i}",
            build_benchmark_5b_prompt(row[wrong_col]),
            list(BENCHMARK_5B["criteria"].keys()),
            row[wrong_col],
        )


def run_pathopen_benchmark5(judge: JudgeModel, model_key: str) -> JudgeCheckpoint:
    checkpoint = JudgeCheckpoint(os.path.join(CHECKPOINT_DIR, f"{model_key}_benchmark5.jsonl"))
    n_scored = n_skipped = n_failed = 0

    for _, row in tqdm(pathopen_df.iterrows(), total=len(pathopen_df), desc=f"{model_key} PathOPEN Benchmark5"):
        image = None
        for option_slot, prompt, criteria, option_text in iter_pathopen_mcq_subtasks(row):
            item_id = f"PathOPEN::None::None::{row['Image_ID']}::{option_slot}"
            if checkpoint.is_done(item_id):
                n_skipped += 1
                continue
            try:
                if image is None:
                    image = load_pathopen_image(row["Image_ID"])
                scores, raw = judge.score(image, prompt, criteria)
            except Exception as e:
                n_failed += 1
                print(f"[checkpoint] FAILED item_id={item_id!r}: {e!r} - will retry next run")
                continue
            checkpoint.append({
                "item_id": item_id,
                "dataset": "PathOPEN",
                "source_category": None,
                "split": None,
                "row_no": row["Image_ID"],
                "option_slot": option_slot,
                "option_text": option_text,
                "scores": scores,
                "raw_response": raw,
            })
            n_scored += 1

    print(f"{model_key} PathOPEN Benchmark5: scored {n_scored} new, {n_skipped} already done, {n_failed} failed this run")
    return checkpoint


# Judges run concurrently - separate GPUs, separate checkpoint files. See parallel_judges.py.
# Benchmark 5 needs BOTH judges: with no human baseline, judge-vs-judge agreement is its
# only reliability signal, so a single-judge run produces no usable number here.
pathopen_b5_checkpoints = run_judges_in_parallel(run_pathopen_benchmark5, judges, MODELS_TO_RUN)


# %% [markdown]
# ## PathMMU and PatchVQA: MCQ options, standalone
# 
# Both datasets share the same JSON shape: `{category: {split: [ {img, question,
# options, answer}, ... ]}}`, with `options` as letter-prefixed strings (e.g. `"A)
# ..."`) and `answer` matching one option's full text exactly. Wrong options are
# simply every option that isn't the answer - a variable count (2-16 per question,
# against PathOPEN's fixed 5).
# 
# ### Scope: `test_tiny` only
# 
# Scoring every split is not feasible. One judge call per option gives **73,034
# calls per judge** across the two datasets; at the ~46 s/call measured on the
# completed PathOPEN and PathVQA runs that is **~39 days** of wall-clock even with
# both judges running concurrently.
# 
# `test_tiny` is PathMMU's own curated representative subset, which makes it a
# defensible published sampling choice rather than an arbitrary cut of our own. It
# covers every category and reduces the job to **~5,700 calls per judge (~3 days)**:
# 
# | | items | resolvable | calls/judge |
# |---|---|---|---|
# | PathMMU `test_tiny` | 1,156 | 905 | 4,348 |
# | PatchVQA `test_tiny` (PathCLS removed) | 430 | 294 | 1,478 |
# 
# Benchmark 5 measures a property of the *dataset's option writing*, which is an
# aggregate claim - a representative sample supports it with a confidence interval,
# where a census would only narrow that interval. Set `BENCHMARK5_SPLITS` below to
# widen the scope; the checkpoint makes any later expansion purely additive, since
# already-scored items are skipped rather than redone.
# 
# ### PathCLS is shared between the two datasets
# 
# PatchVQA's `PathCLS` category is the *same data* as PathMMU's - identical `img`,
# `question`, `options` and `answer` across all 177 `test_tiny` items. It is scored
# once (under PathMMU) and dropped from PatchVQA, which both saves ~1,370 calls per
# judge and keeps a PathMMU-vs-PatchVQA comparison from partly comparing a set with
# itself.

# %%
def load_pathmmu_style_dataset(json_path: str, splits=("test_tiny",)) -> list:
    """Flattens {category: {split: [items]}} into a flat list of dict records,
    each tagged with its source category and split.

    `splits` restricts which splits are kept. Default is `test_tiny` only - see the
    markdown above for why the full set is not scored. Pass `splits=None` to load
    everything (~39 days of judge time; not recommended)."""
    with open(json_path) as f:
        data = _json.load(f)
    records = []
    for category, splits_dict in data.items():
        for split, items in splits_dict.items():
            if splits is not None and split not in splits:
                continue
            for item in items:
                records.append({**item, "_category": category, "_split": split})
    return records


def resolve_pathmmu_style_image(images_dir: str, img_filename: str):
    path = os.path.join(images_dir, img_filename)
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGB")


# Which splits to score. PathMMU publishes `test_tiny` as a curated representative
# subset of `test`, so using it is the dataset authors' own sampling choice rather
# than an arbitrary cut of our own - which is what makes it defensible in the paper.
BENCHMARK5_SPLITS = ("test_tiny",)

pathmmu_items = load_pathmmu_style_dataset(PATHMMU_DATA_JSON, BENCHMARK5_SPLITS)
patchvqa_items = load_pathmmu_style_dataset(PATCHVQA_JSON, BENCHMARK5_SPLITS)

# PathCLS is byte-for-byte shared between the two datasets: verified identical `img`,
# `question`, `options` and `answer` for all 177 test_tiny items (they differ only in
# PathMMU's extra `explanation` field, which Benchmark 5 never reads). Scoring it under
# both names would waste ~1,370 calls per judge AND silently corrupt the headline
# comparison - a PathMMU-vs-PatchVQA contrast would be partly a set compared with
# itself. It is kept under PathMMU and dropped from PatchVQA; the paper must therefore
# describe PatchVQA's Benchmark 5 numbers as covering its PathMMU-disjoint categories.
SHARED_CATEGORIES = {"PathCLS"}
_before = len(patchvqa_items)
patchvqa_items = [it for it in patchvqa_items if it["_category"] not in SHARED_CATEGORIES]
print(f"PatchVQA: dropped {_before - len(patchvqa_items)} items in shared categories "
      f"{sorted(SHARED_CATEGORIES)} (scored once, under PathMMU)")

# Report the real workload before committing to it, counting one judge call per option.
for _name, _items in [("PathMMU", pathmmu_items), ("PatchVQA", patchvqa_items)]:
    _res = [it for it in _items
            if os.path.exists(os.path.join(PATHMMU_IMAGES_DIR, it["img"]))]
    _calls = sum(len(it["options"]) for it in _res)
    print(f"{_name}: {len(_items)} items, {len(_res)} resolvable, {_calls} judge calls/judge")

len(pathmmu_items), len(patchvqa_items)


# %%
def iter_pathmmu_style_subtasks(item: dict):
    options = item["options"]
    answer = item["answer"]
    wrong_options = [opt for opt in options if opt != answer]

    yield ("correct", build_benchmark_5a_prompt(answer), list(BENCHMARK_5A["criteria"].keys()), answer)
    for i, wrong_option in enumerate(wrong_options, start=1):
        yield (
            f"wrong_{i}",
            build_benchmark_5b_prompt(wrong_option),
            list(BENCHMARK_5B["criteria"].keys()),
            wrong_option,
        )


def run_pathmmu_style_benchmark5(judge: JudgeModel, items: list, images_dir: str, dataset_name: str, model_key: str, checkpoint: JudgeCheckpoint) -> None:
    n_scored = n_skipped = n_failed = n_unresolvable = 0

    for item in tqdm(items, desc=f"{model_key} {dataset_name} Benchmark5"):
        item_no = item.get("No", item["img"])
        base_id = f"{dataset_name}::{item['_category']}::{item['_split']}::{item_no}"
        # Skip the whole item cheaply if every subtask is already checkpointed,
        # without touching disk/image loading at all.
        subtasks = list(iter_pathmmu_style_subtasks(item))
        if all(checkpoint.is_done(f"{base_id}::{slot}") for slot, *_ in subtasks):
            n_skipped += len(subtasks)
            continue

        image = resolve_pathmmu_style_image(images_dir, item["img"])
        if image is None:
            n_unresolvable += 1
            continue

        for option_slot, prompt, criteria, option_text in subtasks:
            item_id = f"{base_id}::{option_slot}"
            if checkpoint.is_done(item_id):
                n_skipped += 1
                continue
            try:
                scores, raw = judge.score(image, prompt, criteria)
            except Exception as e:
                n_failed += 1
                print(f"[checkpoint] FAILED item_id={item_id!r}: {e!r} - will retry next run")
                continue
            checkpoint.append({
                "item_id": item_id,
                "dataset": dataset_name,
                "source_category": item["_category"],
                "split": item["_split"],
                "row_no": item_no,
                "option_slot": option_slot,
                "option_text": option_text,
                "scores": scores,
                "raw_response": raw,
            })
            n_scored += 1

    print(f"{model_key} {dataset_name} Benchmark5: scored {n_scored} new, {n_skipped} already done, "
          f"{n_unresolvable} items unresolvable (image not found), {n_failed} failed this run")


# %%



# Also parallel across judges. Each judge still does PathMMU then PatchVQA in sequence
# (one model, one GPU set - there is nothing to overlap within a judge), but the two
# judges progress through both datasets at the same time.
def _run_other_datasets(judge, model_key):
    checkpoint = pathopen_b5_checkpoints[model_key]  # same checkpoint file covers all 3 datasets
    run_pathmmu_style_benchmark5(judge, pathmmu_items, PATHMMU_IMAGES_DIR, "PathMMU", model_key, checkpoint)
    run_pathmmu_style_benchmark5(judge, patchvqa_items, PATCHVQA_IMAGES_DIR, "PatchVQA", model_key, checkpoint)
    return checkpoint


run_judges_in_parallel(_run_other_datasets, judges, MODELS_TO_RUN)


# %% [markdown]
# ## Post-process: assemble the checkpoint into per-dataset CSVs
# 
# Pure re-read of the checkpoint **file** (via a fresh `JudgeCheckpoint(path)`, not
# the in-memory `pathopen_b5_checkpoints` object from the scoring cells above) -
# safe to re-run any time, independent of the scoring cells above, even in a fresh
# kernel (no judge models or GPU needed, just `os`/`pandas`/`json` and the
# `CHECKPOINT_DIR`/`JUDGE_OUTPUT_DIR` paths above).
# 
# Writes `judge_output/evaluator_{model_key}/{dataset}_benchmark5_eval_data.csv` — one
# file per (judge, dataset), matching the layout the PathOPEN, PathVQA, and Quilt-VQA
# runners use.

# %%
def assemble_benchmark5_csv(model_key: str, dataset_name: str) -> pd.DataFrame:
    """Reads checkpoints/{model_key}_benchmark5.jsonl directly from disk (one
    shared checkpoint file covers all 3 datasets) - does NOT depend on the
    `pathopen_b5_checkpoints` dict from the scoring cells, so this is safe to run
    standalone in a fresh kernel."""
    checkpoint = JudgeCheckpoint(os.path.join(CHECKPOINT_DIR, f"{model_key}_benchmark5.jsonl"))
    rows_by_key = {}
    for record in checkpoint.load_all():
        if record["dataset"] != dataset_name:
            continue
        key = (record["source_category"], record["split"], record["row_no"])
        row = rows_by_key.setdefault(key, {
            "dataset": dataset_name,
            "source_category": record["source_category"],
            "split": record["split"],
            "item_id": record["row_no"],
        })
        slot = record["option_slot"]
        if slot == "correct":
            row["correct_option_text"] = record["option_text"]
            row["correct_Standalone_Completeness"] = record["scores"].get("Standalone Completeness")
            row["correct_Visual_Grounding"] = record["scores"].get("Visual Grounding")
        else:
            row[f"{slot}_option_text"] = record["option_text"]
            row[f"{slot}_Distractor_Standalone_Plausibility"] = record["scores"].get("Distractor Standalone Plausibility")
            row[f"{slot}_Distractor_Visual_Grounding_Error"] = record["scores"].get("Distractor Visual Grounding Error")
    return pd.DataFrame.from_records(list(rows_by_key.values()))


# One CSV per (judge, dataset), in the shared judge_output/evaluator_{model_key}/ layout.
# The filename keeps "benchmark5" because that IS the rubric applied - unlike the
# notebook title, which names the three datasets it spans.
for model_key in MODELS_TO_RUN:
    for dataset_name in ["PathOPEN", "PathMMU", "PatchVQA"]:
        out_df = assemble_benchmark5_csv(model_key, dataset_name)
        out_path = judge_output_path(model_key, f"{dataset_name.lower()}_benchmark5_eval_data.csv")
        out_df.to_csv(out_path, index=False)
        print(model_key, dataset_name, "->", out_path, out_df.shape)


# %% [markdown]
# ## Judge-vs-judge agreement (the only reliability signal available for Benchmark 5)
# 
# Since there is no human ground truth for Benchmark 5, this computes weighted
# Cohen's kappa between InternVL and Qwen-VL's scores on the **same items**, per
# dataset and per criterion, as a lower bar of "do two independent judges even
# agree with each other" - not proof of correctness.

# %%
import numpy as np
from sklearn.metrics import cohen_kappa_score

VALID_SCORES = {-1, 0, 1, 2}


def judge_vs_judge_kappa(df_a: pd.DataFrame, df_b: pd.DataFrame, join_cols: list, score_col: str) -> dict:
    """Weighted kappa between the two judges over the FULL ordinal scale {-1, 0, 1, 2}.

    -1 is kept, consistent with judge_pathologist_agreement.ipynb and
    quiltvqa_eval_runner.ipynb: it is the rubric's "unable to comprehend / evaluate"
    level, so whether two judges agree an option is unscorable is part of what this
    measures, not noise to be filtered out first."""
    merged = df_a.merge(df_b, on=join_cols, suffixes=("_a", "_b"))
    a = pd.to_numeric(merged[f"{score_col}_a"], errors="coerce")
    b = pd.to_numeric(merged[f"{score_col}_b"], errors="coerce")
    valid = a.isin(VALID_SCORES) & b.isin(VALID_SCORES)
    n = int(valid.sum())
    if n < 2:
        return {"n": n, "n_neg1_internvl": 0, "n_neg1_qwenvl": 0, "weighted_kappa": np.nan}
    a_valid, b_valid = a[valid].astype(int), b[valid].astype(int)
    kappa = cohen_kappa_score(a_valid, b_valid, weights="quadratic")
    return {"n": n,
            "n_neg1_internvl": int((a_valid == -1).sum()),
            "n_neg1_qwenvl": int((b_valid == -1).sum()),
            "weighted_kappa": kappa}


# Both the 5a (correct-option) and 5b (distractor) criteria are compared. Scoring 5b but
# never analysing it would waste the majority of the run - most calls are distractors.
CORRECT_CRITERIA = ["correct_Standalone_Completeness", "correct_Visual_Grounding"]

jvj_rows = []
for dataset_name, join_cols in [
    ("PathOPEN", ["item_id"]),
    ("PathMMU", ["item_id", "source_category", "split"]),
    ("PatchVQA", ["item_id", "source_category", "split"]),
]:
    fname = f"{dataset_name.lower()}_benchmark5_eval_data.csv"
    path_internvl = judge_output_path("internvl", fname)
    path_qwenvl = judge_output_path("qwenvl", fname)
    missing = [p for p in (path_internvl, path_qwenvl) if not os.path.exists(p)]
    if missing:
        print(f"Skipping {dataset_name}: not found -> {missing}")
        continue
    df_internvl = pd.read_csv(path_internvl)
    df_qwenvl = pd.read_csv(path_qwenvl)

    # 5a: the correct option.
    for score_col in CORRECT_CRITERIA:
        result = judge_vs_judge_kappa(df_internvl, df_qwenvl, join_cols, score_col)
        jvj_rows.append({"dataset": dataset_name, "sub_benchmark": "5a",
                         "criterion": score_col, **result})

    # 5b: the distractors. Option count varies per question (2-16), so the wrong_N
    # columns are discovered from the data rather than assumed.
    wrong_cols = sorted(
        c for c in df_internvl.columns
        if c.startswith("wrong_") and c.endswith(
            ("_Distractor_Standalone_Plausibility", "_Distractor_Visual_Grounding_Error"))
        and c in df_qwenvl.columns
    )
    for score_col in wrong_cols:
        result = judge_vs_judge_kappa(df_internvl, df_qwenvl, join_cols, score_col)
        jvj_rows.append({"dataset": dataset_name, "sub_benchmark": "5b",
                         "criterion": score_col, **result})

judge_vs_judge_df = pd.DataFrame(jvj_rows)
if len(judge_vs_judge_df):
    print(judge_vs_judge_df.to_string(index=False))
judge_vs_judge_df


# %%
judge_vs_judge_df.to_csv(
    os.path.join(AGREEMENT_OUTPUT_DIR, "benchmark5_judge_vs_judge_kappa.csv"), index=False)


# %% [markdown]
# ## Limitations (explicit, per agreed scope)
# 
# - **No human baseline for Benchmark 5** - the reported judge-vs-judge kappa is a
#   weak reliability signal (two models could share the same blind spots) and
#   should not be reported as validation in the same sense as the Benchmark 1-4
#   kappas in `judge_pathologist_agreement.ipynb`.
# - **`test_tiny` split only** for PathMMU and PatchVQA - a representative sample,
#   not a census, so these numbers carry sampling error that PathOPEN's (scored in
#   full) do not. Report them with confidence intervals and state the split.
# - **PathCLS scored once, under PathMMU** - it is the same 177 items in both
#   datasets. PatchVQA's Benchmark 5 numbers therefore cover only its
#   PathMMU-disjoint categories, and pooling the two datasets would double-count
#   PathCLS.
# - **SocialPath category unresolvable** for both PathMMU and PatchVQA (documented
#   data-acquisition gap, images never downloaded locally) - excluded entirely,
#   not scored. This is ~20% of each dataset, and it is not a random 20%: it is a
#   single source category, so the remaining sample is not representative of the
#   full dataset with respect to image provenance.
# - **PathCLS ~9% unresolvable** - individual items skipped and counted, not
#   silently dropped.
# 
# ## Raw checkpoint files
# 
# `checkpoints/{model_key}_benchmark5.jsonl` retains every judge call across all
# three datasets (option text, scores, and full raw model response) and is never
# overwritten - it is the single source of truth this notebook's CSVs and kappa
# table are derived from, and can be re-processed at any time without re-running
# any inference.
# 


