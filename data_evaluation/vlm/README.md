# VLM-as-Judge Pipeline

This directory implements automated (VLM-based) scoring to replicate and extend
the pathologist evaluation described in the PathOPEN paper, using
**InternVL3.5-38B** and **Qwen3-VL-32B-Thinking** as judge models instead of
recruiting more human pathologists.

It reuses the raw datasets, subset CSVs, and rating scale already established by
`data_evaluation/pathologists/`, so a judge's output can be dropped into the
existing scoring-analysis notebooks as if it were an additional evaluator.

## Why this exists

The paper (`PathOPEN_Dataset.pdf`, supplied by the user) describes five
evaluation "pillars." Only some of them involve judging *answers against a
rubric* — the task an LLM/VLM-as-judge is suited to. The rest involve running
candidate VLMs as *answerers* and scoring their raw accuracy/BLEU/BERTScore, or
computing plain dataset statistics (entropy, Gini). This codebase deliberately
covers **only the judge-replaces-pathologist pieces**:

| Pillar | What it is | In scope here? |
|---|---|---|
| 1a — Dataset Quality (human-rated, PathOPEN vs. PathVQA) | Benchmarks 1 & 3 already rated by 5 pathologists | Yes — validate judges against these existing ratings |
| 1b — Judge-validated comparison vs. Quilt-VQA | Judge scores Quilt-VQA once validated | Yes |
| 2 — Wrong-Answer Preference Structure | Tier (near/moderate/far-miss) agreement | Yes |
| 3 — Dual MCQ Evaluation (Benchmark 5) | Standalone MCQ option scoring | Yes (sub-pillar 3a only) |
| 3b — Selection-based shortcut resistance | Blind-accuracy / position-shuffle *answerer* benchmarking | **No** — different pipeline (VLM-as-answerer), later phase |
| 4 — Benchmarking Utility | Cross-dataset zero-shot accuracy/BLEU/BERTScore, resolution/organ stratification | **No** — VLM-as-answerer + automatic metrics, later phase |
| 5 — Augmentation Strategy (5a) | Reuses existing Benchmark 4 pathologist ratings | Implicitly covered via Benchmark 4 judge scoring |
| 5 — Augmentation Strategy (5b) | Shannon entropy / Gini coefficient over metadata | **No** — pure dataset statistics, no judge involved |

## The five benchmarks (rubrics)

All scores are on the ordinal scale **{-1, 0, 1, 2}**, where `-1` always means
"unable to comprehend the question/image or unable to evaluate." Full rubric text
(verbatim from the paper's Tables 5–10) lives in `prompts/benchmarks.py`.

- **Benchmark 1** — Open-ended correct answers. Criteria: *Knowledge
  Interpretation/Deduction*, *Visual Grounding*. Applies to OE correct answers
  and MCQ correct options (posed as open-ended).
- **Benchmark 2** — Open-ended wrong answers. Criteria: *Error Proximity and
  Deductive Plausibility*, *Visual Grounding Error*. Applies to OE wrong answers
  and MCQ wrong options.
- **Benchmark 3** — Close-ended (yes/no) questions. Single criterion: *Visual
  Grounding/Reasoning*.
- **Benchmark 4** — Image augmentation. Criteria: *Clinical Relevance*, *Visual
  Grounding*. Scores an augmented image against the original question/answer
  pair.
- **Benchmark 5a/5b** — MCQ standalone validity. 5a scores the correct option,
  5b each wrong option, **using the option text alone, with no question stem**
  — a deliberately different prompt construction from Benchmarks 1–4. Criteria:
  *Standalone Completeness* / *Distractor Standalone Plausibility*, plus
  *Visual Grounding* / *Distractor Visual Grounding Error*.

**Benchmark 5 was never scored by human pathologists at all.** The paper states
this explicitly (§3.5.4): it was designed from the start to be judged by an
LLM/VLM, validated only informally against a held-out PathOPEN subset. There is
no human baseline to compare it to in this codebase; `benchmark5_mcq_standalone.ipynb`
reports judge-vs-judge agreement as the only reliability signal, and this is
treated as an explicit limitation.

## Checkpointing: running the model vs. post-processing are separate steps

Every notebook that calls a judge model (`judge_runner_pathopen.ipynb`,
`judge_runner_pathvqa.ipynb`, `benchmark5_mcq_standalone.ipynb`,
`quiltvqa_benchmark1.ipynb`) follows the same pattern, built around
`checkpoint.py`'s `JudgeCheckpoint` class:

1. **Every individual judge call is written to disk immediately** as one JSON
   line in a `checkpoints/{model_key}_{task}.jsonl` file — not held in a Python
   list until the whole loop finishes. The write is `fsync`'d, so a crash right
   after a call completes still leaves that result durably saved.
2. **Each record is keyed by a stable `item_id`** — e.g. `"{Image_ID}::OE_Wrong_1"`
   for a PathOPEN sub-task, or `"{dataset}::{category}::{split}::{No}::wrong_2"`
   for a Benchmark 5 MCQ option. Before scoring anything, the notebook checks
   whether that `item_id` is already in the checkpoint file and skips it if so.
3. **A single failed judge call does not abort the run.** If `judge.score(...)`
   raises (OOM, a generation error, anything), the notebook logs which
   `item_id` failed and continues to the next item. Nothing already
   checkpointed is lost, and the failed item is retried automatically the next
   time the same cell runs.
4. **Building the final CSV is a separate step from running the model.** Every
   runner notebook has a "post-process" cell, clearly marked in its own
   section, that does nothing but read the checkpoint file(s) back and
   reshape them into the evaluator-schema CSV (or the Benchmark-5/Quilt-VQA
   equivalent). This cell needs no GPU, no loaded judge model, and is always
   safe to re-run — including in a fresh kernel — as long as the checkpoint
   `.jsonl` file exists.

**PathOPEN-specific detail**: a single PathOPEN row needs up to 11 separate
judge calls (2 OE correct + 2 OE wrong + 1 MCQ correct + 4 MCQ wrong + 1 CE).
Each of those 11 calls is checkpointed individually, so an interruption
mid-row only loses the one in-flight call, not the rest of that row's
progress — the assembly step reconstructs each row by joining all of that
row's checkpointed sub-task records back together.

**`benchmark5_mcq_standalone.ipynb`-specific detail**: this is the largest
inference job in the pipeline (thousands of MCQ options across three
datasets), so it additionally does a cheap "is this whole item already fully
done?" check before even loading an image — if every sub-task for an item is
already checkpointed, the image load and per-option prompt-building are
skipped entirely, not just the judge calls.

### Practical workflow if a run fails partway through

1. Just re-run the same scoring cell (or the whole notebook, top to bottom —
   re-running `JudgeCheckpoint(...)` / `run_*_scoring(...)` is always safe).
   It will print how many items were skipped as already-done vs. newly scored
   vs. failed this pass, and will only spend GPU time on what's missing.
2. If something failed and keeps failing on every rerun (as opposed to a
   one-off transient error), the `[checkpoint] FAILED item_id=... : ...`
   message printed at failure time identifies exactly which item and what
   exception — that's the place to start debugging, not the whole run.
3. The raw `checkpoints/*.jsonl` files are never overwritten or deleted by any
   notebook here. They retain the full raw model response text for every
   scored item (useful for auditing parse failures or spot-checking scores
   against what the model actually said), and remain the source of truth even
   after the CSV/analysis outputs are built — safe to keep around indefinitely,
   and safe to re-derive any downstream CSV from at any time.

## Files

### `prompts/benchmarks.py`
Rubric dictionaries (`BENCHMARK_1` … `BENCHMARK_5B`) transcribed verbatim from
the paper, plus one `build_benchmark_N_prompt(...)` function per benchmark that
assembles the rubric + question/answer (or option, for 5a/5b) into the text sent
to the judge alongside the image. Also defines `JUDGE_SYSTEM_PROMPT`, which
instructs the model to return strict JSON (`{"criterion name": score}`) and
nothing else.

### `judge_models.py`
`JudgeModel` — a thin wrapper around `AutoModelForImageTextToText` +
`AutoProcessor` (matching the loading pattern already validated in
`intern_vl_testing.ipynb` / `qwen_vl_testing.ipynb`, including 4-bit NF4
quantization). Exposes `.score(image, prompt_text, criteria) -> (dict, raw_text)`,
which generates once, parses the JSON response, retries once on failure, and
falls back to `-1` for any criterion it still can't parse (mirroring the human
"unable to evaluate" convention).

### `checkpoint.py`
`JudgeCheckpoint` — an append-only JSONL store that every judge-calling notebook
uses to persist raw results incrementally. See "Checkpointing" below; this is the
piece that makes long unattended runs safe to interrupt and resume.

### `judge_runner_pathopen.ipynb`
Runs both judges over the **entire** PathOPEN core subset
(`pathopen_vqa_part{1-5}.csv`, ~229 resolvable rows) and the image-augmentation
subset (`pathopen_vqa_augmented_part{1-5}.csv`, 174 rows) — not a single
evaluator's slice, all parts pooled — applying Benchmarks 1/2/3 to the core
subset and Benchmark 4 to the augmentation subset. Images are resolved locally
via `pathopen_data/processed/data.json` (not the Google Drive links used to hand
data to human evaluators). Output is written in the **same column schema** as
`data_evaluation/pathologists/scoring_analysis/input/evaluatorN/*.csv`, to
`judge_output/evaluator_internvl/` and `judge_output/evaluator_qwenvl/`, so it can
be copied alongside `evaluator{1-5}/` and run through the existing
`pathopen_scoring_analysis_individual.ipynb` /
`pathopen_image_augmentation_scoring_analysis_individual.ipynb` unchanged.

**Known gap**: cases 9 and 79 have no resolvable image in
`processed/data.json` (an empty `Original` field — a pre-existing data gap, not
introduced by this pipeline). These 2 of 231 rows are skipped and reported, not
silently dropped.

### `judge_runner_pathvqa.ipynb`
Same pattern for the filtered-PathVQA subset (`pathvqa_part{1-5}.csv`, 662 rows,
one question per row tagged by `question_type`). No wrong answers exist in
PathVQA, so only Benchmarks 1 (open-ended types) and 3 (`close-ended`) apply.
Images are already local under
`subsets_processing_output/images/pathvqa/`, one file per `image_id`.

**Important**: unlike PathOPEN, PathVQA's `image_id` is **not unique** — the
same image can host several different questions (224 duplicate `Image_ID`
values were found across the pooled human ratings). A join on `Image_ID` alone
silently inflates row counts through a many-to-many merge. This runner assigns
an explicit `row_uid` (global position across `pathvqa_part1..part5`
concatenated in sorted order) as the reliable join key, matching how the human
evaluator CSVs were generated in the same row order.

### `judge_pathologist_agreement.ipynb`
Phase 1 validation. Pools all 5 human evaluators' ratings per dataset (they are
disjoint row slices — verified zero overlap — so this is "judge vs. whichever
single pathologist rated this row," not classic multi-rater IRR) and computes,
per (dataset × benchmark × criterion):

- **Weighted Cohen's kappa** (quadratic weights, ordinal-appropriate) between
  judge and human score, excluding human `-1` ratings.
- **Mann-Whitney U + rank-biserial effect size** comparing PathOPEN vs.
  filtered-PathVQA score distributions — mirrors the paper's Sub-pillar 1a
  method exactly, computed separately for human ratings and each judge, so you
  can see whether a judge reproduces the same *qualitative* conclusion
  (PathOPEN scoring higher) independent of raw row-level agreement.

Outputs `agreement_output/judge_pathologist_weighted_kappa.csv` and
`agreement_output/pathopen_vs_pathvqa_mannwhitney.csv`.

### `wrong_answer_tier_agreement.ipynb`
Pillar 2, PathOPEN only (PathVQA has no wrong answers to tier). Bins existing
pathologist Benchmark 2 scores into tiers (2→near-miss, 1→moderate-miss,
0→far-miss, -1 excluded), separately for OE-native and MCQ-derived wrong
answers, then has the judge re-score the same items and computes tier-agreement
(weighted kappa + 3×3 confusion matrix). Flags any tier with n < 30, per the
paper's own stated reviewer concern. Also documents (but does not implement) the
paper's *optional* Figure 4 Panel C — a monotonic-difficulty-gradient behavioral
test — since that requires VLM-as-answerer benchmarking, out of scope here.

### `benchmark5_mcq_standalone.ipynb`
Pillar 3 / Benchmark 5. Scores every resolvable MCQ option **standalone** (no
question stem) across three datasets:

- **PathOPEN** — its own MCQ (correct + 4 wrong options), ~229 cases.
- **PathMMU** (`jamessyx/PathMMU`, already downloaded to
  `/data/mn27889/pathology-datasets/PathMMU/`) — all splits (`val`, `test`,
  `test_tiny`) across 5 source categories.
- **PatchVQA** (part of PathBench,
  `/data/mn27889/pathology-datasets/PathBench/data/PatchVQA.json`) — same
  structure, 4 categories. Its images are sourced from PathMMU's own image pool
  (documented in PathBench's README), not a separate directory.

Since Benchmark 5 has no human baseline, this notebook reports **judge-vs-judge
weighted kappa** (InternVL vs. Qwen-VL, on the same items) as the only
reliability signal — explicitly flagged as weaker evidence than the Benchmark
1–4 kappas (two models could share the same blind spots).

**Known gaps** (pre-existing in the downloaded data, not introduced here): the
`SocialPath` category is **100% unresolvable** for both PathMMU and PatchVQA —
its images were never downloaded (the dataset's own README documents a separate
manual acquisition step for several `PathCLS` sub-datasets and social-media
content that wasn't completed). `PathCLS` is ~8.4% unresolvable. Unresolvable
items are skipped and counted per run, not silently dropped.

This is a **large inference job** by design (per explicit user choice): every
resolvable item across all three datasets, not a sample.

### `quiltvqa_benchmark1.ipynb`
Pillar 1b. Scores Quilt-VQA's (`wisdomik/Quilt_VQA`, cached locally) 724
`OPEN`-type rows (out of 985 total — the 261 `CLOSED` rows are out of the
paper's stated Sub-pillar 1b scope and are left unscored here) against
Benchmark 1. Then reproduces the paper's Figure 3 Panel B (PathOPEN vs.
Quilt-VQA, judge-rated Benchmark 1 scores, Mann-Whitney U/rank-biserial) using
each judge's own PathOPEN scores from `judge_runner_pathopen.ipynb`'s output.
This notebook does **not** re-derive judge credibility on its own — it assumes
`judge_pathologist_agreement.ipynb` has already established the Benchmark 1
weighted kappa, and its final markdown cell reminds you to report both together
(Figure 3 Panel A + Panel B), per the paper's own structure.

## Run order

1. `judge_runner_pathopen.ipynb`
2. `judge_runner_pathvqa.ipynb`
3. `judge_pathologist_agreement.ipynb` (needs 1 & 2's output)
4. `wrong_answer_tier_agreement.ipynb` (needs 1's output)
5. `benchmark5_mcq_standalone.ipynb` (independent of 1–4, but large — expect a long run)
6. `quiltvqa_benchmark1.ipynb` (needs 1's output for the Mann-Whitney comparison; conceptually depends on 3 having already been reviewed)

`MODELS_TO_RUN` at the top of each runner notebook controls which judge(s) to
use for that pass (`["internvl", "qwenvl"]` by default — both fit comfortably
given the 8×A6000 GPUs available; there's no need to run them sequentially to
save memory unless you want to).

Each notebook writes its own `checkpoints/` and output directory relative to
its **working directory at kernel start** (i.e. `data_evaluation/vlm/`, assuming
you launch Jupyter from there as usual) — they don't share a checkpoint
namespace with each other, so re-running one notebook never touches another's
saved progress.

## Environment

All notebooks assume the `path-opendata-vlms` conda environment (has
`transformers`, `torch`, `qwen_vl_utils`, plus `scipy`/`scikit-learn`, which
were added to this env specifically for the agreement/kappa computations).

## Validation notes (for future reference)

Every notebook here was dry-run end-to-end against the real repository data
(with a stub `JudgeModel` returning random scores in place of actual GPU
inference) before being considered done. This caught several bugs that would
not have been obvious from reading the code alone:

- An image-extension bug (`.jpeg` files silently excluded from the resolver)
  that hid 113 of 115 apparently-missing PathOPEN images.
- A directory-path-reconstruction bug in the same resolver.
- The PathVQA non-unique-`Image_ID` issue described above (would have silently
  corrupted every PathVQA agreement statistic via a many-to-many join).
- A human/judge column-name mismatch in the Mann-Whitney comparison that
  silently produced `NaN` for judges specifically (the loop matched columns by
  a shared name that only existed in the human CSV).
- `REPO_ROOT` was resolved one directory level too high in all three
  originally-drafted notebooks (`"..","..",".."` instead of `"..",".."`
  from `data_evaluation/vlm/`).
- An f-string syntax error (escaped quote inside a `{...}` expression).

If you extend or modify any notebook here, re-running it against a stub judge
(swap `JudgeModel` for one that returns fixed/random scores) before a real GPU
run is a fast way to catch the same class of join/path/schema bugs.

**Checkpointing was validated the same way**, using a stub `JudgeModel` that
deterministically raises on a specific call number to simulate a mid-run crash,
then re-running the same notebook to confirm it (a) skips every item already
checkpointed, (b) only retries what actually failed or never ran, and (c)
produces the same final CSV shape/schema as an uninterrupted run. This caught
one real bug: the first checkpointed version of `judge_runner_pathopen.ipynb`
only persisted judge *scores* per sub-task, not the question/answer text those
scores belong to — so the assembled CSV silently dropped 14 text columns
compared to the target schema (22 columns instead of 36). Fixed by having the
assembly step join checkpointed scores back against the source subset
DataFrame (already in memory, no need to checkpoint text that's cheaply
available locally) for the text columns.
