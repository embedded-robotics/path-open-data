# PathOPEN

A pathologist-curated visual question answering dataset for histopathology, and the
full pipeline that builds, augments, evaluates and validates it.

157 cases curated by two clinical pathologists, each with 1–3 H&E images at up to five
magnifications, open-ended questions with **both correct and wrong answers**, one
five-option MCQ, and one close-ended question. Five further pathologists rated the
result against four scoring rubrics. The dataset is accompanied by a 30× augmented
variant and a class-balanced retention variant.

The accompanying manuscript is being prepared for *Scientific Data* as a Data
Descriptor. **The manuscript source is not in this repository** - it lives with the
corresponding author, restructured into the journal's mandated section order
(Background & Summary -> Methods -> Data Records -> Technical Validation -> Usage Notes,
with no Results or Discussion section). §7 records what changed between the original draft
and that version, since several of the fixes were to claims this code generates.

---

## 1. Orientation — read this first

The repository is organised as **pipeline stages**, each a top-level directory, in the
order data flows through them:

```
data_processing/     raw collection -> structured JSON; PathVQA H&E filtering
data_augmentation/   30x transform pipeline; class-balanced retention variant
data_analysis/       descriptive statistics of the raw datasets
data_evaluation/     the evidence: pathologist ratings, VLM judges, VLM answerers
data_graphics/       publication figures and LaTeX tables
pathopen_data/       the dataset itself (gitignored - see §6)
```

Two rules that explain most of the layout:

**Generated output is never committed.** Every directory that holds results —
`output/`, `analysis/`, `checkpoints/`, `figures/`, `tables/` — is gitignored. The
*code* is tracked; the outputs are reproduced by running it. See `.gitignore` for the
complete list. This means a fresh clone has no results in it, and you must re-run the
pipeline to get them.

**Several subdirectories carry their own README**, and those are the authoritative
detail for their stage. This file is the map; they are the territory:

| README | Covers |
|---|---|
| `data_evaluation/vlm_judge/README.md` | Judge pipeline, the five rubrics, checkpoint invariant |
| `data_evaluation/vlm_zeroshot/README.md` | Answerer pipeline, the six-model set, metrics |
| `data_evaluation/vlm_zeroshot/ENVIRONMENT.md` | **Why two conda environments are required** |
| `data_graphics/README.md` | Figure conventions, Nature sizing, table generation, **the two sn-jnl float traps** |

---

## 2. Environment

```bash
conda activate path-opendata-vlms
```

**Use this environment, not system Python** — the base interpreter lacks matplotlib and
the scientific stack. Nearly every script and notebook here assumes it.

A **second environment, `path-llava`**, exists solely for LLaVA-Med and Quilt-LLaVA.
Those two are original-LLaVA-repo checkpoints whose dependencies would downgrade the
stack the other four models and both judges need. `ENVIRONMENT.md` explains the
conflict and the `LD_LIBRARY_PATH` workaround; don't try to merge the two.

GPU work (judges, answerers) expects multiple CUDA devices; `gpu_allocation.py` in both
evaluation folders handles placement.

---

## 3. The pipeline, stage by stage

### 3.1 `data_processing/` — raw data to structured records

| File | Role |
|---|---|
| `raw_pathopen_data_processing.ipynb` | Google Form responses + images → `pathopen_data/processed/data.json` |
| `raw_pathvqa_data_filtering_histocartography.ipynb` | Filters PathVQA to H&E-stained images |
| `histocartography/` | Vendored dependency (gitignored) |

**The PathVQA filter is a three-step funnel**, and the numbers matter because the paper
quotes them:

1. **H&E detection** — HistoCartography nuclei detection; an image is kept if it has
   **more than 10 nuclei** (`nuclei_centers.shape[0] > 10`). → 1,241 images, 8,992 pairs
   from the original 4,998 images / 32,799 pairs.
2. **Question-type split** — "yes/no" answers are close-ended, the rest open-ended.
   → 4,630 close-ended, 4,362 open-ended.
3. **Word-count floor** — both question and answer must have ≥ 5 words (close-ended:
   question only). → **612 open-ended, 3,005 close-ended**, about 10% of the original.

> ⚠️ A stale comment above the nuclei line mentions 5 neighbours (a kNN parameter). The
> threshold is 10. The manuscript's §2.8 currently says 5 and is wrong; §3.5.2 says 10
> and is right.

### 3.2 `data_augmentation/` — scaling and rebalancing

| File | Role |
|---|---|
| `raw_pathopen_data_augmentation.ipynb` | Applies the 30× transform pipeline |
| `targeted_augmentation_balance.ipynb` | Class-balanced retention variant, entropy/Gini |
| `pathology-he-autoaugmetation/` | Vendored augmentation library (gitignored) |

**The 30× pipeline.** Each original image gets 30 augmentations. One augmentation =
5 transforms drawn at random from 16, each at a random magnitude 0–5, applied in random
order on top of one another. Elastic deformation, Gaussian blur and Gaussian noise are
**deliberately excluded** — pathologists judged they destroy identifiability at higher
magnitudes.

Setup for the vendored library:

```bash
git clone git@github.com:DIAGNijmegen/pathology-he-autoaugmetation.git
pip install scikit-image albumentations==1.1.0
```

Then edit `randaugment/train/randaugement_new_ranges.py`:
- `_REPLACE = 225` — makes background semi-white to match slide background
- `available_ops = ['Scaling','TranslateX','TranslateY','ShearX','ShearY','Brightness',
  'Sharpness','Color','Contrast','Rotate','Equalize','HsvH','HsvS','HsvV','HedH','HedE','HedD']`

**The balanced variant is a *selection*, not a second generation run.** This is the
single most misunderstood part of the repo. All 30 augmentations per image stay on disk.
`targeted_augmentation_balance.ipynb` computes, per image, how many of its existing 30 to
*retain* so a chosen metadata field's class distribution flattens:

```
keep = clip(n_max / n_class, 1, MAX_MULTIPLIER=10), further capped by what exists
```

A rare-class image keeps up to 10; a common-class image keeps 1. The output is a
**manifest** in `pathopen_data/augmented/balanced_manifests/` naming which files a
training run should consume — not a folder of copies.

Consequences worth knowing:
- Each of the four fields is balanced **separately**, giving four different variants of
  852 (Organ) / 988 (Categorization) / 994 (Regional Anatomy) / 427 (Magnification)
  images. Regional anatomy nests inside organ, so balancing one perturbs the others.
  There is no single variant that flattens all four.
- **Magnification is not one of the paper's three stated fields** (§3.5 names organ,
  categorization, regional anatomy). It is balanced anyway because it is imbalanced on
  the same order (10.4× ratio) and, unlike the other three, is a property of the *image*
  rather than the *case* — so balancing it leaves case-level representation untouched and
  composes with the others. Note the notebook's own justification for including it cites
  "augmented 400× images rated lowest by every rater", which is **false for Qwen**, whose
  minimum is 20×. The inclusion is defensible on imbalance grounds; that sentence is not.
- `MAX_MULTIPLIER = 10` is an **overfitting guard**, chosen for that reason. The
  multiplier sweep confirms it captures most of the available correction; it did not
  select it.
- Uniform 30× augmentation leaves Gini and entropy **exactly** unchanged, because
  scaling every class by a constant cannot change class proportions. This is arithmetic,
  not measurement, and it is why differential retention is needed at all.

Outputs → `augmentation_balance_output/`: `pillar5b_balance_before_after.csv`,
`pillar5b_class_counts.csv`, `pillar5b_multiplier_sweep.csv`,
`figure10b_magnification_quality.csv`, `figure10b_magnification_trend.csv`.

### 3.3 `data_analysis/` — descriptive statistics

`raw_pathopen_data_analysis.ipynb` and `raw_pathvqa_data_analysis.ipynb` produce the 29
PNGs in `output/`: word-count distributions and thresholds, organ / categorization /
regional-anatomy / magnification breakdowns.

> ⚠️ **These are no longer the manuscript's figures.** The three `dataset_*.ipynb`
> notebooks in `data_graphics/` supersede them: same material, consolidated into three
> publication figures, recomputed against the **229 as-released images** rather than the
> 231 image slots, plus analyses these PNGs never had (Lorenz/Gini concentration,
> magnification co-occurrence, organ x category coverage, question-opening words).
>
> This stage is kept as the exploratory record — the funnel numbers are easier to
> re-derive here, and the word-count threshold logic (`>=` not `>`, which moves every
> figure 2-5 points) originates here. **To change a dataset-statistics number in the
> paper, edit `data_graphics/dataset_*.ipynb`.**

### 3.4 `data_evaluation/` — the evidence

Three independent evaluation tracks. **Understanding the judge/answerer distinction is
essential**:

| | `vlm_judge/` | `vlm_zeroshot/` |
|---|---|---|
| model's job | grade existing text against a rubric | produce an answer |
| output | ordinal score {−1,0,1,2} | predicted option / free text |
| metric | weighted κ, Gwet's AC1 vs pathologists | accuracy, token F1, BLEU, chrF, BERTScore |
| models | 2 large judges (32B / 38B) | 6 candidates (4–8B) |

#### `pathologists/` — the human ground truth

`subsets_processing/` builds the rating packets handed to the five pathologists;
`scoring_analysis/` aggregates what came back. Each subset has an `_individual` and a
`_final` notebook (per-rater vs pooled).

**Each item was rated by exactly one pathologist.** There is zero overlap, so no
per-item inter-rater statistic (Fleiss' κ, Krippendorff's α) is computable on this
design. What *can* be tested is whether raters' score **distributions** differ —
χ² and Kruskal–Wallis — which is what the analysis does.

#### `vlm_judge/` — VLM replaces pathologist

InternVL3.5-38B and Qwen3-VL-32B-Thinking score text against the same rubrics the
pathologists used. Rubric text is verbatim from the paper in `prompts/benchmarks.py`.

```
judge_runner_pathopen.ipynb     Benchmarks 1-4 on PathOPEN
judge_runner_pathvqa.ipynb      Benchmarks 1, 3 on filtered PathVQA
quiltvqa_eval_runner.ipynb      Benchmark 1 on Quilt-VQA
mcq_option_validity_runner.py   Benchmark 5a/5b across PathOPEN, PathMMU, PatchVQA
judge_pathologist_agreement.ipynb   kappa / AC1 vs the humans
wrong_answer_tier_agreement.ipynb   difficulty-tier agreement
```

> ⚠️ **Checkpoints are the source of truth.** `checkpoint.py` writes JSONL as each item
> completes. CSVs in `agreement_output/` are *derived* and regenerable. Never hand-edit
> a CSV.
>
> ⚠️ **`wc -l` lies on these JSONL files** — `raw_response` contains embedded newlines.
> Always parse each line as JSON to count records. This error has caused real
> misreporting (a complete 6,951-record run was reported as 4,750).

#### `vlm_zeroshot/` — VLM as answerer

Six models spanning a specialization gradient: Qwen2.5-VL-7B and InternVL3.5-8B
(general), MedGemma-4B (broad medical, **gated**), LLaVA-Med and Quilt-LLaVA
(biomedical/pathology, **need `path-llava`**), Patho-R1 (pathology SOTA, built on
Qwen2.5-VL so the pairing isolates what pathology tuning buys).

```
launch_all.py              orchestrates every model x dataset run
run_answerer.py            MCQ / close-ended
run_answerer_oe.py         open-ended generation
score_mcq.py               accuracy, above-chance, position shuffle, blind condition
score_open_ended.py        token F1, BLEU, chrF, BERTScore (raw + baseline-rescaled)
score_open_ended_strata.py open- AND close-ended by magnification / organ system
score_dataset_comparison.py  bootstrap CIs, Holm-corrected Mann-Whitney, Spearman
tier_difficulty.py         does accuracy fall across difficulty tiers?
tier_open_ended.py         same question, open-ended metrics
smoke_tests/               one notebook per model - run before a full launch
```

Outputs → `analysis/` (20 CSVs). Run a smoke test first; a full launch is many GPU-hours.

### 3.5 `data_graphics/` — manuscript figures and tables

Eight notebooks: three describing the dataset, five reporting the validation pillars, all
importing `nature_style.py`.

```
nature_style.py      shared rcParams, palette, mm(), save(), panel_label_below()

dataset_composition.ipynb          Fig 2  183x231 mm   8 panels   <- metadata composition
dataset_text_characteristics.ipynb Fig 3  183x211 mm   7 panels   <- question/answer text
dataset_pathvqa_funnel.ipynb       Fig 4  155x 72 mm   1 panel    <- PathVQA filtering

pillar1_dataset_quality.ipynb      Fig 5  171x222 mm   7 panels
pillar2_wrong_answers.ipynb        Fig 6  182x153 mm   3 panels
pillar3_dual_mcq.ipynb             Fig 7  172x224 mm   6 panels
pillar4_benchmarking.ipynb         Fig 8  165x236 mm  11 panels
pillar5_augmentation.ipynb         Fig 9  177x213 mm   8 panels

make_tables.py       generates 8 main-text tables + 1 supplementary into tables/
check_tables.sh      compiles every table to catch overfull boxes
```

Fig. 1 is the workflow diagram at the repo root. **The three `dataset_*` notebooks are new
and supersede `data_analysis/output/`** — see §3.3.

Hard constraint: **183 × 247 mm**, 5 pt minimum type. See `data_graphics/README.md` for
the colour system, the plotting conventions, and the **two sn-jnl float traps** that affect
every wide table (one of them silent — it drops the table, caption and label while
compiling clean).

```bash
conda activate path-opendata-vlms
cd data_graphics
python make_tables.py
./check_tables.sh          # needs `tectonic` on PATH
```

> ⚠️ `check_tables.sh` exists because structural checks cannot see page overflow. Table 1
> once ran **243 pt (8.6 cm) off the page** — two columns entirely invisible — while
> passing every column-count and environment check. It is now wrapped in
> `\resizebox{\linewidth}{!}{...}`, which requires `\usepackage{graphicx}` alongside
> `booktabs` and `siunitx`.
>
> There is no system LaTeX and a full TeX Live will not fit (`/data` is at 100%). Use
> `tectonic` — a single static binary that fetches only what a document needs. Keep it
> and its cache outside the repo.

---

## 4. The five validation pillars

Most of `data_evaluation/`, and the five `pillar*` notebooks in `data_graphics/`, exist to
serve these. (The three `dataset_*` notebooks serve Data Records instead — they describe
what the dataset *is*, not how well it rates.)

| Pillar | Question | Artifacts (manuscript numbering) |
|---|---|---|
| **1 — Dataset quality** | Is PathOPEN better rated than filtered PathVQA? | Fig 5, Tables 2 / 3 / 4 |
| **2 — Wrong-answer structure** | Do wrong answers form graded difficulty tiers? | Fig 6, Table 5 |
| **3 — MCQ option validity** | Do MCQ options work as standalone statements? | Fig 7 |
| **4 — Benchmarking utility** | How does PathOPEN compare as a benchmark? | Fig 8, Tables 6 / 7 |
| **5 — Augmentation strategy** | Does augmentation preserve content and fix imbalance? | Fig 9, Table 8 |

The dataset itself is described by Figs. 2-4 and Table 1, which are *not* pillar outputs —
they belong to Data Records. Supplementary Tables S1-S6 are the six benchmark rubrics
(hand-written, in the manuscript package); S7 is the generated axis-label key.

All ratings use the ordinal scale **{−1, 0, 1, 2}**, where **−1 always means "unable to
evaluate"** — not a low score. Analyses must handle it explicitly rather than treating
it as ordinal.

### Findings a successor should not have to rediscover

These are established and were expensive to get right:

- **Pillar 1** — PathOPEN's headline is the *defect rate*: one in three filtered-PathVQA
  close-ended items was unusable; PathOPEN had none. 8 of 9 comparisons exclude zero
  (not all nine — InternVL's B3 CI crosses). Report Gwet's AC1 alongside κ: at 95%
  exact agreement κ collapses to −0.022 while AC1 is 0.984.
- **Pillar 2** — Judges **cannot** reproduce pathologist difficulty tiers: 24 κ cells,
  all below 0.26 (max 0.2533, one negative; means 0.114 InternVL / 0.142 Qwen). Table 5
  diagnoses why (near-miss inflation 2.4–3.0×, and 2.65×/1.54× worse on free-text than
  option-shaped distractors). This is a reportable negative result, not a failure to hide.
  Note the tiers are *similar* in near-miss rate across authoring routes (17.2% vs 14.0%);
  the routes differ in the far-miss tail (39.1% vs 60.6%).
- **Pillar 3** — Benchmark 5a survives both judges; **5b inverts between them** —
  InternVL ranks PathOPEN first on distractor criteria, Qwen ranks it last. Report both
  judges for every 5b claim. **Do not claim position-bias superiority**: PathOPEN
  averages 3.64 pp shuffle spread vs PathMMU's 1.42, second of three.
- **Pillar 4** — PathOPEN is **mid-range in difficulty, not hardest** (22.36
  [19.88, 24.98] vs PathMMU 25.75 [24.42, 27.04]) and has the **narrowest** model spread
  of three. Magnification is a null result (2 of 18 tests, roughly chance). Cross-format
  rank instability is the pillar's real finding, stated as the *absence* of a stable
  relationship.
- **Pillar 5** — Uniform 30× ≡ raw, bit-identical, by arithmetic. Five pathologists
  differ significantly on Benchmark 4 (*P* < 0.001), driven by one rater who marked
  12.2% / 9.5% of augmented images unscorable where four marked none.

---

## 5. Reproducing from scratch

```bash
conda activate path-opendata-vlms

# 1. Process raw data
jupyter nbconvert --execute --inplace data_processing/raw_pathopen_data_processing.ipynb
jupyter nbconvert --execute --inplace data_processing/raw_pathvqa_data_filtering_histocartography.ipynb

# 2. Augment (slow - writes 30 images per original)
jupyter nbconvert --execute --inplace data_augmentation/raw_pathopen_data_augmentation.ipynb
jupyter nbconvert --execute --inplace data_augmentation/targeted_augmentation_balance.ipynb

# 3. Descriptive statistics
jupyter nbconvert --execute --inplace data_analysis/raw_pathopen_data_analysis.ipynb
jupyter nbconvert --execute --inplace data_analysis/raw_pathvqa_data_analysis.ipynb

# 4. Evaluation - GPU, many hours. Smoke test first.
#    See the two evaluation READMEs; do not run blind.
python data_evaluation/vlm_zeroshot/launch_all.py

# 5. Figures and tables
cd data_graphics && python make_tables.py && ./check_tables.sh
#    then run the three dataset_* notebooks and the five pillar notebooks
```

Stages 1–3 are CPU and take minutes to hours. Stage 4 is the expensive one.

---

## 6. Data availability — open issue

`pathopen_data/` is **gitignored**, so a clone contains no images or JSON. The dataset
has not yet been deposited publicly: the manuscript's Data Availability section still
carries `\TODO` placeholders for repository, DOI and licence, while this code repository
is public.

*Scientific Data* **requires deposition in a public repository before submission.** This
is a blocker for the manuscript, not a formatting detail. Whoever takes this over needs
to resolve it (Zenodo, Figshare or Hugging Face) and fill in Data Availability.

Layout of the directory when populated:

```
pathopen_data/
  raw/images/                 as collected
  processed/images/           + data.json  <- the canonical record
  augmented/images/           30x per original
  augmented/balanced_manifests/   retention manifests (see 3.2)
  gdrive_mappings/            provenance back to collection forms
```

---

## 7. Manuscript corrections traceable to this code

All of the issues previously listed here have been resolved in the restructured
manuscript. They are kept, with their resolutions, because **each one was settled by
reading the code or the generated CSVs** — so if a number here changes, the manuscript
claim changes with it.

| Issue in the original draft | Resolution |
|---|---|
| "Harder / more discriminating benchmark" | **Withdrawn.** PathOPEN is mid-range on chance-corrected accuracy (22.36 [19.88, 24.98] vs PathMMU 25.75) with the narrowest model spread of three. |
| Promised Fleiss' κ / Krippendorff's α | **Not computable** — one rater per item, zero overlap. Replaced by distributional homogeneity (χ², Kruskal-Wallis). |
| Nuclei threshold "more than 5" | **10.** `nuclei_centers.shape[0] > 10`; the stale comment mentioning 5 is a kNN parameter. |
| Filtered PathVQA shown as 1,504 images | **1,241**, counted from the pickles. QA counts were right. |
| Close-ended count 3,005 vs 3,009 | **3,005.** |
| Dangling reference to a Benchmark 5 validation section | **No such validation exists.** Pathologist ratings cover Benchmarks 1-4 only; B5 is judge-vs-judge, so its agreement is a reliability signal, not validation. |
| 4-6-word MCQ option mandate vs Benchmark 5a scoring a fragment as 0 | Disclosed as measured: options average **8.58 words** over distinct strings (median 9, only 10.6% in the 4-6 band), or **9.67** over the 1,145 options B5 actually rated. Framed as the pathologists reading the mandate as a floor. |
| **Image count** | The release has **229 images, not 231** — cases 9 and 79 were never retrieved. Every image-dependent count changed; `table_data_records.tex` prints both columns. |
| PathVQA pair count 32,799 | **32,795** is what the released pickles contain; both are disclosed. |
| Regional-anatomy classes "112 → 111" on case folding | **113 → 111**, on two collisions (`Mucosa`/`mucosa`, `Tongue`/`tongue`). |
| PathMMU MCQ option length 4.1 words | **4.81** (PatchVQA 4.98), measured from the B5 checkpoints. The "roughly twice as long" comparison survives. |
| "Near-miss tier more than twice as frequent in the open-ended-native pool" | **False** — 17.2% vs 14.0%, a ratio of 1.23, on either B2 criterion. The real asymmetry is the far-miss tail (39.1% vs 60.6%). |
| Structure | Remapped to *Scientific Data*'s mandated order; no Results or Discussion section. |

### Counting units that have two correct answers

Four quantities differ depending on the unit, and all four appear in the manuscript. **None
is an error; do not "fix" one to match the other.**

| Quantity | Per distinct item | Per rated unit |
|---|---|---|
| MCQ option words | 8.58 (775 distinct strings) | 9.67 (1,145 rated options) |
| Open-ended question words | 17.6 (309 questions) | 18.67 (457 question-image pairs) |
| Metadata Gini | per case: 0.44 / 0.66 / 0.24 / 0.37 | per image: 0.45 / 0.71 / 0.32 / 0.37 |
| Benchmark 2 tier counts | 349 OE-native + 920 MCQ-derived wrong answers | 686 + 1,831 **ratings** (both B2 criteria scored per answer) |

The B2 one is the easiest to misread: `wrong_answer_tier_agreement.csv` reports 668 and
1,773, which are those ratings *further restricted* to items where the judge also returned
a tier. Percentages agree within 0.7 pp either way.

---

## 8. What changed during the manuscript restructure

For anyone returning to this repo after the move to *Scientific Data* format. Dates are
approximate; the git log is authoritative.

**Three Data Records notebooks** — `dataset_composition.ipynb`,
`dataset_text_characteristics.ipynb` and `dataset_pathvqa_funnel.ipynb` — produce the
manuscript's Figs. 2-4. They **replace `data_analysis/output/`'s 29 exploratory PNGs** as
the published dataset statistics, and carry four analyses the PNGs never had: a
**Lorenz/Gini concentration** panel over all four metadata fields, a **magnification
co-occurrence matrix**, an **organ x category coverage grid**, and a
**question-opening-word** breakdown. The pillar figures' panel content is unchanged; only
their captions and the text around them were revised.

**`make_tables.py` grew from 8 outputs to 9.** Added `table_data_records` (item counts,
as-collected vs as-released). The `S` prefix was **dropped from every filename** — the
manuscript's S1-S6 are now the six benchmark rubrics, which live in the manuscript package,
not here, so the generated tables are main-sequence (the one exception,
`table_label_abbreviations`, became S7). Five of the tables use the
savebox + `tableorg` pattern that the manuscript class requires; see
`data_graphics/README.md` for both traps, one of which fails silently.

**`check_tables.sh`** now stubs `\pathopentablebox`, `tableorg` and `\TODO` so the
converted tables still compile standalone. It cannot reproduce the silent dropped-float
trap — that needs the real class.

**One data fix.** The `Gastrointestinal - Small Intenstine` typo was corrected in the
collected data.

**Numbers that changed as a result**, all re-derived from the CSVs and now matching the
manuscript: the release has **229 images** (not 231), regional anatomy has **113** released
classes, PathMMU's option length is **4.81** words, and the Benchmark 2 near-miss ratio
between authoring routes is **1.23×**, not the ">2×" previously claimed. §7 has the full
list with resolutions.

---

## 9. Repo conventions

- **Branch**: work happens on `vlm_process`; `main` is the integration branch.
- **Notebooks are the interface** for exploratory stages; `.py` modules for anything
  run at scale or imported.
- **Paths are built with `os.path.join(...)`**, so a literal grep for a path fragment
  will miss things. After any rename, re-run the notebooks rather than trusting a text
  search.
- **Vendored dependencies** (`histocartography/`, `pathology-he-autoaugmetation/`) are
  gitignored and must be cloned separately.

### Root-level files

| File | Status |
|---|---|
| `main_workflow.png`, `PathOPEN_workflow.svg` | Workflow diagram (manuscript **Fig. 1**) |
| `datasets_models_download.ipynb` | Fetches comparator datasets and model checkpoints |
| `intern_vl_testing.ipynb`, `qwen_vl_testing.ipynb` | Early scratch notebooks, superseded by `smoke_tests/` |
| `vlm_codes_archive/`, `vlm_data_archive/` | Archived earlier work, not part of the pipeline |
| `test_data/` | Scratch, gitignored |
| `box_config.json`, `.env` | Credentials, gitignored |

![Main Workflow of Curating PathOPEN Dataset](main_workflow.png)
