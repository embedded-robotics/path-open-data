# Figures and tables

**Eight** figures in Nature submission format: five reporting the validation pillars, and
three describing the dataset itself. (The original plan called for ten pillar figures
numbered 2-11; they were consolidated into five multi-panel figures, which is why the panel
counts are high.)

One notebook per figure, all sharing `nature_style.py` so the set reads as one design rather
than eight independent plots.

```
data_graphics/
  nature_style.py            shared rcParams, palette, mm(), save() - import this first

  # Data Records figures - what the dataset contains
  dataset_composition.ipynb          Fig 2  183x231 mm   8 panels (a-h)
  dataset_text_characteristics.ipynb Fig 3  183x211 mm   7 panels (a-g)
  dataset_pathvqa_funnel.ipynb       Fig 4  155x 72 mm   single panel

  # Technical Validation figures - one per validation pillar
  pillar1_dataset_quality.ipynb      Fig 5  171x222 mm   7 panels (a-g)
  pillar2_wrong_answers.ipynb        Fig 6  182x153 mm   3 panels (a-c)
  pillar3_dual_mcq.ipynb             Fig 7  172x224 mm   6 panels (a-f)
  pillar4_benchmarking.ipynb         Fig 8  165x236 mm  11 panels (a-k)
  pillar5_augmentation.ipynb         Fig 9  177x213 mm   8 panels (a-h)

  make_tables.py             generates 8 main + 1 supplementary table into tables/
  check_tables.sh            compiles every table; catches overfull boxes
  figures/                   figD1_*/fig01_* .svg + .pdf (flat, not per-figure)
  tables/                    table1_*.tex, table_*.tex
```

Figure *numbers* are the manuscript's; file *names* keep the `figD1-3` / `fig01-05`
prefixes from when the two sets were numbered separately. Fig. 1 is the workflow diagram
at the repo root, not generated here.

> **`pathvqa_raw_word_counts.csv` is committed on purpose** — the only tracked data file
> here. `dataset_text_characteristics.ipynb` needs word-count histograms of the *full,
> unfiltered* PathVQA corpus (32,795 pairs), which lives **outside this repository** at
> `/data/mn27889/pathology-datasets/pvqa/`. Caching the 112-row histogram keeps Fig. 3
> reproducible from a clone. Everything else in `figures/` and `tables/` is gitignored.

**Panel counting gotcha.** Several panel letters are not string literals in the source:
Fig. 8's panels f-k are drawn in a loop over `("f","g")`, `("h","i")`, `("j","k")`, and
Fig. 6's panel b is a *pair* of confusion matrices labelled once via `fig.text`. A grep for
`panel_label(ax, "x")` undercounts both. There are **50 panels across the seven multi-panel
figures**, and every one is cited by letter in the manuscript body text.

Both `figures/` and `tables/` are generated output and gitignored; the code is tracked.

**Re-render and re-run the layout audit after ANY edit.** The audit checks rendered
geometry - panel letters clear of tick labels, legends outside the data area,
annotations not covering curves, nothing past 183x247 mm.

> The audit's text-vs-data rules compare text against each artist's **bounding box**, not
> its geometry. A full-height reference line (`axvline`) therefore makes *any* text in
> that panel flag as "covers plotted data", however empty the region. Those are false
> positives. To settle one, sample along each polyline and test point-in-box, skipping
> two-point reference rules. To place an annotation in a dense panel, compute an
> occupancy grid and find the largest free rectangle rather than guessing.

## Why this is a top-level folder

The repo's top-level directories are pipeline stages - `data_processing`,
`data_augmentation`, `data_analysis`, `data_evaluation`. Figures are their own stage.

More concretely, the figures span data sources: Pillar 5 draws from
`data_augmentation/augmentation_balance_output/`, Pillars 1-3 from
`data_evaluation/vlm_judge/agreement_output/`, Pillar 4 from
`data_evaluation/vlm_zeroshot/analysis/`, and the three Data Records figures straight from
`pathopen_data/` and `pathvqa_data_filtered/`. Nesting the figure code under any one of
those would misfile the others.

### Relationship to `data_analysis/output/`

`data_analysis/` holds 29 exploratory PNGs of dataset statistics (word counts, organ and
magnification breakdowns). **Those are no longer the manuscript's figures.** The three
`dataset_*.ipynb` notebooks here supersede them: they carry the same material, consolidated
into three publication figures at Nature sizing, recomputed against the as-released 229
images rather than the 231 image slots, and with several analyses the PNGs never had — a
Lorenz/Gini concentration panel, a magnification co-occurrence matrix, an organ x category
coverage grid, and a question-opening-word breakdown.

`data_analysis/` is kept because it is the exploratory record and the funnel numbers are
easier to re-derive there. If a dataset-statistics number needs changing, change it in
`data_graphics/dataset_*.ipynb` — that is what the manuscript renders.

## Nature requirements encoded in `nature_style.py`

| | |
|---|---|
| single column | 89 mm |
| double column | 183 mm |
| max height | 247 mm |
| type | 7 pt sans-serif (Helvetica, Arial fallback) |
| rules | 0.5 pt, top/right spines removed |
| format | SVG (submission) + PDF (LaTeX), vector, text kept as text |
| panel labels | bold lowercase **a**, **b**, **c** |

`svg.fonttype: none` keeps text as text rather than converting to paths, so labels stay
editable and searchable in the typeset PDF. `pdf.fonttype: 42` embeds TrueType - journals
reject Type 3.

## Colour

**Okabe-Ito**, the standard colourblind-safe categorical palette; it also survives
greyscale conversion for print proofs.

Each model has a fixed colour assigned once in `MODEL_COLORS`, so Patho-R1 is the same
colour in Figures 6, 7, 8 and 9 (the Pillar 2-5 figures) - a reader tracking one model
across the results should not have to re-read the legend each time. The metadata-field
palette is shared the same way between Fig. 2's Lorenz panel and Fig. 9's balance panels,
so "Categorization" is one hue in Data Records and in Technical Validation. `MODEL_ORDER` runs along the specialization
gradient (general -> broad medical -> biomedical -> pathology -> pathology SOTA), which
is the axis Pillar 4 argues about.

For datasets, PathOPEN carries the one saturated colour and comparators are muted, so the
subject of each comparison is visually obvious without the caption saying so.

## Conventions these figures follow

- **Sample sizes are shown, not hidden.** `annotate_n()` prints `n=` above a bar and
  colours it red below the paper's own n>=30 flag threshold. Several strata in
  sub-pillars 4b/4c fall below it; a figure that concealed that would present 13-item
  cells as if they were as solid as 44-item ones.
- **Null results are labelled `n.s.`**, not left blank.
- **Chance level is drawn** on every accuracy axis. "42% accuracy" means different things
  at 5 options and at 16.

## Tables

`make_tables.py` generates the manuscript's LaTeX tables into `tables/`, which — like
`figures/` — is generated output and is gitignored. The script is tracked; its output is
not, and is reproduced by running it.

```bash
conda activate path-opendata-vlms
cd data_graphics
python make_tables.py            # write every .tex into tables/
python make_tables.py --print    # also echo to stdout
```

Nine tables are generated. **The `S` prefix was dropped from the filenames** when the
rubrics took over the supplementary numbering: the manuscript's `S1-S6` are the six
benchmark rubrics, which are hand-written in the manuscript package, not generated here.
Everything this script emits is a main-text table except the axis-label key.

| file | manuscript | role | source data |
|---|---|---|---|
| `table_data_records.tex` | Table 1 | Item counts, as-collected vs as-released | `pathopen_data/processed/data.json` |
| `table1_dataset_quality.tex` | Table 2 | Pillar 1 headline | `pathopen_vs_pathvqa_mannwhitney.csv`, `evaluator_score_distributions.csv` |
| `table_defect_mechanism.tex` | Table 3 | Pillar 1, per-rater unscorable rates | `evaluator_score_distributions.csv` |
| `table_per_evaluator.tex` | Table 4 | Pillar 1, rater homogeneity | `evaluator_score_distributions.csv`, `evaluator_homogeneity_tests.csv` |
| `table_judge_calibration.tex` | Table 5 | Pillar 2, why judges fail | judge tier-agreement CSVs |
| `table_dataset_comparison.tex` | Table 6 | Pillar 4, CIs + pairwise tests | `dataset_comparison_*.csv` |
| `table_format_rank_correlations.tex` | Table 7 | Pillar 4, cross-format ranks | `format_rank_correlations.csv` |
| `table_benchmark4_rating.tex` | Table 8 | Pillar 5, B4 reliability | `evaluator_score_distributions.csv`, `judge_pathologist_weighted_kappa.csv` |
| `table_label_abbreviations.tex` | Table **S7** | Axis-label key for Fig. 2 | hard-coded, mirrors the notebook's map |

Table order in the manuscript follows **first mention in the text**, which is why
`table_defect_mechanism` precedes `table_per_evaluator` despite the pillar order.

The axis-label key is the one supplementary output: it is a figure-reading aid rather than
primary data, and moving it out keeps the main sequence at 8, under *Scientific Data*'s
"generally fewer than ten tables" guidance.

Preamble: `\usepackage{booktabs}`, `\usepackage{siunitx}`, `\usepackage{graphicx}` and
**`\newsavebox{\pathopentablebox}`** — five of these tables use the savebox pattern below.

### The two sn-jnl traps (read before editing a wide table)

The manuscript class redefines `table` to wrap its body in `threeparttable`. Two
consequences, the second of which cost real debugging time:

1. A scaling box (`\resizebox`/`\scalebox`) inside that `table` leaves a group open and
   the float fails with `Missing \endgroup inserted`.
2. **Silently**, if the float body is a *box* rather than a real `tabular`,
   `threeparttable` **drops the entire float** — table, caption *and* `\label` — with no
   error or warning. The document compiles clean and every `\ref` to it becomes `??`.

The fix both tables use: fill a savebox **before** the float, then place it in the class's
own unwrapped `tableorg` environment, which still numbers as "Table N" and resolves `\ref`.
`table1_dataset_quality.tex` carries the canonical comment block; the others point at it.

Verify with `./check_tables.sh`, which needs `tectonic` on PATH - there is no system
LaTeX and a full TeX Live will not fit (`/data` is at 100%). Structural checks cannot
see page overflow: Table 1 once ran 243 pt off the page with two columns invisible while
passing every column-count and environment check.

> `check_tables.sh` builds each table in a plain `article`, so it defines `tableorg` as a
> thin wrapper around `table` and stubs `\pathopentablebox` and `\TODO`. **It therefore
> cannot reproduce trap 2** — a dropped float needs the real class. After changing a wide
> table, compile the manuscript itself and grep the PDF for `??`.

Tables are generated rather than hand-typed because a transcribed number goes stale
silently when the analysis is re-run. Figures show structure; tables carry precision —
Table 2 collects the exact values behind Figure 5's panels b, c, d and g so a claim can
be checked without reading four axes.
