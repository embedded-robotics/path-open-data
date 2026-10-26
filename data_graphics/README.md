# Figures and tables

**Five** figures, one per validation pillar, in Nature submission format. (The original
plan called for ten figures numbered 2-11; they were consolidated into five multi-panel
figures, which is why the panel counts are high.)

One notebook per pillar, all sharing `nature_style.py` so the set reads as one design
rather than five independent plots.

```
data_graphics/
  nature_style.py            shared rcParams, palette, mm(), save() - import this first
  pillar1_dataset_quality.ipynb      Fig 1  171x222 mm   7 panels (a-g)
  pillar2_wrong_answers.ipynb        Fig 2  182x153 mm   3 panels (a-c)
  pillar3_dual_mcq.ipynb             Fig 3  172x224 mm   6 panels (a-f)
  pillar4_benchmarking.ipynb         Fig 4  164x236 mm  11 panels (a-k)
  pillar5_augmentation.ipynb         Fig 5  177x213 mm   8 panels (a-h)
  make_tables.py             generates Table 1 + S1-S6 into tables/
  check_tables.sh            compiles every table; catches overfull boxes
  figures/                   fig01_*.svg + .pdf, ... (flat, not per-pillar)
  tables/                    table1_*.tex, tableS1_*.tex, ...
```

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
`data_evaluation/vlm_judge/agreement_output/`, and Pillar 4 from
`data_evaluation/vlm_zeroshot/analysis/`. Nesting the figure code under any one of those
would misfile the others.

Note this is separate from `data_analysis/output/`, which holds the 29 **dataset
statistics** figures already in the paper (word counts, organ and magnification
breakdowns). Those describe the dataset; these report the evaluation.

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
colour in Figures 6, 7, 8 and 9 - a reader tracking one model across the results should
not have to re-read the legend each time. `MODEL_ORDER` runs along the specialization
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

| file | role | source data |
|---|---|---|
| `table1_dataset_quality.tex` | Main text, Pillar 1 | `pathopen_vs_pathvqa_mannwhitney.csv`, `evaluator_score_distributions.csv` |
| `tableS1_per_evaluator.tex` | Supplementary, Pillar 1 | `evaluator_score_distributions.csv`, `evaluator_homogeneity_tests.csv` |
| `tableS2_defect_mechanism.tex` | Supplementary, Pillar 1 | `evaluator_score_distributions.csv` |
| `tableS3_judge_calibration.tex` | Supplementary, Pillar 2 | judge tier-agreement CSVs |
| `tableS4_dataset_comparison.tex` | Supplementary, Pillar 4 | `dataset_comparison_*.csv` |
| `tableS5_format_rank_correlations.tex` | Supplementary, Pillar 4 | `format_rank_correlations.csv` |
| `tableS6_benchmark4_rating.tex` | Supplementary, Pillar 5 | `evaluator_score_distributions.csv`, `judge_pathologist_weighted_kappa.csv` |

Preamble: `\usepackage{booktabs}`, `\usepackage{siunitx}` and **`\usepackage{graphicx}`**
(Table 1 is wrapped in `\resizebox`).

Verify with `./check_tables.sh`, which needs `tectonic` on PATH - there is no system
LaTeX and a full TeX Live will not fit (`/data` is at 100%). Structural checks cannot
see page overflow: Table 1 once ran 243 pt off the page with two columns invisible while
passing every column-count and environment check.

The test build numbers tables 1-7 sequentially because it is a plain `article`. In the
manuscript, S1-S6 must sit inside whatever supplementary environment resets the counter
and prefixes "S". Two cross-references inside captions assume that numbering (Table S6
says "as in Table S1"); they are hard-coded strings, not `\ref`s.

Tables are generated rather than hand-typed because a transcribed number goes stale
silently when the analysis is re-run. Figures show structure; tables carry precision —
Table 1 collects the exact values behind Figure 1's panels b, c, d and g so a claim can
be checked without reading four axes.
