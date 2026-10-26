"""Bootstrapped CIs and pairwise tests for the cross-dataset difficulty claim.

    python score_dataset_comparison.py

Pillar 4's headline is a comparison BETWEEN datasets - PathOPEN 22.4 points above chance
against PathMMU 25.7 and PatchVQA 20.5 - but until now the pillar reported bare point
estimates: no intervals, no tests. Pillars 1 and 3 both carry bootstrapped CIs, so this
was the one comparative claim in the paper with nothing to say whether the gap it rests on
is distinguishable from noise.

That matters most for the claim being WITHDRAWN. "PathOPEN is mid-range, not hardest"
asserts an ordering; if the intervals overlap, the defensible statement is the stronger
"statistically indistinguishable", which concedes nothing about PathOPEN being easier.

## What is computed

Per dataset and question format:
  - mean performance with a percentile bootstrap 95% CI, resampling ITEMS (not models),
    because the population being generalised over is the dataset's items
  - for MCQ, the same on accuracy-above-chance, which is the comparable quantity when
    option counts differ (2-16 across datasets)
  - pairwise Mann-Whitney U between datasets on per-item correctness, Holm-corrected
    across the three comparisons within each format

Items are pooled across the six models: the question is whether the DATASETS differ, so
each (model, item) pair is one observation. Clustering by item is not modelled, which makes
these intervals mildly anti-conservative; the Holm correction and the effect sizes are
reported so a reader can judge.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from metrics_open_ended import score_close_ended, token_f1

OUTPUT_DIR = os.path.join(_HERE, "analysis")
N_BOOT = 2000
SEED = 0


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """Percentile CI on a mean. Seeded, so the table is reproducible."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return float(values.mean()) if len(values) else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean()
             for _ in range(n_boot)]
    return (float(values.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def load_mcq() -> pd.DataFrame:
    """Per-item MCQ correctness, and per-item chance so above-chance can be bootstrapped."""
    rows = []
    for path in sorted(glob.glob(os.path.join(_HERE, "checkpoints", "*_mcq.jsonl"))):
        model = os.path.basename(path).replace("_mcq.jsonl", "")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("condition") != "full":
                continue
            options = max(record.get("n_options") or 4, 1)
            rows.append({
                "model": model,
                "dataset": record["dataset"],
                "correct": 100.0 * (record.get("predicted_index")
                                    == record.get("correct_index")),
                "chance": 100.0 / options,
            })
    frame = pd.DataFrame(rows)
    frame["above_chance"] = frame.correct - frame.chance
    return frame


def load_free_text() -> pd.DataFrame:
    """Per-item open-ended token F1 and close-ended correctness."""
    rows = []
    for path in sorted(glob.glob(os.path.join(_HERE, "checkpoints",
                                              "*_open_ended.jsonl"))):
        model = os.path.basename(path).replace("_open_ended.jsonl", "")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            dataset = record.get("dataset", "")
            if dataset.endswith("_OE"):
                rows.append({"model": model, "dataset": dataset, "format": "Open-ended",
                             "value": 100 * token_f1(record["prediction"],
                                                     record["reference"])})
            elif dataset.endswith("_CE"):
                scored = score_close_ended(record["prediction"], record["reference"])
                rows.append({"model": model, "dataset": dataset, "format": "Close-ended",
                             "value": 100 * scored["correct"]})
    return pd.DataFrame(rows)


def holm(p_values: list) -> list:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(p_values) - rank) * p_values[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def pairwise(frame: pd.DataFrame, value_column: str, label: str) -> list:
    """Mann-Whitney U between every pair of datasets, Holm-corrected within the format."""
    from scipy.stats import mannwhitneyu

    datasets = sorted(frame.dataset.unique())
    raw, rows = [], []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            left = frame[frame.dataset == datasets[i]][value_column].to_numpy()
            right = frame[frame.dataset == datasets[j]][value_column].to_numpy()
            statistic, p_value = mannwhitneyu(left, right, alternative="two-sided")
            # Rank-biserial from U, so the magnitude is readable alongside the p-value.
            effect = 2 * statistic / (len(left) * len(right)) - 1
            rows.append({"format": label, "a": datasets[i], "b": datasets[j],
                         "mean_a": round(left.mean(), 2), "mean_b": round(right.mean(), 2),
                         "rank_biserial": round(effect, 3), "p_raw": p_value})
            raw.append(p_value)
    for row, adjusted in zip(rows, holm(raw)):
        row["p_holm"] = adjusted
        row["significant"] = bool(adjusted < 0.05)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()

    mcq = load_mcq()
    free_text = load_free_text()

    interval_rows, test_rows = [], []

    for label, frame, column in (("MCQ (accuracy)", mcq, "correct"),
                                 ("MCQ (above chance)", mcq, "above_chance")):
        for dataset, group in frame.groupby("dataset"):
            mean, low, high = bootstrap_ci(group[column].to_numpy())
            interval_rows.append({"format": label, "dataset": dataset,
                                  "n_items": len(group), "mean": round(mean, 2),
                                  "ci_low": round(low, 2), "ci_high": round(high, 2)})
    test_rows += pairwise(mcq, "above_chance", "MCQ (above chance)")

    for fmt, group in free_text.groupby("format"):
        for dataset, by_dataset in group.groupby("dataset"):
            mean, low, high = bootstrap_ci(by_dataset.value.to_numpy())
            interval_rows.append({"format": fmt, "dataset": dataset,
                                  "n_items": len(by_dataset), "mean": round(mean, 2),
                                  "ci_low": round(low, 2), "ci_high": round(high, 2)})
        test_rows += pairwise(group, "value", fmt)

    # Rank-correlation significance for panel d: with six models, is rho = 0.20
    # distinguishable from zero at all?
    from scipy.stats import spearmanr

    ranks = pd.DataFrame({
        "MCQ": mcq[mcq.dataset == "PathOPEN"].groupby("model").correct.mean(),
        "Close-ended": free_text[(free_text.format == "Close-ended")
                                 & (free_text.dataset == "PathOPEN_CE")]
                        .groupby("model").value.mean(),
        "Open-ended": free_text[(free_text.format == "Open-ended")
                                & (free_text.dataset == "PathOPEN_OE")]
                       .groupby("model").value.mean()})
    correlation_rows = []
    for left, right in (("MCQ", "Open-ended"), ("MCQ", "Close-ended"),
                        ("Close-ended", "Open-ended")):
        rho, p_value = spearmanr(ranks[left], ranks[right])
        correlation_rows.append({"pair": f"{left} vs {right}", "n_models": len(ranks),
                                 "spearman_rho": round(rho, 3),
                                 "p_value": round(p_value, 4),
                                 "significant": bool(p_value < 0.05)})

    # Per model x dataset as well as pooled: the figure's panels a-c plot one bar per
    # (model, dataset), so pooled intervals are the wrong granularity for error bars there.
    per_model_rows = []
    for (dataset, model), group in mcq.groupby(["dataset", "model"]):
        mean, low, high = bootstrap_ci(group.correct.to_numpy())
        per_model_rows.append({"format": "MCQ", "dataset": dataset, "model": model,
                               "n_items": len(group), "mean": round(mean, 2),
                               "ci_low": round(low, 2), "ci_high": round(high, 2)})
    for (fmt, dataset, model), group in free_text.groupby(["format", "dataset", "model"]):
        mean, low, high = bootstrap_ci(group.value.to_numpy())
        per_model_rows.append({"format": fmt, "dataset": dataset, "model": model,
                               "n_items": len(group), "mean": round(mean, 2),
                               "ci_low": round(low, 2), "ci_high": round(high, 2)})

    os.makedirs(args.out, exist_ok=True)
    outputs = {
        "dataset_comparison_intervals.csv": pd.DataFrame(interval_rows),
        "per_model_intervals.csv": pd.DataFrame(per_model_rows),
        "dataset_comparison_tests.csv": pd.DataFrame(test_rows),
        "format_rank_correlations.csv": pd.DataFrame(correlation_rows),
    }
    for name, table in outputs.items():
        table.to_csv(os.path.join(args.out, name), index=False)
        print(f"wrote {name} ({len(table)} rows)")

    print("\n=== bootstrapped 95% CIs ===")
    print(outputs["dataset_comparison_intervals.csv"].to_string(index=False))
    print("\n=== pairwise tests (Holm-corrected within format) ===")
    print(outputs["dataset_comparison_tests.csv"]
          [["format", "a", "b", "mean_a", "mean_b", "rank_biserial", "p_holm",
            "significant"]].to_string(index=False))
    print("\n=== cross-format rank correlations (PathOPEN, 6 models) ===")
    print(outputs["format_rank_correlations.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
