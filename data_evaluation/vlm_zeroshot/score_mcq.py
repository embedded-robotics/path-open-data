"""Score the MCQ checkpoints into the tables sub-pillars 3b, 4a, 4b and 4c need.

    python score_mcq.py

Companion to `score_open_ended.py`. Pure post-processing over
`checkpoints/{model}_mcq.jsonl` - no GPU, no model loading, safe to re-run any time.
Generation and scoring stay separate so a metric can be changed without re-running
inference, which is the same split the judge pipeline uses.

Writes four CSVs to analysis/:

    mcq_accuracy.csv          per (model, dataset, condition) - the Figure 7 input
    mcq_shortcut.csv          full vs blind vs shuffle per (model, dataset) - Figure 6
    mcq_by_magnification.csv  PathOPEN accuracy by magnification - Figure 8 (4b)
    mcq_by_organ.csv          PathOPEN accuracy by raw organ label (42 bins, sparse)
    mcq_by_organ_system.csv   pooled to organ system - the Figure 9 input (4c)

## Two things this reports that a bare accuracy number hides

**Parse rate is separate from accuracy.** A model that will not emit a parseable answer
is failing differently from one that answers wrongly, and LLaVA-Med does the former on
~6% of items. Folding them together would report a formatting problem as a knowledge
problem - exactly the artifact Pillar 4's cross-model ranking must not contain.

**Chance accuracy is computed per item, not assumed.** Option counts range from 2 to 16
across PathMMU/PatchVQA, so "above chance" means different things per dataset; a flat
1/4 assumption would misjudge every subset.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUTPUT_DIR = os.path.join(_HERE, "analysis")
# The paper's own stated threshold for flagging thin strata (sub-pillars 4b/4c).
MIN_BIN_N = 30
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Sub-pillar 4c is reported at ORGAN SYSTEM level, not raw organ.
#
# PathOPEN's 229 MCQ items spread over 42 distinct organ labels: 21 of them hold 1-4
# items and the median is 4. At that density a heatmap cell is one or two questions, and
# five organs would show "100% accuracy" off a single lucky answer - presenting noise as
# a finding. Pooling to the top-level system (organs are named "System - Subsite") gives
# 10 bins at n>=10 covering ~90% of items.
#
# Even pooled, only Gastrointestinal (n=44) clears the paper's own n>=30 threshold, so
# every cell still carries its `n` and a `low_n` flag. The honest claim is that PathOPEN
# *enables* organ stratification structurally - no comparator dataset carries organ
# metadata at all - while this 157-case sample is too small to power it.
ORGAN_SYSTEM_ALIASES = {
    # Same system, two spellings in the source metadata.
    "CNS": "Central Nervous System",
    # Bare subsites that belong to a system named elsewhere in the data.
    "Bone": "Musculoskeletal",
    "Soft tissue": "Musculoskeletal",
}
# Systems below this are pooled into "Other" rather than shown as their own row - a bin
# of 1-5 items is not a measurement, and a long tail of them makes a heatmap unreadable.
MIN_SYSTEM_N = 10


def organ_system(organ: str) -> str:
    """'Gastrointestinal - Colon' -> 'Gastrointestinal'. Aliases normalised."""
    label = str(organ).split(" - ")[0].strip() if " - " in str(organ) else str(organ).strip()
    return ORGAN_SYSTEM_ALIASES.get(label, label)


def load_records(model_key: str) -> list:
    path = os.path.join(_HERE, "checkpoints", f"{model_key}_mcq.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise(records: list) -> dict:
    """Accuracy, parse rate and chance for one group of predictions."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    parsed = [r for r in records if r["predicted_index"] is not None]
    correct = sum(1 for r in parsed if r["predicted_index"] == r["correct_index"])
    # Per-item chance: 1/n_options averaged, because option counts vary 2-16.
    chance = sum(1.0 / max(r.get("n_options") or 4, 1) for r in records) / n
    return {
        "n": n,
        "n_parsed": len(parsed),
        "parse_rate": round(100 * len(parsed) / n, 2),
        # accuracy counts unparsed as wrong (what a user would see);
        # accuracy_parsed is competence given a usable answer.
        "accuracy": round(100 * correct / n, 2),
        "accuracy_parsed": round(100 * correct / len(parsed), 2) if parsed else float("nan"),
        "chance": round(100 * chance, 2),
        "above_chance": round(100 * correct / n - 100 * chance, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    import pandas as pd
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = args.models or sorted(
        os.path.basename(p).replace("_mcq.jsonl", "")
        for p in glob.glob(os.path.join(_HERE, "checkpoints", "*_mcq.jsonl")))
    data = {m: load_records(m) for m in models}
    print(f"loaded {len(models)} models, "
          f"{sum(len(v) for v in data.values())} records total\n")

    accuracy_rows, shortcut_rows = [], []
    magnification_rows, organ_rows, system_rows = [], [], []

    for model_key, records in data.items():
        datasets = sorted({r["dataset"] for r in records})

        for dataset in datasets:
            subset = [r for r in records if r["dataset"] == dataset]

            # ---- per-condition accuracy (Figure 7 / 4a) ----
            for condition in ("full", "blind", "shuffle"):
                group = [r for r in subset if r["condition"] == condition]
                if not group:
                    continue
                if condition == "shuffle":
                    # One row per seed: sub-pillar 3b measures variance ACROSS seeds, so
                    # averaging them here would destroy the quantity of interest.
                    for seed in sorted({r["seed"] for r in group}):
                        seeded = [r for r in group if r["seed"] == seed]
                        accuracy_rows.append({"model": model_key, "dataset": dataset,
                                              "condition": f"shuffle_seed{seed}",
                                              **summarise(seeded)})
                else:
                    accuracy_rows.append({"model": model_key, "dataset": dataset,
                                          "condition": condition, **summarise(group)})

            # ---- shortcut resistance (Figure 6 / 3b) ----
            full = [r for r in subset if r["condition"] == "full"]
            blind = [r for r in subset if r["condition"] == "blind"]
            shuffles = [r for r in subset if r["condition"] == "shuffle"]
            if not (full and blind):
                continue
            full_stats, blind_stats = summarise(full), summarise(blind)
            seed_accuracies = [summarise([r for r in shuffles if r["seed"] == s])["accuracy"]
                               for s in sorted({r["seed"] for r in shuffles})]

            # Positional bias: how often each letter is PREDICTED vs how often it is the
            # truth. Raw accuracy hides this when the ground truth happens to be spread
            # the same way, so both distributions are needed.
            predicted = [r["predicted_index"] for r in full if r["predicted_index"] is not None]
            truth = [r["correct_index"] for r in full]
            top_letter = max(set(predicted), key=predicted.count) if predicted else None
            shortcut_rows.append({
                "model": model_key, "dataset": dataset,
                "full_accuracy": full_stats["accuracy"],
                "blind_accuracy": blind_stats["accuracy"],
                # The headline 3b number: how much of the score depends on the image.
                "image_gain": round(full_stats["accuracy"] - blind_stats["accuracy"], 2),
                "chance": full_stats["chance"],
                "shuffle_accuracies": ";".join(f"{a:.2f}" for a in seed_accuracies),
                "shuffle_mean": round(sum(seed_accuracies) / len(seed_accuracies), 2) if seed_accuracies else None,
                "shuffle_spread": round(max(seed_accuracies) - min(seed_accuracies), 2) if seed_accuracies else None,
                "most_predicted_letter": LETTERS[top_letter] if top_letter is not None else "",
                "most_predicted_pct": round(100 * predicted.count(top_letter) / len(predicted), 2) if predicted else None,
                "truth_pct_same_letter": round(100 * truth.count(top_letter) / len(truth), 2) if predicted else None,
                "parse_rate": full_stats["parse_rate"],
            })

        # ---- stratified: PathOPEN only (4b, 4c) ----
        # PathMMU and PatchVQA publish no magnification or organ labels, which is exactly
        # why Table 2 scopes these two sub-pillars to PathOPEN.
        pathopen_full = [r for r in records
                         if r["dataset"] == "PathOPEN" and r["condition"] == "full"]

        for magnification in sorted({r["magnification"] for r in pathopen_full if r["magnification"]},
                                    key=lambda m: float(m.lower().replace("x", "") or 0)):
            group = [r for r in pathopen_full if r["magnification"] == magnification]
            magnification_rows.append({"model": model_key, "magnification": magnification,
                                       **summarise(group), "low_n": len(group) < MIN_BIN_N})

        # Raw organ - kept for traceability, but too sparse to plot (see above).
        for organ in sorted({r["organ"] for r in pathopen_full if r["organ"]}):
            group = [r for r in pathopen_full if r["organ"] == organ]
            organ_rows.append({"model": model_key, "organ": organ,
                               **summarise(group), "low_n": len(group) < MIN_BIN_N})

        # Pooled organ system - this is what Figure 9 plots.
        labelled = [r for r in pathopen_full if r["organ"]]
        system_counts = collections.Counter(organ_system(r["organ"]) for r in labelled)
        for record in labelled:
            record["_system"] = (organ_system(record["organ"])
                                 if system_counts[organ_system(record["organ"])] >= MIN_SYSTEM_N
                                 else "Other")
        for system in sorted({r["_system"] for r in labelled},
                             key=lambda s: (s == "Other", s)):  # "Other" sorts last
            group = [r for r in labelled if r["_system"] == system]
            system_rows.append({"model": model_key, "organ_system": system,
                                **summarise(group), "low_n": len(group) < MIN_BIN_N,
                                "pooled_from": system_counts.get(system, len(group))})

    frames = {
        "mcq_accuracy.csv": pd.DataFrame(accuracy_rows),
        "mcq_shortcut.csv": pd.DataFrame(shortcut_rows),
        "mcq_by_magnification.csv": pd.DataFrame(magnification_rows),
        "mcq_by_organ.csv": pd.DataFrame(organ_rows),
        "mcq_by_organ_system.csv": pd.DataFrame(system_rows),
    }
    for filename, frame in frames.items():
        frame.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    pd.set_option("display.width", 220)
    print("=== SHORTCUT RESISTANCE (sub-pillar 3b, Figure 6) ===")
    print(frames["mcq_shortcut.csv"][
        ["model", "dataset", "full_accuracy", "blind_accuracy", "image_gain",
         "chance", "shuffle_spread", "parse_rate"]].to_string(index=False))

    print("\n=== ACCURACY BY MAGNIFICATION (sub-pillar 4b, Figure 8) ===")
    magnification_df = frames["mcq_by_magnification.csv"]
    if len(magnification_df):
        pivot = magnification_df.pivot(index="magnification", columns="model", values="accuracy")
        order = sorted(pivot.index, key=lambda m: float(str(m).lower().replace("x", "") or 0))
        print(pivot.loc[order].to_string())
        thin = magnification_df[magnification_df.low_n]
        print(f"\nbins below n={MIN_BIN_N}: "
              f"{sorted(set(thin.magnification)) if len(thin) else 'none'}")

    print("\n=== ACCURACY BY ORGAN SYSTEM (sub-pillar 4c, Figure 9) ===")
    system_df = frames["mcq_by_organ_system.csv"]
    if len(system_df):
        pivot = system_df.pivot(index="organ_system", columns="model", values="accuracy")
        sizes = (system_df[system_df.model == system_df.model.iloc[0]]
                 .set_index("organ_system")["n"])
        pivot.insert(0, "n", sizes)
        pivot = pivot.sort_values("n", ascending=False)
        print(pivot.to_string())
        thin = sorted(set(system_df[system_df.low_n].organ_system))
        print(f"\nsystems below n={MIN_BIN_N} (annotate these in the figure): {thin}")
        raw_n = frames["mcq_by_organ.csv"].organ.nunique()
        print(f"pooled {raw_n} raw organ labels -> {system_df.organ_system.nunique()} systems "
              f"(those under n={MIN_SYSTEM_N} grouped as 'Other')")

    print("\nwrote:")
    for filename in frames:
        print(f"  {os.path.join(OUTPUT_DIR, filename)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
