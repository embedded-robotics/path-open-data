"""Figure 4 Panel C: does MCQ difficulty follow the pathologist-assigned tiers?

    python tier_difficulty.py

Sub-pillar 2 bins each pathologist-authored wrong answer by its Benchmark 2 score:

    score 2 -> near-miss      subtle, close to the correct answer
    score 1 -> moderate-miss
    score 0 -> far-miss       obviously wrong
    score -1 -> excluded (no tier; the paper's own binning)

The tier *distribution* already shows the graded structure exists. What it cannot show is
whether the grading reflects genuine difficulty or is a rubric artifact - and the paper
names Panel C as "the strongest evidence" precisely because Pillar 2 has no comparator
dataset to argue from.

## The test

If tiering is real, a near-miss distractor should fool a model more often than a far-miss
one. Two complementary measurements, both from the answerer MCQ checkpoints:

1. **Distractor attraction rate** - of the times a model answered wrongly, how often did
   it pick a near-miss vs a far-miss? Normalised by how many distractors of each tier were
   available, so a tier that simply appears more often does not look more attractive.
2. **Item accuracy by hardest distractor present** - items whose hardest distractor is a
   near-miss should be answered correctly less often.

Monotonic decrease across tiers = the tiers track real difficulty.

## The join

The answerer stores options SHUFFLED (option order is randomised at load time so a model
cannot score by always answering "A"), while pathologist tiers are keyed to the source
column order MCQ_OE_Wrong_Answer_1..4. Each stored option is therefore matched back to
its source slot by exact text - verified to match 250/250 on a sample - and the tier read
from that slot's Benchmark 2 score.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
SUBSETS_DIR = os.path.join(REPO_ROOT, "data_evaluation", "pathologists",
                           "subsets_processing_output", "data")
HUMAN_INPUT_DIR = os.path.join(REPO_ROOT, "data_evaluation", "pathologists",
                               "scoring_analysis", "input")
OUTPUT_DIR = os.path.join(_HERE, "analysis")

TIER_LABEL = {2: "near-miss", 1: "moderate-miss", 0: "far-miss"}
TIER_ORDER = ["near-miss", "moderate-miss", "far-miss"]
MIN_BIN_N = 30  # the paper's own threshold for flagging thin strata


def load_tier_lookup() -> dict:
    """{(image_id, wrong_answer_text): tier} from the pooled pathologist ratings.

    Uses the Error Proximity criterion (Benchmark 2's first), which is the one the paper's
    tier definition names. Items the pathologist scored -1 have no tier and are skipped,
    per the same binning `../vlm/wrong_answer_tier_agreement.ipynb` uses."""
    import pandas as pd

    source_parts = sorted(glob.glob(os.path.join(SUBSETS_DIR, "pathopen_vqa_part*.csv")))
    source = pd.concat([pd.read_csv(p) for p in source_parts], ignore_index=True)
    source = source.set_index("Image_ID")

    evaluator_dirs = sorted(glob.glob(os.path.join(HUMAN_INPUT_DIR, "evaluator[0-9]*")),
                            key=lambda p: int(os.path.basename(p).replace("evaluator", "")))
    frames = []
    for directory in evaluator_dirs:
        path = os.path.join(directory, "pathopen_eval_data.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path).drop(index=0).reset_index(drop=True))
    human = pd.concat(frames, ignore_index=True).set_index("Image_ID")

    lookup = {}
    for image_id, row in human.iterrows():
        if image_id not in source.index:
            continue
        source_row = source.loc[image_id]
        for slot in (1, 2, 3, 4):
            text = str(source_row.get(f"MCQ_OE_Wrong_Answer_{slot}", "")).strip()
            if not text or text.lower() == "nan":
                continue
            score = pd.to_numeric(
                row.get(f"Evaluation MCQ_OE_Wrong_Answer_{slot}\n(Benchmark 2)"),
                errors="coerce")
            if pd.isna(score) or int(score) not in TIER_LABEL:
                continue  # -1 or blank: no tier, per the paper's binning
            lookup[(image_id, text)] = TIER_LABEL[int(score)]
    return lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    import pandas as pd
    from scipy.stats import chi2_contingency, kendalltau

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tiers = load_tier_lookup()
    print(f"tier lookup: {len(tiers)} (image, distractor) pairs, "
          f"{collections.Counter(tiers.values())}\n")

    models = args.models or sorted(
        os.path.basename(p).replace("_mcq.jsonl", "")
        for p in glob.glob(os.path.join(_HERE, "checkpoints", "*_mcq.jsonl")))

    attraction_rows, item_rows = [], []
    for model_key in models:
        with open(os.path.join(_HERE, "checkpoints", f"{model_key}_mcq.jsonl")) as f:
            records = [json.loads(line) for line in f if line.strip()]
        records = [r for r in records
                   if r["dataset"] == "PathOPEN" and r["condition"] == "full"]

        available = collections.Counter()   # distractors offered, by tier
        chosen = collections.Counter()      # distractors the model fell for, by tier
        by_hardest = collections.defaultdict(lambda: [0, 0])  # tier -> [correct, total]

        for record in records:
            image_id = record["task_id"].split("::", 1)[1]
            options = record["options"]
            correct_index = record["correct_index"]
            predicted = record["predicted_index"]

            present = []
            for index, option in enumerate(options):
                if index == correct_index:
                    continue
                tier = tiers.get((image_id, str(option).strip()))
                if tier is None:
                    continue
                present.append(tier)
                available[tier] += 1
                if predicted == index:
                    chosen[tier] += 1

            # Hardest distractor on the item: near-miss > moderate > far.
            if present:
                hardest = min(present, key=TIER_ORDER.index)
                by_hardest[hardest][1] += 1
                if predicted == correct_index:
                    by_hardest[hardest][0] += 1

        for tier in TIER_ORDER:
            n_available = available.get(tier, 0)
            if not n_available:
                continue
            attraction_rows.append({
                "model": model_key, "tier": tier,
                "distractors_offered": n_available,
                "times_chosen": chosen.get(tier, 0),
                # Normalised: a tier that simply appears more often must not look more
                # attractive purely because of its frequency.
                "attraction_rate": round(100 * chosen.get(tier, 0) / n_available, 2),
                "low_n": n_available < MIN_BIN_N,
            })
        for tier in TIER_ORDER:
            correct, total = by_hardest.get(tier, [0, 0])
            if total:
                item_rows.append({
                    "model": model_key, "hardest_distractor_tier": tier,
                    "n_items": total, "accuracy": round(100 * correct / total, 2),
                    "low_n": total < MIN_BIN_N,
                })

    attraction_df = pd.DataFrame(attraction_rows)
    item_df = pd.DataFrame(item_rows)

    # Monotonicity: Kendall's tau-b between tier rank (near=0 .. far=2) and the measure.
    # Attraction should DECREASE across that order (negative tau); item accuracy should
    # INCREASE (positive tau). Kendall rather than Pearson because three ordered bins is
    # a rank question, not a linear one.
    trend_rows = []
    for model_key in models:
        for frame, value_column, label, expected in (
                (attraction_df, "attraction_rate", "distractor_attraction", "decreasing"),
                (item_df, "accuracy", "item_accuracy", "increasing")):
            key = "tier" if label == "distractor_attraction" else "hardest_distractor_tier"
            subset = frame[frame.model == model_key]
            ordered = [subset[subset[key] == t][value_column].values
                       for t in TIER_ORDER]
            values = [v[0] for v in ordered if len(v)]
            if len(values) < 3:
                continue
            tau, p = kendalltau([0, 1, 2], values)
            trend_rows.append({
                "model": model_key, "measure": label, "expected": expected,
                "near_miss": values[0], "moderate_miss": values[1], "far_miss": values[2],
                "kendall_tau": round(tau, 3), "p_value": round(p, 4),
                "monotonic_as_expected": bool(
                    (tau < 0) if expected == "decreasing" else (tau > 0)),
            })
    trend_df = pd.DataFrame(trend_rows)

    for name, frame in (("tier_distractor_attraction.csv", attraction_df),
                        ("tier_item_accuracy.csv", item_df),
                        ("tier_monotonicity.csv", trend_df)):
        frame.to_csv(os.path.join(OUTPUT_DIR, name), index=False)

    pd.set_option("display.width", 200)
    print("=== DISTRACTOR ATTRACTION (% of offered distractors that fooled the model) ===")
    print(attraction_df.pivot(index="tier", columns="model",
                              values="attraction_rate").loc[TIER_ORDER].to_string())
    print("\n=== ITEM ACCURACY BY HARDEST DISTRACTOR PRESENT ===")
    print(item_df.pivot(index="hardest_distractor_tier", columns="model",
                        values="accuracy").reindex(TIER_ORDER).to_string())
    print("\n=== MONOTONICITY (Kendall tau) ===")
    print(trend_df.to_string(index=False))

    ok = trend_df[trend_df.measure == "distractor_attraction"].monotonic_as_expected.sum()
    total = (trend_df.measure == "distractor_attraction").sum()
    print(f"\nattraction decreases near->far for {ok}/{total} models")
    print(f"wrote 3 CSVs to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
