"""Does the wrong-answer tier predict OPEN-ENDED answer quality?

    python tier_open_ended.py

Companion to `tier_difficulty.py`, which tests the same question on MCQ items. The result
here is a NULL, and it is reported rather than dropped: without it a reader cannot tell
whether the MCQ-only scope of Figure 2c was a deliberate scope limit or a favourable
subset.

## Why the test is worth running, and why it is expected to fail

Panel 2c validates the pathologist tiers behaviourally: a near-miss distractor should fool
a model more often than a far-miss one. That test is only possible on MCQ items, because
only there is the tiered distractor actually presented as a choice. PathOPEN's OE-native
wrong answers (668 of the 2,441 rated) are never shown to any model - they exist for the
pathologists to score under Benchmark 2.

So the open-ended analogue can only work INDIRECTLY: if items whose wrong answer was rated
near-miss happen to be intrinsically harder questions, models should score lower on them.
That is a second-order hypothesis, much weaker than the MCQ version, and the measured
answer is that no such effect is detectable.

## What it computes

For each PathOPEN OE item carrying a pathologist tier, the open-ended answer quality of
all six answerer models, grouped by tier. Reported on token-F1 (which contains recall) and
sentence BLEU. Monotonicity is tested with Kendall's tau, with the same caveat as
`tier_difficulty.py`: three ordered bins cannot reach significance, so the direction is
the only readable signal.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
HUMAN_INPUT_DIR = os.path.join(REPO_ROOT, "data_evaluation", "pathologists",
                               "scoring_analysis", "input")
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
OUTPUT_DIR = os.path.join(_HERE, "analysis")

TIER_LABEL = {2: "near-miss", 1: "moderate-miss", 0: "far-miss"}
TIER_ORDER = ["near-miss", "moderate-miss", "far-miss"]


def load_oe_tiers() -> pd.DataFrame:
    """(image_id, slot) -> tier, pooled over the five pathologists.

    Each item is rated by exactly one pathologist per slot in principle, but the five
    evaluator files are read together so that any item rated more than once is averaged
    rather than silently taking whichever file was read last."""
    ratings = []
    pattern = os.path.join(HUMAN_INPUT_DIR, "evaluator*", "pathopen_eval_data.csv")
    for path in sorted(glob.glob(pattern)):
        evaluator = re.search(r"evaluator\d", path).group(0)
        table = pd.read_csv(path)
        id_columns = [c for c in table.columns
                      if c.strip().lower() in ("image_id", "image id", "imageid")]
        if not id_columns:
            continue
        id_column = id_columns[0]
        for slot in (1, 2):
            # The header carries a newline ("Evaluation OE_Wrong_Answer_1\n(Benchmark 2)"),
            # so match on the prefix rather than the full string.
            matches = [c for c in table.columns
                       if c.startswith(f"Evaluation OE_Wrong_Answer_{slot}")]
            if not matches:
                continue
            for _, row in table.iterrows():
                value = str(row[matches[0]]).strip()
                if value not in ("-1", "0", "1", "2") or pd.isna(row[id_column]):
                    continue
                ratings.append({"image_id": row[id_column], "slot": slot,
                                "evaluator": evaluator, "score": int(value)})

    frame = pd.DataFrame(ratings)
    # -1 means "unable to evaluate" and carries no tier - the same exclusion the paper's
    # own binning uses.
    scored = frame[frame.score >= 0]
    pooled = scored.groupby(["image_id", "slot"]).score.mean().round().astype(int)
    return pooled.map(TIER_LABEL).rename("tier").reset_index()


def load_open_ended_scores() -> pd.DataFrame:
    """Per-item open-ended quality for every model, on PathOPEN only."""
    from metrics_open_ended import sentence_bleu, token_f1

    records = []
    for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*_open_ended.jsonl"))):
        model = os.path.basename(path).replace("_open_ended.jsonl", "")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("dataset") != "PathOPEN_OE":
                continue
            parsed = re.match(r"PathOPEN_OE::(.+)::q(\d)", record["item_id"])
            if not parsed:
                continue
            records.append({
                "model": model,
                "image_id": parsed.group(1),
                "slot": int(parsed.group(2)),
                "token_f1": 100 * token_f1(record["prediction"], record["reference"]),
                "bleu": sentence_bleu(record["prediction"], record["reference"]),
            })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()

    tiers = load_oe_tiers()
    scores = load_open_ended_scores()
    joined = scores.merge(tiers, on=["image_id", "slot"], how="inner")

    print(f"OE items carrying a pathologist tier : {len(tiers)}")
    print(f"  tier counts: {tiers.tier.value_counts().to_dict()}")
    print(f"joined model-item rows               : {len(joined)}"
          f" ({joined.groupby(['image_id', 'slot']).ngroups} items x "
          f"{joined.model.nunique()} models)")

    from scipy.stats import kendalltau

    rows = []
    for metric in ("token_f1", "bleu"):
        table = (joined.pivot_table(index="model", columns="tier", values=metric)
                 .reindex(columns=TIER_ORDER))
        print(f"\n=== {metric} by tier of the item's wrong answer ===")
        print(table.round(2).to_string())
        print("mean over models: "
              + "  ".join(f"{t} {table[t].mean():.2f}" for t in TIER_ORDER))
        for model in table.index:
            values = table.loc[model, TIER_ORDER].to_numpy(dtype=float)
            # Predicted: near-miss items are HARDER, so quality should rise near->far.
            tau, p_value = kendalltau([0, 1, 2], values)
            rows.append({"metric": metric, "model": model,
                         "near_miss": values[0], "moderate_miss": values[1],
                         "far_miss": values[2], "kendall_tau": tau, "p_value": p_value,
                         "increases_as_expected": bool(values[0] < values[2])})

    summary = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "tier_open_ended_quality.csv")
    summary.to_csv(path, index=False)

    print("\n=== monotonicity (predicted: quality INCREASES near -> far) ===")
    print(summary.round(4).to_string(index=False))
    for metric in ("token_f1", "bleu"):
        part = summary[summary.metric == metric]
        print(f"  {metric:9s}: {int(part.increases_as_expected.sum())}/{len(part)} "
              f"models in the predicted direction, best P = {part.p_value.min():.3f}")
    print(f"\nwrote {path}")
    print("\nRead this as a NULL result: in open-ended answering the model never sees the\n"
          "distractor, so the tier can act only indirectly through item difficulty. The\n"
          "behavioural validation in Figure 2c is therefore MCQ-only by necessity, not by\n"
          "selection.")


if __name__ == "__main__":
    main()
