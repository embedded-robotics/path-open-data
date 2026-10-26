"""PathOPEN open-ended AND close-ended quality stratified by magnification and organ.

    python score_open_ended_strata.py

Companion to `score_mcq.py`, which produces the same stratification for MCQ items. Written
because Figure 4's panels g-i were MCQ-only while PathOPEN's stated purpose is open-ended
VQA, so the stratified analysis did not cover the question type the dataset exists to
support.

Open-ended items carry the same `magnification` and `organ` metadata as the MCQs and there
are roughly TWICE as many of them (two OE questions per image against one MCQ), which
materially helps the thin bins: 400x goes from 17 MCQ items to 33 open-ended ones, clearing
the paper's own n>=30 flag threshold.

Quality is reported as token F1 - it contains recall, needs no baseline rescaling, and is
the metric used elsewhere for per-item open-ended comparisons. BLEU is also emitted; it is
far noisier per item because a single-sentence reference gives n-gram precision little to
work with.

Thresholds and the organ-system pooling are imported from score_mcq, so the two
stratifications cannot drift apart.
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

from metrics_open_ended import score_close_ended, sentence_bleu, token_f1
from score_mcq import MIN_BIN_N, MIN_SYSTEM_N, organ_system

OUTPUT_DIR = os.path.join(_HERE, "analysis")
MAG_ORDER = ["20x", "40x", "50x", "100x", "200x", "400x"]


def load_open_ended(dataset: str = "PathOPEN_OE") -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(_HERE, "checkpoints",
                                              "*_open_ended.jsonl"))):
        model = os.path.basename(path).replace("_open_ended.jsonl", "")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("dataset") != dataset:
                continue
            rows.append({
                "model": model,
                "magnification": record.get("magnification"),
                "organ": record.get("organ"),
                "token_f1": 100 * token_f1(record["prediction"], record["reference"]),
                "bleu": sentence_bleu(record["prediction"], record["reference"]),
            })
    return pd.DataFrame(rows)


def summarise(group: pd.DataFrame, metric: str) -> dict:
    """One stratum. `metric` is the column to average - token_f1 for open-ended,
    accuracy for close-ended - so both question types share this code path."""
    return {
        "n": len(group),
        metric: round(group[metric].mean(), 2),
        "low_n": len(group) < MIN_BIN_N,
    }


def load_close_ended() -> pd.DataFrame:
    """PathOPEN close-ended, scored with the same yes/no extractor the main scorer uses.

    `score_close_ended` counts an unparsed reply as incorrect rather than dropping it,
    which is what a user of the benchmark would see."""
    rows = []
    for path in sorted(glob.glob(os.path.join(_HERE, "checkpoints",
                                              "*_open_ended.jsonl"))):
        model = os.path.basename(path).replace("_open_ended.jsonl", "")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("dataset") != "PathOPEN_CE":
                continue
            scored = score_close_ended(record["prediction"], record["reference"])
            rows.append({
                "model": model,
                "magnification": record.get("magnification"),
                "organ": record.get("organ"),
                "accuracy": 100 * scored["correct"],
            })
    return pd.DataFrame(rows)


def stratify(frame: pd.DataFrame, metric: str) -> tuple:
    """(by-magnification, by-organ-system) for one question type."""
    frame = frame[frame.magnification.notna() & frame.organ.notna()]

    magnification_rows = []
    for model, by_model in frame.groupby("model"):
        for magnification in MAG_ORDER:
            group = by_model[by_model.magnification == magnification]
            if group.empty:
                continue
            magnification_rows.append({"model": model, "magnification": magnification,
                                       **summarise(group, metric)})

    # Same pooling rule as score_mcq, so every stratified panel shares a row set.
    frame = frame.assign(system=frame.organ.map(organ_system))
    system_counts = frame[frame.model == frame.model.iloc[0]].system.value_counts()
    frame = frame.assign(system_pooled=frame.system.where(
        frame.system.map(system_counts).ge(MIN_SYSTEM_N), "Other"))

    system_rows = []
    for model, by_model in frame.groupby("model"):
        for system in sorted(by_model.system_pooled.unique(),
                             key=lambda s: (s == "Other", s)):
            group = by_model[by_model.system_pooled == system]
            system_rows.append({"model": model, "organ_system": system,
                                **summarise(group, metric)})
    return pd.DataFrame(magnification_rows), pd.DataFrame(system_rows)


def magnification_tests() -> pd.DataFrame:
    """Kruskal-Wallis H per model x question type: does accuracy depend on magnification?

    Run across all three question types because the result is a NULL and that is only
    convincing if it holds everywhere. With 18 tests and no multiplicity correction, two
    hits at P<0.05 is about what chance produces - and the two are different models, so no
    model is magnification-sensitive across types and no type is across models."""
    from scipy.stats import kruskal

    rows = []
    sources = (("MCQ", None), ("Close-ended", "accuracy"), ("Open-ended", "token_f1"))
    mcq = pd.read_csv(os.path.join(OUTPUT_DIR, "mcq_by_magnification.csv"))

    for question_type, metric in sources:
        if question_type == "MCQ":
            # The MCQ scorer already stores per-bin accuracy; the per-item correctness it
            # was computed from is in the checkpoints.
            frame = _mcq_items()
            value_column = "correct"
        else:
            frame = (load_close_ended() if question_type == "Close-ended"
                     else load_open_ended())
            value_column = metric
        frame = frame[frame.magnification.notna()]
        for model, by_model in frame.groupby("model"):
            groups = [by_model[by_model.magnification == m][value_column].to_numpy()
                      for m in MAG_ORDER]
            groups = [grp for grp in groups if len(grp) > 1]
            if len(groups) < 2:
                continue
            statistic, p_value = kruskal(*groups)
            rows.append({"model": model, "question_type": question_type,
                         "H": round(statistic, 2), "p": round(p_value, 4),
                         "significant": bool(p_value < 0.05)})
    return pd.DataFrame(rows)


def _mcq_items() -> pd.DataFrame:
    """Per-item MCQ correctness with metadata, for the Kruskal-Wallis test."""
    rows = []
    for path in sorted(glob.glob(os.path.join(_HERE, "checkpoints", "*_mcq.jsonl"))):
        model = os.path.basename(path).replace("_mcq.jsonl", "")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("dataset") != "PathOPEN" or record.get("condition") != "full":
                continue
            rows.append({
                "model": model,
                "magnification": record.get("magnification"),
                "correct": float(record.get("predicted_index")
                                 == record.get("correct_index")),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT_DIR)
    args = parser.parse_args()

    open_ended = load_open_ended()
    close_ended = load_close_ended()
    print(f"PathOPEN open-ended predictions : {len(open_ended)}")
    print(f"PathOPEN close-ended predictions: {len(close_ended)}")

    oe_magnification, oe_system = stratify(open_ended, "token_f1")
    ce_magnification, ce_system = stratify(close_ended, "accuracy")

    os.makedirs(args.out, exist_ok=True)
    outputs = {
        "open_ended_by_magnification.csv": oe_magnification,
        "open_ended_by_organ_system.csv": oe_system,
        "close_ended_by_magnification.csv": ce_magnification,
        "close_ended_by_organ_system.csv": ce_system,
        "magnification_tests_all_types.csv": magnification_tests(),
    }
    for name, table in outputs.items():
        table.to_csv(os.path.join(args.out, name), index=False)
        print(f"wrote {name} ({len(table)} rows)")

    for label, magnification, systems in (("open-ended", oe_magnification, oe_system),
                                          ("close-ended", ce_magnification, ce_system)):
        counts = magnification.groupby("magnification").n.first().reindex(MAG_ORDER)
        below = int((counts < MIN_BIN_N).sum())
        system_n = systems.groupby("organ_system").n.first()
        print(f"\n{label}: {below} of {len(counts)} magnification bins below "
              f"n>={MIN_BIN_N}; {int((system_n >= MIN_BIN_N).sum())} of {len(system_n)} "
              f"organ systems clear it")
        print("   " + "  ".join(f"{m} n={int(counts[m])}" for m in MAG_ORDER
                                if m in counts.index and pd.notna(counts[m])))


if __name__ == "__main__":
    main()
