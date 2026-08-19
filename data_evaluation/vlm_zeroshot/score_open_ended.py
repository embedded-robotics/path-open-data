"""Score the open-ended checkpoints: per (model, dataset) metrics including BERTScore.

    python score_open_ended.py                 # all models, writes analysis/ CSVs
    python score_open_ended.py --no-bertscore  # fast pass, BLEU/chrF/F1 only

Separate from `run_answerer_oe.py` on purpose. Generation stores raw text; scoring is a
re-runnable pass over the checkpoints, so a metric can be added or corrected without
spending GPU time on inference again - the same split that let the judge pipeline
re-parse its checkpoints after a scoring bug was found.

BERTScore needs a GPU (it loads roberta-large) and is the reason this is a script rather
than a notebook cell: ~10,700 pairs takes long enough to want it detached and logged.

## Why BERTScore matters here, and BLEU alone does not

The models answer in their own words. On a 12-item probe, MedGemma scored BLEU 8.2 and
BERTScore 87.75 on the SAME predictions - it was saying the right thing in different
words. BLEU is also strongly confounded by length: Quilt-LLaVA writes ~76 words against
~17-word references and ranks last on BLEU everywhere, which measures verbosity more than
correctness. BERTScore is far less sensitive to both.

PathVQA_OE is the extreme case - its references average 7.8 words, so every model's BLEU
collapses to ~0.2-0.6 there. Reporting BLEU alone would say all six models fail on
PathVQA, when what actually differs is answer length.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

OUTPUT_DIR = os.path.join(_HERE, "analysis")
# roberta-large is bert-score's default for English and the model most published
# BERTScore figures use, so keeping it makes our numbers comparable to the literature.
BERTSCORE_MODEL = "roberta-large"


def load_records(model_key: str) -> list:
    path = os.path.join(_HERE, "checkpoints", f"{model_key}_open_ended.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before torch loads
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import pandas as pd
    from metrics_open_ended import score_close_ended, score_open_ended_corpus

    models = args.models or sorted(
        os.path.basename(p).replace("_open_ended.jsonl", "")
        for p in glob.glob(os.path.join(_HERE, "checkpoints", "*_open_ended.jsonl")))
    print(f"scoring {len(models)} models on GPU {args.gpu} "
          f"(BERTScore {'OFF' if args.no_bertscore else 'ON'})", flush=True)

    open_rows, close_rows = [], []
    for model_key in models:
        records = load_records(model_key)
        by_dataset = {}
        for record in records:
            by_dataset.setdefault(record["dataset"], []).append(record)

        for dataset, subset in sorted(by_dataset.items()):
            started = time.time()
            if subset[0]["answer_type"] == "close_ended":
                scored = [score_close_ended(r["prediction"], r["reference"]) for r in subset]
                parsed = sum(1 for s in scored if s["parsed"])
                close_rows.append({
                    "model": model_key, "dataset": dataset, "n": len(subset),
                    # A model that gives no verdict is a different failure from one that
                    # guesses wrong; keeping them apart stops a refusal being scored as
                    # an error.
                    "verdict_rate": round(100 * parsed / len(subset), 2),
                    "accuracy": round(100 * sum(s["correct"] for s in scored) / len(subset), 2),
                })
            else:
                metrics = score_open_ended_corpus(
                    [r["prediction"] for r in subset], [r["reference"] for r in subset],
                    compute_bertscore=not args.no_bertscore,
                    bertscore_model=BERTSCORE_MODEL, device="cuda:0",
                    batch_size=args.batch_size)
                open_rows.append({"model": model_key, "dataset": dataset, **metrics})
            print(f"  {model_key:12s} {dataset:14s} n={len(subset):4d} "
                  f"({time.time() - started:.1f}s)", flush=True)

    open_df = pd.DataFrame(open_rows)
    close_df = pd.DataFrame(close_rows)
    open_path = os.path.join(OUTPUT_DIR, "open_ended_metrics.csv")
    close_path = os.path.join(OUTPUT_DIR, "close_ended_metrics.csv")
    open_df.to_csv(open_path, index=False)
    close_df.to_csv(close_path, index=False)

    pd.set_option("display.width", 200)
    print("\n=== OPEN-ENDED ===")
    print(open_df.to_string(index=False))
    print("\n=== CLOSE-ENDED ===")
    print(close_df.to_string(index=False))
    print(f"\nwrote {open_path}\n      {close_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
