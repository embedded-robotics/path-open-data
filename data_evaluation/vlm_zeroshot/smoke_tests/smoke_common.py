"""Shared body of the per-model smoke tests.

Every smoke test asks the same six questions of a different model, so the logic lives here
and each notebook supplies only the model key. That is not just deduplication: if the
checks drift apart between notebooks, cross-model numbers stop being comparable, and
comparability is the entire point of sub-pillar 4a.

Deliberately importable from BOTH environments. The main env (`path-opendata-vlms`) runs
four models; `path-llava` runs the other two. Nothing here imports transformers or llava
at module level - the caller passes an already-loaded model object exposing
`answer(image_path, prompt, max_new_tokens)`.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from metrics import parse_choice, position_bias, score_predictions  # noqa: E402

MAIN_PYTHON = "/data/mn27889/miniconda3/envs/path-opendata-vlms/bin/python"

INSTRUCTION = (
    "Answer the multiple-choice question about the pathology image. "
    "Respond with ONLY the letter of the correct option and nothing else."
)
INSTRUCTION_BLIND = (
    "Answer the following multiple-choice question. "
    "Respond with ONLY the letter of the correct option and nothing else."
)


def load_tasks(dataset: str = "PathOPEN", n: int = 30, shuffle_seed: int = 0) -> list:
    """Task dicts for one dataset.

    `tasks.py` needs pandas and python 3.14, which the `path-llava` env does not have, so
    when running there this shells out to the main interpreter and returns JSON. Same data
    either way - the alternative (duplicating the loaders) would let the two environments
    silently diverge on which items they score."""
    script = f'''
import sys, json; sys.path.insert(0, {_PARENT!r})
import tasks
ts = tasks.MCQ_LOADERS[{dataset!r}](shuffle_seed={shuffle_seed})[:{n}]
print(json.dumps([{{"item_id": t.item_id, "question": t.question,
                   "lettered": t.lettered_options(), "options": t.options,
                   "image_path": t.image_path, "correct_index": t.correct_index,
                   "correct_letter": t.correct_letter, "magnification": t.magnification,
                   "organ": t.organ}} for t in ts]))
'''
    if sys.version_info >= (3, 12):  # main env: import directly, no subprocess needed
        import tasks as _tasks
        loaded = _tasks.MCQ_LOADERS[dataset](shuffle_seed=shuffle_seed)[:n]
        return [{"item_id": t.item_id, "question": t.question,
                 "lettered": t.lettered_options(), "options": t.options,
                 "image_path": t.image_path, "correct_index": t.correct_index,
                 "correct_letter": t.correct_letter, "magnification": t.magnification,
                 "organ": t.organ} for t in loaded]
    result = subprocess.run([MAIN_PYTHON, "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"task loading failed:\n{result.stderr[-1500:]}")
    return json.loads(result.stdout)


def build_prompt(task: dict, blind: bool = False) -> str:
    instruction = INSTRUCTION_BLIND if blind else INSTRUCTION
    return (f"{instruction}\n\nQuestion: {task['question']}\n\n"
            + "\n".join(task["lettered"]) + "\n\nAnswer:")


def shuffle_task(task: dict, seed: int) -> dict:
    """Permute options and remap the correct index.

    Deterministic in (item_id, seed) so a repeat run reproduces the same permutation -
    the shuffle diagnostic must vary because option ORDER changed, not because the
    randomness did."""
    import random
    rng = random.Random(f"{task['item_id']}::{seed}")
    order = list(range(len(task["options"])))
    rng.shuffle(order)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = [task["options"][i] for i in order]
    return {**task,
            "options": options,
            "lettered": [f"{letters[i]}) {o}" for i, o in enumerate(options)],
            "correct_index": order.index(task["correct_index"]),
            "correct_letter": letters[order.index(task["correct_index"])]}


def run_condition(model, task_list: list, blind: bool = False,
                  shuffle_seed: int | None = None, max_new_tokens: int | None = None,
                  show: int = 0) -> tuple:
    """Predictions for one condition. Returns (records, seconds_per_item)."""
    records, timings = [], []
    for i, task in enumerate(task_list):
        item = shuffle_task(task, shuffle_seed) if shuffle_seed is not None else task
        prompt = build_prompt(item, blind=blind)
        started = time.time()
        kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens else {}
        response = model.answer(None if blind else item["image_path"], prompt, **kwargs)
        timings.append(time.time() - started)
        index, method = parse_choice(response, item["options"])
        records.append({"item_id": item["item_id"], "options": item["options"],
                        "correct_index": item["correct_index"],
                        "predicted_index": index, "parse_method": method,
                        "raw_response": response})
        if i < show:
            mark = "OK " if index == item["correct_index"] else "MISS"
            print(f"  {mark} got={response[:40]!r} -> {index} ({method}), "
                  f"correct={item['correct_letter']}")
    return records, sum(timings) / len(timings) if timings else 0.0


def check_vision(model, task_list: list, max_new_tokens: int = 60) -> None:
    """Two different slides, then no slide at all.

    The cheapest way to detect a broken vision path: a corrupted model still emits fluent
    text, it just stops being about the image. The judge pipeline lost days to exactly
    this (4-bit InternVL described a completely different specimen, confidently).

    The no-image reply is diagnostic in its own right. MedGemma says it sees an abstract
    painting - it notices the absence. Both LLaVA models confabulate a plausible
    histology description instead, which means their blind accuracy measures text priors
    PLUS confabulation, and the paper should say so."""
    prompt = "Describe this image in one sentence."
    print("A   :", model.answer(task_list[0]["image_path"], prompt,
                                max_new_tokens=max_new_tokens)[:110])
    print("B   :", model.answer(task_list[min(5, len(task_list) - 1)]["image_path"], prompt,
                                max_new_tokens=max_new_tokens)[:110])
    print("NONE:", model.answer(None, prompt, max_new_tokens=max_new_tokens)[:110])
    print("\n  A and B must differ (vision path live). NONE reveals whether the model "
          "notices\n  a missing image or confabulates - both are informative, neither is "
          "a bug.")


def report(records: list, label: str, seconds_per_item: float = 0.0) -> dict:
    stats = score_predictions(records)
    print(f"{label:22s} acc {stats['accuracy']:5.1f}%  (chance {stats['chance_accuracy']:4.1f}%)  "
          f"parse {stats['parse_rate']:5.1f}%  {seconds_per_item:5.2f}s/item")
    print(f"{'':22s} methods: {stats['parse_methods']}")
    return stats


def blind_comparison(model, task_sets: dict, full_results: dict,
                     max_new_tokens: int | None = None) -> dict:
    """Full vs blind accuracy per dataset (sub-pillar 3b).

    A small gap has two very different causes: the questions are answerable from text
    alone (a real dataset finding), or the image never reached the model (a harness bug
    that would invalidate Pillars 3b and 4). `check_vision` above settles which, so run
    it first.

    At n=30 this diagnostic is weak. MedGemma measured a 0pp gap here while genuinely
    using the image - 25/30 predictions were identical between conditions and the 5 that
    flipped happened to cancel out. Treat anything under ~5pp as uninformative at this
    sample size; the production run uses all 1,428 tasks for exactly this reason."""
    print(f"{'dataset':10s} {'full':>8s} {'blind':>8s} {'gap':>8s}  note")
    out = {}
    for name, task_list in task_sets.items():
        blind_records, _ = run_condition(model, task_list, blind=True,
                                         max_new_tokens=max_new_tokens)
        full_accuracy = score_predictions(full_results[name])["accuracy"]
        blind_accuracy = score_predictions(blind_records)["accuracy"]
        gap = round(full_accuracy - blind_accuracy, 2)
        note = "image used" if gap > 5 else "gap too small to read at n=30"
        print(f"{name:10s} {full_accuracy:7.1f}% {blind_accuracy:7.1f}% {gap:7.1f}pp  {note}")
        out[name] = {"full": full_accuracy, "blind": blind_accuracy, "gap": gap}
    return out


def shuffle_comparison(model, task_sets: dict, seeds=(1, 2, 3),
                       max_new_tokens: int | None = None) -> dict:
    """Accuracy across repeated option permutations (sub-pillar 3b).

    Compare `predicted` against `truth`, not `predicted` alone: raw accuracy can hide
    positional bias when the ground truth happens to be spread the same way. Predictions
    piling onto one letter while truth is spread evenly is the signature."""
    out = {}
    for name, task_list in task_sets.items():
        accuracies = []
        for seed in seeds:
            records, _ = run_condition(model, task_list, shuffle_seed=seed,
                                       max_new_tokens=max_new_tokens)
            accuracies.append(score_predictions(records)["accuracy"])
            if seed == seeds[0]:
                bias = position_bias(records)
                print(f"{name} seed{seed} position bias:")
                print(f"    predicted {bias['predicted_pct']}")
                print(f"    truth     {bias['truth_pct']}")
        spread = max(accuracies) - min(accuracies)
        flag = "  <- large, investigate" if spread > 15 else ""
        print(f"{name}: {accuracies} -> spread {spread:.1f}pp{flag}\n")
        out[name] = {"accuracies": accuracies, "spread": spread}
    return out


def cost_projection(seconds_per_item: float, n_tasks: int = 1428,
                    conditions: int = 5) -> None:
    """conditions = full + blind + 3 shuffles."""
    calls = n_tasks * conditions
    hours = calls * seconds_per_item / 3600
    print(f"throughput          {seconds_per_item:.2f}s/item")
    print(f"calls per model     {calls} ({n_tasks} tasks x {conditions} conditions)")
    print(f"this model          {hours:.1f}h")


CHECKLIST = """
checklist before wiring the full run:
  [ ] section 2: A and B descriptions DIFFER (vision path live)
  [ ] accuracy comfortably above chance on all three datasets
  [ ] parse_rate high - a low rate is a FORMAT problem, not a knowledge problem,
      and must be reported separately or it masquerades as low accuracy
  [ ] blind accuracy below full (or gap explained by section 2's NONE reply)
  [ ] predicted position distribution tracks truth, not piled on one letter
  [ ] shuffle spread small
"""
