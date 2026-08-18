"""Production runner for the OPEN-ENDED half of sub-pillar 4a.

    python run_answerer_oe.py --model qwen25vl

Companion to `run_answerer.py` (the MCQ half). Kept as a separate script rather than a
flag because almost everything differs:

    MCQ                              open-ended
    ---------------------------      ----------------------------------------
    5 conditions (full/blind/x3)     1 condition - shuffling and blinding have
                                     no meaning without options to permute
    16-256 new tokens                512 new tokens - the model must produce a
                                     full clinical sentence, not a letter
    parse a letter, score exactly    store raw text; metrics are a separate pass
    1,428 tasks                      2,325 tasks across 6 subsets

The one thing that is deliberately identical is the checkpoint contract: append-only
JSONL, `fsync` per record, keyed by a stable `item_id`, so a killed run resumes exactly
where it stopped.

Metrics are NOT computed here. BERTScore loads a RoBERTa model and would triple the run
time; more importantly, keeping generation and scoring separate means a metric can be
changed or added later without re-running a single model call - the same split that let
the judge pipeline re-parse its checkpoints after a scoring bug was found.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gpu_allocation  # noqa: E402

# Open-ended answers must be complete clinical statements. PathOPEN references average
# ~20 words, so a budget that suits an MCQ letter would truncate every answer mid-sentence
# and make BLEU meaningless.
#
# Set to 512 as a safety margin. Measured with a 512-token budget on 26 mixed items per
# model (PathOPEN OE / QuiltVQA OE / PathVQA OE / PathOPEN CE), letting each model stop
# naturally:
#
#   model        mean words   max words   est. max tokens   exceeded 256
#   pathor1            53.3          99               139         0/26
#   qwen25vl           41.3          89               125         0/26
#   llavamed           47.6         100               140         0/26
#   internvl8b         28.1          52                73         0/26
#   medgemma           24.7          40                56         0/26
#
# Not one of 130 responses looked truncated, and the longest reached ~140 tokens - so 256
# was already ~1.8x the observed maximum and 512 is ~3.7x. The margin is for rare outliers
# only; every model emits an EOS long before the cap, so the extra budget costs nothing on
# a typical item and is spent only where a model genuinely rambles.
#
# Reference lengths corroborate this: p99 is 53 words (~74 tokens) and the longest
# reference in all 2,325 tasks is 95 words (~133 tokens).
OE_MAX_NEW_TOKENS = 512

OE_INSTRUCTION = (
    "You are a pathologist examining the image. Answer the question in one or two "
    "complete sentences, describing what you observe. Do not restate the question."
)
# Close-ended items need a yes/no verdict FIRST so it can be extracted reliably. Asking
# for bare "yes"/"no" was rejected: PathOPEN's own close-ended references are bare yes/no,
# but models routinely add justification anyway, and a prompt that forbids it just makes
# them disobey. Asking for verdict-then-reason matches how Quilt-VQA's own close-ended
# answers are written, and metrics_open_ended.extract_yes_no takes the first verdict.
CE_INSTRUCTION = (
    "You are a pathologist examining the image. Answer the question with Yes or No, "
    "followed by a brief reason."
)


def build_oe_prompt(question: str, close_ended: bool = False) -> str:
    instruction = CE_INSTRUCTION if close_ended else OE_INSTRUCTION
    return f"{instruction}\n\nQuestion: {question}\n\nAnswer:"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(gpu_allocation.MODEL_GPU))
    parser.add_argument("--datasets", nargs="*", default=["PathOPEN", "PathVQA", "QuiltVQA"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=OE_MAX_NEW_TOKENS)
    parser.add_argument("--no-echo", action="store_true")
    args = parser.parse_args()

    model_key = args.model
    gpu = args.gpu if args.gpu is not None else gpu_allocation.gpu_for(model_key)
    if gpu in gpu_allocation.RESERVED_GPUS and args.gpu is None:
        print(f"refusing GPU {gpu} - reserved for the judge run in ../vlm/", file=sys.stderr)
        return 2
    # Must precede every torch-touching import; see run_answerer.py for why.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import tasks  # noqa: E402
    from checkpoint import AnswererCheckpoint, checkpoint_path  # noqa: E402
    from run_logging import RunLogger  # noqa: E402

    log = RunLogger(model_key, "open_ended", echo=not args.no_echo)
    try:
        log.write(f"GPU {gpu} | env {gpu_allocation.env_for(model_key)} | "
                  f"max_new_tokens={args.max_new_tokens}")

        all_tasks = []
        for dataset in args.datasets:
            loaded = tasks.OPEN_ENDED_LOADERS[dataset]()
            if args.limit:
                loaded = loaded[:args.limit]
            all_tasks.extend(loaded)
            log.write(f"{dataset}: {len(loaded)} tasks")
        log.write(f"{len(all_tasks)} open-ended tasks total")

        checkpoint = AnswererCheckpoint(checkpoint_path(model_key, "open_ended"))
        log.write(f"checkpoint holds {len(checkpoint)} records; resuming")
        pending = [t for t in all_tasks if not checkpoint.is_done(t.item_id)]
        if not pending:
            log.write("all tasks already complete; nothing to do")
            log.close("SKIPPED - already complete")
            return 0
        log.write(f"{len(pending)} pending ({len(all_tasks) - len(pending)} already done)")

        if gpu_allocation.env_for(model_key) == "llava":
            from answerer_llava import load_llava_answerer
            model = load_llava_answerer(model_key, device="cuda:0")
        else:
            from answerer_models import load_answerer
            model = load_answerer(model_key, device="cuda:0")
        log.write(f"loaded | {model.memory_summary()}")

        done = failed = 0
        for task in pending:
            close_ended = task.answer_type == "close_ended"
            prompt = build_oe_prompt(task.question, close_ended=close_ended)
            try:
                response = model.answer(task.image_path, prompt,
                                        max_new_tokens=args.max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.write(f"FAILED {task.item_id}: {exc!r} - will retry next run")
                continue
            checkpoint.append({
                "item_id": task.item_id,
                "model": model_key,
                "dataset": task.dataset,
                "answer_type": task.answer_type,
                "question": task.question,
                # Raw text only. Metrics are a separate pass so BLEU/BERTScore can be
                # recomputed or corrected without spending GPU time again.
                "prediction": response,
                "reference": task.reference,
                "magnification": task.magnification,
                "organ": task.organ,
                "categorization": task.categorization,
            })
            done += 1
            if done % 100 == 0:
                log.progress(done, len(pending))

        log.close(f"generated {done}, {failed} failed; "
                  f"checkpoint now holds {len(checkpoint)} records")
        return 1 if failed else 0
    except BaseException as exc:  # noqa: BLE001
        log.write(f"ABORTED: {exc!r}")
        log.close()
        raise


if __name__ == "__main__":
    sys.exit(main())
