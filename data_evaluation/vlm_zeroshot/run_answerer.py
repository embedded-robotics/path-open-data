"""Production runner: one model, all datasets, all conditions.

Invoked once per model, as its own process:

    python run_answerer.py --model qwen25vl

`launch_all.py` starts all six at once on their assigned GPUs. One process per model
rather than threads (which is how the judges run) because two of the six live in a
different conda environment - a single process cannot import both transformers 5.x and
transformers 4.36.

## Conditions (sub-pillar 3b)

    full        image + question + options       the headline accuracy
    blind       question + options, NO image     text-prior shortcut probe
    shuffle_1   options permuted, seed 1  \\
    shuffle_2   options permuted, seed 2   >     position-bias probe
    shuffle_3   options permuted, seed 3  /

5 conditions x 1,428 tasks = 7,140 calls per model.

Everything is checkpointed per call, keyed by `{condition}::{task_id}` (plus the seed for
shuffles), so a killed run resumes exactly where it stopped and re-running only spends GPU
time on what is missing. That pattern has already survived several multi-day interruptions
in the judge pipeline.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gpu_allocation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(gpu_allocation.MODEL_GPU))
    parser.add_argument("--datasets", nargs="*",
                        default=["PathOPEN", "PathMMU", "PatchVQA"])
    parser.add_argument("--conditions", nargs="*",
                        default=["full", "blind", "shuffle"],
                        help="'shuffle' expands to one run per --shuffle-seeds entry")
    parser.add_argument("--shuffle-seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--limit", type=int, default=None,
                        help="cap tasks per dataset - for a quick end-to-end check")
    parser.add_argument("--gpu", type=int, default=None,
                        help="override the GPU from gpu_allocation.py")
    parser.add_argument("--no-echo", action="store_true",
                        help="log to file only; use when several models share a console")
    args = parser.parse_args()

    model_key = args.model

    # CUDA_VISIBLE_DEVICES must be set BEFORE torch is imported anywhere. Every import
    # that could pull in torch is deferred until after this line.
    gpu = args.gpu if args.gpu is not None else gpu_allocation.gpu_for(model_key)
    if gpu in gpu_allocation.RESERVED_GPUS and args.gpu is None:
        print(f"refusing to use GPU {gpu} - reserved for the judge run in ../vlm/",
              file=sys.stderr)
        return 2
    # CUDA_VISIBLE_DEVICES RENUMBERS devices: exposing only physical GPU 4 makes torch
    # see exactly one device and call it cuda:0. So every `device="cuda:0"` below lands on
    # whichever physical card this line names - the model code never translates indices
    # and can never address the wrong card.
    #
    # Verified on this box: with CUDA_VISIBLE_DEVICES set to 4, 6 and 7 in turn, an
    # allocation to cuda:0 appeared on physical 4, 6 and 7 respectively (matched by GPU
    # UUID against nvidia-smi).
    #
    # This is deliberately the opposite of ../vlm/gpu_allocation.py, where each 32B/38B
    # judge is SHARDED over three cards and needs a max_memory dict keyed by logical ids.
    # Here every model is self-contained on one card, so one visible device is simpler and
    # safer.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    problems = gpu_allocation.check_fit()
    if problems:
        print("GPU allocation problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    import tasks  # noqa: E402
    from checkpoint import AnswererCheckpoint, checkpoint_path, make_item_id  # noqa: E402
    from metrics import parse_choice  # noqa: E402
    from run_logging import RunLogger  # noqa: E402

    # Expand "shuffle" into one condition per seed.
    conditions = []
    for name in args.conditions:
        if name == "shuffle":
            conditions.extend(("shuffle", seed) for seed in args.shuffle_seeds)
        else:
            conditions.append((name, None))

    log = RunLogger(model_key, "mcq", echo=not args.no_echo)
    try:
        log.write(f"GPU {gpu} (visible as cuda:0) | env {gpu_allocation.env_for(model_key)} | "
                  f"co-resident with {[m for m in gpu_allocation.models_on(gpu) if m != model_key] or 'nothing'}")

        # Load the task set once; every condition reuses it.
        all_tasks = []
        for dataset in args.datasets:
            loaded = tasks.MCQ_LOADERS[dataset](shuffle_seed=0)
            if args.limit:
                loaded = loaded[:args.limit]
            all_tasks.extend(loaded)
            log.write(f"{dataset}: {len(loaded)} tasks")
        total_calls = len(all_tasks) * len(conditions)
        log.write(f"{len(all_tasks)} tasks x {len(conditions)} conditions = {total_calls} calls")

        checkpoint = AnswererCheckpoint(checkpoint_path(model_key, "mcq"))
        log.write(f"checkpoint holds {len(checkpoint)} records; resuming")

        # Nothing left to do? Skip the model load entirely - on a re-run that saves a
        # minute of loading and, more importantly, does not occupy a GPU for no reason.
        pending = sum(
            1 for cond, seed in conditions for task in all_tasks
            if not checkpoint.is_done(make_item_id(cond, task.item_id, seed))
        )
        if pending == 0:
            log.write("all conditions already complete; nothing to do")
            log.close("SKIPPED - already complete")
            return 0
        log.write(f"{pending} calls pending ({total_calls - pending} already done)")

        # Import the model adapter only now: it pulls in torch, and the LLaVA variant is
        # importable only in the `path-llava` environment.
        if gpu_allocation.env_for(model_key) == "llava":
            from answerer_llava import LLAVA_SPECS, load_llava_answerer
            model = load_llava_answerer(model_key, device="cuda:0")
            max_new_tokens = LLAVA_SPECS[model_key]["max_new_tokens"]
            build_prompt = _build_prompt
        else:
            from answerer_models import build_mcq_prompt, load_answerer, max_new_tokens_for
            model = load_answerer(model_key, device="cuda:0")
            max_new_tokens = max_new_tokens_for(model_key)
            build_prompt = build_mcq_prompt
        log.write(f"loaded | {model.memory_summary()} | max_new_tokens={max_new_tokens}")

        # Record which PHYSICAL card the weights actually landed on, rather than trusting
        # the config. `cuda:0` is a logical name - this resolves it back to a real GPU by
        # UUID, so the log proves placement instead of restating an intention. Cheap
        # insurance against six concurrent models silently piling onto one card.
        try:
            import torch
            properties = torch.cuda.get_device_properties(0)
            # torch 2.0.1 (the pinned `path-llava` env) has no `.uuid` on device
            # properties - it arrived in a later release. Fall back to the device name so
            # the LLaVA runs still record their placement instead of logging a failure.
            uuid = getattr(properties, "uuid", None)
            suffix = f" | uuid {str(uuid)[-12:]}" if uuid else " | uuid unavailable (torch<2.1)"
            log.write(f"physical GPU {gpu} confirmed | {torch.cuda.get_device_name(0)}{suffix}")
        except Exception as exc:  # noqa: BLE001 - diagnostic only, never fatal
            log.write(f"could not confirm physical GPU: {exc!r}")

        done = failed = 0
        for condition, seed in conditions:
            label = condition if seed is None else f"{condition}::seed{seed}"
            blind = condition == "blind"
            for task in all_tasks:
                item_id = make_item_id(condition, task.item_id, seed)
                if checkpoint.is_done(item_id):
                    continue
                item = task.shuffled(seed) if seed is not None else task
                prompt = build_prompt(item.question, item.lettered_options(), blind=blind)
                try:
                    response = model.answer(
                        None if blind else item.image_path, prompt,
                        max_new_tokens=max_new_tokens)
                except Exception as exc:  # noqa: BLE001 - logged, then move on
                    failed += 1
                    log.write(f"FAILED {item_id}: {exc!r} - will retry on the next run")
                    continue
                index, method = parse_choice(response, item.options)
                checkpoint.append({
                    "item_id": item_id,
                    "model": model_key,
                    "condition": condition,
                    "seed": seed,
                    "dataset": item.dataset,
                    "task_id": item.item_id,
                    "predicted_index": index,
                    "correct_index": item.correct_index,
                    "parse_method": method,
                    "n_options": len(item.options),
                    # Kept so a parsing change can be re-applied later without re-running
                    # the model - the checkpoint is the source of truth, the CSV a view.
                    "raw_response": response,
                    "options": item.options,
                    "magnification": item.magnification,
                    "organ": item.organ,
                    "source_category": item.source_category,
                })
                done += 1
                if done % 100 == 0:
                    log.progress(done, pending, f"[{label}]")

        log.close(f"scored {done} new, {failed} failed; "
                  f"checkpoint now holds {len(checkpoint)} records")
        return 1 if failed else 0
    except BaseException as exc:  # noqa: BLE001
        log.write(f"ABORTED: {exc!r}")
        log.close()
        raise


def _build_prompt(question: str, lettered_options: list, blind: bool = False) -> str:
    """Prompt builder for the LLaVA env, which cannot import answerer_models.

    Kept byte-identical to `answerer_models.build_mcq_prompt` - if the two drift, the
    LLaVA models are answering a different question from the other four and the
    cross-model comparison in sub-pillar 4a becomes meaningless."""
    instruction = (
        "Answer the following multiple-choice question. "
        "Respond with ONLY the letter of the correct option and nothing else."
        if blind else
        "Answer the multiple-choice question about the pathology image. "
        "Respond with ONLY the letter of the correct option and nothing else."
    )
    return (f"{instruction}\n\nQuestion: {question}\n\n"
            + "\n".join(lettered_options) + "\n\nAnswer:")


if __name__ == "__main__":
    sys.exit(main())
