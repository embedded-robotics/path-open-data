"""Launch every answerer at once, two per GPU.

    python launch_all.py              # start all six, detached
    python launch_all.py --status     # progress of a run already going
    python launch_all.py --dry-run    # show the commands without running them

Each model is a separate OS process because two of the six need a different conda
environment - one process cannot import both transformers 5.x and 4.36. That also means a
crash in one model cannot take down the others, and each writes its own checkpoint file,
so no cross-process locking is needed anywhere.

Six models on three GPUs, two co-resident per card. At the ~0.65 s/item measured for
InternVL that is ~1.3 h per model; co-residency slows each somewhat, so budget 2-4 h for
the whole set rather than the ~8 h six sequential runs would take.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gpu_allocation  # noqa: E402
from checkpoint import CHECKPOINT_DIR, LOG_DIR, checkpoint_path  # noqa: E402

# MCQ: 1,428 tasks x 5 conditions (full + blind + 3 shuffles).
# Open-ended: 2,325 tasks, one condition - shuffling and blinding are meaningless
# without options to permute.
EXPECTED_CALLS = {"mcq": 1428 * 5, "open_ended": 2325}
RUNNER_SCRIPT = {"mcq": "run_answerer.py", "open_ended": "run_answerer_oe.py"}


def build_command(model_key: str, extra: list, task: str = "mcq") -> list:
    return [gpu_allocation.python_for(model_key),
            os.path.join(_HERE, RUNNER_SCRIPT[task]),
            "--model", model_key, "--no-echo", *extra]


def launch(model_keys: list, extra: list, dry_run: bool = False,
           task: str = "mcq") -> dict:
    problems = gpu_allocation.check_fit(model_keys)
    if problems:
        print("REFUSING TO LAUNCH:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(2)

    print(gpu_allocation.describe_allocation(model_keys))
    print()

    os.makedirs(LOG_DIR, exist_ok=True)
    processes = {}
    for model_key in model_keys:
        command = build_command(model_key, extra, task=task)
        if dry_run:
            print(f"  [{model_key}] CUDA_VISIBLE_DEVICES={gpu_allocation.gpu_for(model_key)} "
                  + " ".join(command))
            continue
        # stdout goes to a per-model file: six processes sharing one console interleave
        # into something unreadable, and these files outlive the launching shell.
        stdout_path = os.path.join(LOG_DIR, f"{model_key}_{task}_launch.out")
        # Set CUDA_VISIBLE_DEVICES in the child's environment as well as inside
        # run_answerer.py. The runner does set it before importing torch, so this is
        # belt-and-braces - but without it the dry-run output above would be a lie, and
        # anyone copy-pasting those commands would land every model on GPU 0, which is a
        # reserved judge card.
        child_env = dict(os.environ,
                         CUDA_VISIBLE_DEVICES=gpu_allocation.cuda_visible_devices_for(model_key))
        with open(stdout_path, "a") as stdout:
            process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=child_env)
        processes[model_key] = process.pid
        print(f"  [{model_key}] pid {process.pid} -> GPU {gpu_allocation.gpu_for(model_key)} "
              f"| {stdout_path}")

    if not dry_run:
        state_path = os.path.join(LOG_DIR, f"launch_state_{task}.json")
        with open(state_path, "w") as f:
            json.dump({"started": time.time(), "pids": processes}, f, indent=1)
        print(f"\nstate: {state_path}")
        flag = "" if task == "mcq" else f" --task {task}"
        print(f"progress:  python launch_all.py --status{flag}")
        print(f"follow:    tail -f logs/<model>_{task}_*.log")
    return processes


def status(task: str = "mcq") -> None:
    state_path = os.path.join(LOG_DIR, f"launch_state_{task}.json")
    pids = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            pids = json.load(f).get("pids", {})

    print(f"{'model':12s} {'pid':>8s} {'alive':>6s} {'records':>9s} {'progress':>9s}")
    for model_key in gpu_allocation.MODEL_GPU:
        path = checkpoint_path(model_key, task)
        records = 0
        if os.path.exists(path):
            with open(path) as f:
                records = sum(1 for line in f if line.strip())
        pid = pids.get(model_key)
        alive = "-"
        if pid:
            try:
                os.kill(int(pid), 0)
                alive = "yes"
            except (OSError, ValueError):
                alive = "no"
        pct = 100 * records / EXPECTED_CALLS[task]
        print(f"{model_key:12s} {str(pid or '-'):>8s} {alive:>6s} {records:>9d} {pct:8.1f}%")

    detail = ("1,428 tasks x 5 conditions" if task == "mcq"
              else "2,325 open-ended tasks, 1 condition")
    print(f"\nexpected {EXPECTED_CALLS[task]} calls/model ({detail})")
    print(f"checkpoints: {CHECKPOINT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(gpu_allocation.MODEL_GPU))
    parser.add_argument("--task", choices=["mcq", "open_ended"], default="mcq",
                        help="mcq = Pillar 3b + 4a MCQ half; open_ended = 4a free text")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap tasks per dataset - for a quick end-to-end check")
    args, extra = parser.parse_known_args()

    if args.status:
        status(args.task)
        return
    if args.limit:
        extra = ["--limit", str(args.limit), *extra]
    launch(args.models, extra, dry_run=args.dry_run, task=args.task)


if __name__ == "__main__":
    main()
