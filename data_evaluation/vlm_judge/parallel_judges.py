"""Run several judges concurrently instead of one after the other.

The runner notebooks originally drove judges with a dict comprehension:

    core_checkpoints = {k: run_core_scoring(judges[k], k) for k in MODELS_TO_RUN}

which is strictly sequential - the second judge does not start until the first has
finished every row. With ~24 h for Qwen and ~22 h for InternVL that is ~46 h serial,
versus ~24 h if they overlap.

Overlapping is safe here, and specifically because of three properties:

1. **Separate GPUs.** `gpu_allocation.py` puts each judge in its own NUMA island
   (Qwen 0-2, InternVL 4-6), so they never contend for the same card or PCIe switch.
2. **Separate checkpoint files.** Every writer targets `{model_key}_*.jsonl`, so no two
   threads ever touch the same file - the per-record `fsync` in `JudgeCheckpoint.append`
   stays correct without any cross-thread locking.
3. **The GIL is released during inference.** `model.generate()` spends its time in CUDA
   kernels, so Python threads genuinely overlap rather than time-slicing. Threads (not
   processes) are what we want: each judge holds tens of GB of GPU state that must not
   be forked or re-loaded.

What this does NOT do is parallelize a *single* judge across its own rows. That would
need batching inside `JudgeModel`, and would break the one-call-per-checkpoint-record
contract that makes a run resumable.
"""
import os
import sys
import threading
import traceback
from datetime import datetime


class _PerThreadStdout:
    """Routes each judge thread's `print()` to its own log file.

    `sys.stdout` is process-global, so a thread cannot simply be handed its own stream -
    reassigning it would hijack every other thread's output too. This proxy is installed
    once as `sys.stdout` and dispatches per *calling thread*: a registered judge thread
    writes to its own file, and anything else (the notebook's own cells, tqdm, library
    warnings) falls through to the original stdout untouched.
    """

    def __init__(self, original):
        self._original = original
        self._streams = {}          # thread ident -> open file handle
        self._lock = threading.Lock()

    def register(self, stream):
        with self._lock:
            self._streams[threading.get_ident()] = stream

    def unregister(self):
        with self._lock:
            self._streams.pop(threading.get_ident(), None)

    def _target(self):
        return self._streams.get(threading.get_ident(), self._original)

    def write(self, data):
        return self._target().write(data)

    def flush(self):
        try:
            self._target().flush()
        except ValueError:      # stream already closed during teardown
            pass

    def isatty(self):
        return getattr(self._original, "isatty", lambda: False)()

    def fileno(self):
        # tqdm and friends probe this; always answer for the real stdout, never a log file
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def run_judges_in_parallel(fn, judges: dict, model_keys=None, log_dir=None) -> dict:
    """Calls `fn(judge, model_key)` once per judge, all at the same time.

    Returns {model_key: result}, matching the shape of the dict comprehension it
    replaces. If a judge raises, the exception is re-raised here after every other
    judge has finished - work already checkpointed by the others is never lost, and
    the failing judge's own completed items remain on disk for the next resume.

    With a single judge this is equivalent to calling `fn` directly (one thread).

    `log_dir` (default `logs/` beside this module) receives one file per judge, named
    `{model_key}_{fn_name}_{timestamp}.log`. The scoring loops print one line per item,
    so with two judges running at once a shared console interleaves them into something
    unreadable; per-judge files also survive a closed notebook, which matters for a run
    measured in days. Pass `log_dir=False` to keep everything on the console instead.

    Only per-item chatter is redirected. Progress and failure summaries are still
    printed to the console so the notebook shows liveness without being flooded.
    """
    if model_keys is None:
        model_keys = list(judges)

    fn_name = getattr(fn, "__name__", "judges")
    use_logs = log_dir is not False
    # Resolved to a concrete str whenever logging is on, so the path joins below never
    # have to reason about the None/False sentinels again.
    log_root = ""
    if use_logs:
        log_root = (
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            if log_dir is None else str(log_dir)
        )
        os.makedirs(log_root, exist_ok=True)

    console = sys.stdout
    proxy = console if isinstance(console, _PerThreadStdout) else _PerThreadStdout(console)
    installed_here = proxy is not console
    if installed_here:
        sys.stdout = proxy

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_paths, results, errors = {}, {}, {}

    def worker(key):
        stream = None
        if use_logs:
            path = os.path.join(log_root, f"{key}_{fn_name}_{stamp}.log")
            log_paths[key] = path
            # line-buffered so the file is tail-able while the run is in flight
            stream = open(path, "a", buffering=1)
            stream.write(f"=== {key} | {fn_name} | started {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            proxy.register(stream)
        try:
            results[key] = fn(judges[key], key)
        except BaseException as exc:  # noqa: BLE001 - recorded and re-raised below
            errors[key] = exc
            if stream is not None:
                traceback.print_exc(file=stream)
            # Failures go to the console too - they must not be buried in a log file.
            print(f"[parallel] judge {key!r} FAILED: {exc!r}", file=console, flush=True)
        finally:
            if stream is not None:
                stream.write(f"=== {key} | finished {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
                proxy.unregister()
                stream.close()

    threads = [threading.Thread(target=worker, args=(k,), name=f"judge-{k}") for k in model_keys]
    print(f"[parallel] starting {len(threads)} judge(s) concurrently: {list(model_keys)}",
          file=console, flush=True)
    if use_logs:
        for key in model_keys:
            print(f"[parallel]   {key} -> {os.path.join(log_root, f'{key}_{fn_name}_{stamp}.log')}",
                  file=console, flush=True)
        print("[parallel]   follow with:  tail -f <path>", file=console, flush=True)

    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        if installed_here:
            sys.stdout = console

    print(f"[parallel] all judges finished; ok={list(results)} failed={list(errors)}",
          file=console, flush=True)

    if errors:
        first = next(iter(errors))
        raise RuntimeError(
            f"judge(s) {list(errors)} failed; other judges' work is checkpointed and safe. "
            f"Re-running the cell resumes only what is missing."
        ) from errors[first]
    return results
