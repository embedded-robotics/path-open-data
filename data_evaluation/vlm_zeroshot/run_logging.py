"""Per-run log files for answerer runs.

Mirrors what `../vlm/parallel_judges.py` does for the judges, for the same reason: a run
prints one line per item, and with several models on different GPUs a shared console
interleaves them into something unreadable. Per-run files also survive a closed notebook,
which matters when a run is measured in hours.

Simpler than the judge version because the concurrency model is different. The judges run
two models inside ONE process (threads sharing a hijacked `sys.stdout`, which needed a
per-thread proxy). Answerers are 4-8B and fit one per GPU, so they run as separate
processes - each gets its own log file with no cross-thread dispatch needed.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

from checkpoint import LOG_DIR


class RunLogger:
    """Writes to a log file and (optionally) mirrors to the console.

    Line-buffered so the file is `tail -f`-able while a run is in flight - the judge runs
    were monitored that way for days and it is the main reason to prefer a file over
    notebook output that only materialises at the end."""

    def __init__(self, model_key: str, task: str, echo: bool = True,
                 log_dir: str = LOG_DIR):
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{model_key}_{task}_{stamp}.log")
        self.model_key = model_key
        self.task = task
        self.echo = echo
        self._stream = open(self.path, "a", buffering=1)
        self._t0 = datetime.now()
        self.write(f"=== {model_key} | {task} | started {self._t0:%Y-%m-%d %H:%M:%S} ===")

    def write(self, message: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {message}"
        self._stream.write(line + "\n")
        if self.echo:
            print(line, file=sys.stdout, flush=True)

    def progress(self, done: int, total: int, extra: str = "") -> None:
        """Progress with an ETA derived from actual elapsed time.

        A projected finish time is the number that decides whether to let a run continue
        or kill it - on this pipeline that judgement has already been worth days of GPU
        time, so it is computed rather than eyeballed."""
        elapsed = (datetime.now() - self._t0).total_seconds()
        rate = elapsed / done if done else 0.0
        remaining = rate * (total - done)
        self.write(f"{done}/{total} ({100 * done / total:.1f}%) "
                   f"| {rate:.1f}s/item | ETA {remaining / 3600:.1f}h {extra}")

    def close(self, summary: str = "") -> None:
        elapsed = (datetime.now() - self._t0).total_seconds()
        if summary:
            self.write(summary)
        self.write(f"=== finished in {elapsed / 3600:.2f}h ===")
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # A traceback belongs in the log too - a run that died at hour 6 is unreadable
        # otherwise, and the console may be long gone.
        if exc is not None:
            import traceback
            self.write(f"FAILED: {exc!r}")
            traceback.print_exception(exc_type, exc, tb, file=self._stream)
        self.close()
        return False
