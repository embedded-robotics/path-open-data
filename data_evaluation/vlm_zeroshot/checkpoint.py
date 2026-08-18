"""Crash-safe, resumable checkpointing for answerer runs.

Same contract as `../vlm/checkpoint.py`, which has already survived several multi-day
interruptions in the judge pipeline: append-only JSONL, one record per model call,
`fsync` on every write, keyed by a stable `item_id`. Re-running a scoring loop skips
whatever is already on disk and only spends GPU time on what is missing.

Two differences the answerer needs:

1. **`item_id` includes the CONDITION.** Sub-pillar 3b asks the same question three ways
   - full context, blind (image removed), and position-shuffled. Those are three distinct
   predictions for one dataset item, so the key is `{condition}::{task_id}` rather than
   just the task id. Without that, resuming a full-context run would wrongly mark the
   blind run as already done.

2. **Paths are resolved relative to THIS file, not the process CWD.** The judge pipeline
   builds `CHECKPOINT_DIR` from `os.getcwd()`, which bakes an absolute path into the
   running process - renaming the directory mid-run silently orphans the checkpoint
   (which is exactly why `../vlm/` cannot be renamed while its Benchmark 5 job runs).
   Anchoring to `__file__` makes this package movable.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
LOG_DIR = os.path.join(_HERE, "logs")
OUTPUT_DIR = os.path.join(_HERE, "answerer_output")


def checkpoint_path(model_key: str, task: str) -> str:
    """checkpoints/{model_key}_{task}.jsonl - one file per (model, task) pair.

    Kept separate per model so two models can run concurrently on different GPUs without
    any cross-process locking, exactly as the two judges do."""
    return os.path.join(CHECKPOINT_DIR, f"{model_key}_{task}.jsonl")


class AnswererCheckpoint:
    """Append-only JSONL store for one (model, task) pair."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._completed_ids = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A half-written final line from a crash mid-write. Skip it; the item
                    # is simply re-scored on resume, which is why writes are idempotent.
                    continue
                self._completed_ids.add(record["item_id"])

    def is_done(self, item_id) -> bool:
        return item_id in self._completed_ids

    def append(self, record: dict) -> None:
        """Writes one record and fsyncs immediately, so a crash right after this call
        still leaves the record durably on disk."""
        if "item_id" not in record:
            raise ValueError("checkpoint record must include an 'item_id' key")
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._completed_ids.add(record["item_id"])

    def load_all(self) -> list:
        """Every completed record, e.g. to build the accuracy tables."""
        records = []
        if not os.path.exists(self.path):
            return records
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def __len__(self) -> int:
        return len(self._completed_ids)


def make_item_id(condition: str, task_id: str, seed: int | None = None) -> str:
    """Stable key for one (condition, item) prediction.

    The seed is part of the key for shuffle runs: sub-pillar 3b measures variance ACROSS
    repeated shuffles, so seed 1 and seed 2 are different predictions of the same item and
    must not collide."""
    if seed is None:
        return f"{condition}::{task_id}"
    return f"{condition}::seed{seed}::{task_id}"
