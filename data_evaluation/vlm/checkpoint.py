"""
Crash-safe, resumable checkpointing for judge-model runs.

Every judge call is expensive (large VLM inference) and runs can be long
(hundreds to thousands of items). This module separates "run the model and
record raw results" from "post-process results into an evaluator-schema CSV",
so that:

- Every item's raw response + parsed scores is written to disk immediately
  after scoring, not held in memory until the whole loop finishes.
- Re-running the same cell after a crash/OOM/kernel restart skips items
  already checkpointed and only scores what's missing.
- The checkpoint file (JSONL) is the source of truth; the final CSV is a
  cheap, always-safe-to-regenerate view over it.
"""
import json
import os


class JudgeCheckpoint:
    """Append-only JSONL checkpoint store for one (judge, task) pair.

    Each record is a flat dict that must include a stable `"item_id"` key.
    `item_id` should uniquely identify the (image, question/option, ...)
    instance being scored *within this checkpoint file* - callers are
    responsible for choosing a key that's stable across reruns (e.g. a row's
    `Image_ID`, or a source dataset's own row index).
    """

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
                    # A partially-written last line from a crash mid-write; skip it,
                    # the corresponding item will simply be re-scored on resume.
                    continue
                self._completed_ids.add(record["item_id"])

    def is_done(self, item_id) -> bool:
        return item_id in self._completed_ids

    def append(self, record: dict) -> None:
        """Writes one record and fsyncs immediately, so a crash right after this
        call still leaves the record durably on disk."""
        if "item_id" not in record:
            raise ValueError("checkpoint record must include an 'item_id' key")
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._completed_ids.add(record["item_id"])

    def load_all(self) -> list:
        """Reads back every completed record, e.g. to build the final CSV."""
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


def run_with_checkpoint(items: list, item_id_fn, score_fn, checkpoint: JudgeCheckpoint, desc: str = ""):
    """Iterates `items`, skipping any whose `item_id_fn(item)` is already
    checkpointed, calling `score_fn(item) -> dict` (a flat record, WITHOUT
    'item_id' - this wrapper adds it) for the rest, and appending each result
    to `checkpoint` immediately. Errors from `score_fn` are caught, logged, and
    the item is skipped (not checkpointed) rather than aborting the whole run -
    every item scored before the error remains saved, and a rerun will retry
    only the failed/remaining items.

    Returns (n_scored_this_run, n_skipped_already_done, n_failed).
    """
    from tqdm.auto import tqdm

    n_scored = 0
    n_skipped = 0
    n_failed = 0
    for item in tqdm(items, desc=desc):
        item_id = item_id_fn(item)
        if checkpoint.is_done(item_id):
            n_skipped += 1
            continue
        try:
            record = score_fn(item)
        except Exception as e:
            n_failed += 1
            print(f"[checkpoint] FAILED on item_id={item_id!r}: {e!r} - skipping, will retry on next run")
            continue
        record["item_id"] = item_id
        checkpoint.append(record)
        n_scored += 1

    print(f"[checkpoint] {desc}: scored {n_scored} new, {n_skipped} already done, {n_failed} failed this run")
    return n_scored, n_skipped, n_failed
