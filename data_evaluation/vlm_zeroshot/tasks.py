"""Dataset loaders that flatten PathOPEN / PathMMU / PatchVQA into one Task record.

Each dataset stores its MCQs differently:

  PathOPEN   correct and wrong answers in SEPARATE columns, no letters, no fixed order
  PathMMU    an `options` list of letter-prefixed strings, `answer` = one option verbatim
  PatchVQA   identical shape to PathMMU (it is PathBench's patch-level subset)

Normalising them here means the runner and the metrics never branch on dataset. It also
puts option ORDER under our control, which sub-pillar 3b's position-shuffle diagnostic
depends on: for PathOPEN the source order is an artifact of column layout, and for
PathMMU/PatchVQA the published letters bake in one particular arrangement.

`answer_type` is carried on every Task even though only "mcq" is produced today. The
open-ended half of sub-pillar 4a (BLEU/BERTScore over PathOPEN OE, PathVQA OE and
Quilt-VQA) is planned, and the runner branches on this field, so adding it later means
adding a loader and a metric rather than reshaping what already exists.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
from dataclasses import dataclass, field, asdict

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PATHOPEN_SUBSETS_DIR = os.path.join(
    REPO_ROOT, "data_evaluation", "pathologists", "subsets_processing_output", "data")
PATHOPEN_DATA_JSON = os.path.join(REPO_ROOT, "pathopen_data", "processed", "data.json")
PATHMMU_ROOT = "/data/mn27889/pathology-datasets/PathMMU"
PATHMMU_DATA_JSON = os.path.join(PATHMMU_ROOT, "data.json")
PATHMMU_IMAGES_DIR = os.path.join(PATHMMU_ROOT, "images")
# PatchVQA is PathBench's patch-level subset; its images come from PathMMU's pool.
PATCHVQA_JSON = "/data/mn27889/pathology-datasets/PathBench/data/PatchVQA.json"
PATCHVQA_IMAGES_DIR = PATHMMU_IMAGES_DIR

# PathMMU publishes test_tiny as a curated representative subset; using it keeps this
# consistent with the Benchmark 5 scoping in ../vlm/mcq_option_validity_runner.ipynb.
DEFAULT_SPLITS = ("test_tiny",)
# PathCLS is byte-identical between PathMMU and PatchVQA (verified: same img, question,
# options and answer for all 177 test_tiny items). Scored once, under PathMMU.
PATCHVQA_SHARED_CATEGORIES = frozenset({"PathCLS"})

_LETTER_PREFIX = re.compile(r"^\s*([A-Z])\s*[\)\.\:]\s*")


def strip_letter(option: str) -> str:
    """'C) Storiform pattern' -> 'Storiform pattern'. PathOPEN options carry no letter;
    PathMMU/PatchVQA options do, and we re-letter them ourselves after shuffling."""
    return _LETTER_PREFIX.sub("", str(option).strip()).strip()


@dataclass
class Task:
    """One question posed to an answerer model.

    `options` is stored WITHOUT letters and in the order the model should see them;
    `correct_index` indexes into it. Keeping the answer as an index rather than a letter
    is what makes position-shuffling safe - shuffle the list, remap the index, and the
    ground truth follows automatically.
    """
    item_id: str
    dataset: str
    answer_type: str            # "mcq" today; "open_ended" reserved for sub-pillar 4a
    question: str
    image_path: str
    options: list = field(default_factory=list)
    correct_index: int = -1
    # Stratification metadata for sub-pillars 4b/4c. Absent for PathMMU/PatchVQA, which
    # publish no magnification or organ labels - those figures are PathOPEN-only, exactly
    # as Table 2 states.
    magnification: str = ""
    organ: str = ""
    categorization: str = ""
    source_category: str = ""
    split: str = ""

    @property
    def correct_letter(self) -> str:
        return LETTERS[self.correct_index] if 0 <= self.correct_index < len(LETTERS) else ""

    def lettered_options(self) -> list:
        return [f"{LETTERS[i]}) {opt}" for i, opt in enumerate(self.options)]

    def shuffled(self, seed: int) -> "Task":
        """A copy with options permuted and `correct_index` remapped.

        Deterministic in (item_id, seed): the same item under the same seed always gets
        the same permutation, so a resumed run reproduces pre-crash predictions and the
        shuffle condition is reproducible across models."""
        rng = random.Random(f"{self.item_id}::{seed}")
        order = list(range(len(self.options)))
        rng.shuffle(order)
        return Task(
            **{**asdict(self),
               "options": [self.options[i] for i in order],
               "correct_index": order.index(self.correct_index)},
        )


def _pathopen_image_index() -> dict:
    with open(PATHOPEN_DATA_JSON) as f:
        cases = json.load(f)
    root = os.path.dirname(PATHOPEN_DATA_JSON)
    index = {}
    for case in cases:
        for image in case["Images"]:
            original = image.get("Original", "")
            if not original:
                continue
            image_id = original.rsplit("_orig.", 1)[0]
            rel_dir = image["Directory"].split("processed/")[-1]
            index[image_id] = {
                "path": os.path.normpath(os.path.join(root, rel_dir, original)),
                "magnification": image.get("Magnification", "").strip(),
                "organ": str(case.get("Organ", "")).strip(),
                "categorization": str(case.get("Categorization", "")).strip(),
            }
    return index


def load_pathopen_mcq(shuffle_seed: int | None = 0) -> list:
    """PathOPEN MCQs: 1 correct + 4 wrong options per row, from separate columns.

    `shuffle_seed` sets the canonical option order. It is NOT optional in spirit: the
    source columns always list the correct answer first, so feeding them unshuffled would
    let a model score 100% by always answering A. Pass None only to inspect raw order."""
    index = _pathopen_image_index()
    parts = sorted(glob.glob(os.path.join(PATHOPEN_SUBSETS_DIR, "pathopen_vqa_part*.csv")))
    import pandas as pd
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    tasks = []
    for _, row in df.iterrows():
        image_id = row.get("Image_ID")
        if image_id not in index:
            continue
        question = str(row.get("MCQ_OE_Question", "")).strip()
        correct = str(row.get("MCQ_OE_Correct_Answer", "")).strip()
        wrong = [str(row.get(f"MCQ_OE_Wrong_Answer_{i}", "")).strip() for i in (1, 2, 3, 4)]
        wrong = [w for w in wrong if w and w.lower() != "nan"]
        if not question or not correct or len(wrong) < 2:
            continue
        meta = index[image_id]
        task = Task(
            item_id=f"PathOPEN::{image_id}",
            dataset="PathOPEN",
            answer_type="mcq",
            question=question,
            image_path=meta["path"],
            options=[correct] + wrong,
            correct_index=0,
            magnification=meta["magnification"],
            organ=meta["organ"],
            categorization=meta["categorization"],
        )
        tasks.append(task if shuffle_seed is None else task.shuffled(shuffle_seed))
    return tasks


def _load_pathmmu_style(json_path: str, images_dir: str, dataset: str,
                        splits=DEFAULT_SPLITS, drop_categories=frozenset(),
                        shuffle_seed: int | None = 0) -> list:
    with open(json_path) as f:
        data = json.load(f)
    tasks = []
    for category, by_split in data.items():
        if category in drop_categories:
            continue
        for split, items in by_split.items():
            if splits is not None and split not in splits:
                continue
            for item in items:
                image_path = os.path.join(images_dir, item["img"])
                if not os.path.exists(image_path):
                    continue  # SocialPath images were never downloaded; counted by caller
                options = [strip_letter(o) for o in item["options"]]
                answer = strip_letter(item["answer"])
                if answer not in options:
                    continue  # malformed row: answer does not match any option
                item_no = item.get("No", item["img"])
                task = Task(
                    item_id=f"{dataset}::{category}::{split}::{item_no}",
                    dataset=dataset,
                    answer_type="mcq",
                    question=str(item["question"]).strip(),
                    image_path=image_path,
                    options=options,
                    correct_index=options.index(answer),
                    source_category=category,
                    split=split,
                )
                tasks.append(task if shuffle_seed is None else task.shuffled(shuffle_seed))
    return tasks


def load_pathmmu_mcq(splits=DEFAULT_SPLITS, shuffle_seed: int | None = 0) -> list:
    return _load_pathmmu_style(PATHMMU_DATA_JSON, PATHMMU_IMAGES_DIR, "PathMMU",
                               splits=splits, shuffle_seed=shuffle_seed)


def load_patchvqa_mcq(splits=DEFAULT_SPLITS, shuffle_seed: int | None = 0) -> list:
    """PatchVQA minus PathCLS, which duplicates PathMMU's copy exactly."""
    return _load_pathmmu_style(PATCHVQA_JSON, PATCHVQA_IMAGES_DIR, "PatchVQA",
                               splits=splits, drop_categories=PATCHVQA_SHARED_CATEGORIES,
                               shuffle_seed=shuffle_seed)


MCQ_LOADERS = {
    "PathOPEN": load_pathopen_mcq,
    "PathMMU": load_pathmmu_mcq,
    "PatchVQA": load_patchvqa_mcq,
}


def load_all_mcq(shuffle_seed: int | None = 0) -> list:
    tasks = []
    for loader in MCQ_LOADERS.values():
        tasks.extend(loader(shuffle_seed=shuffle_seed))
    return tasks
