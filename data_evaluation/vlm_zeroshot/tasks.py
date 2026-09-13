"""Dataset loaders that flatten PathOPEN / PathMMU / PatchVQA into one Task record.

Each dataset stores its MCQs differently:

  PathOPEN   correct and wrong answers in SEPARATE columns, no letters, no fixed order
  PathMMU    an `options` list of letter-prefixed strings, `answer` = one option verbatim
  PatchVQA   identical shape to PathMMU (it is PathBench's patch-level subset)

Normalising them here means the runner and the metrics never branch on dataset. It also
puts option ORDER under our control, which sub-pillar 3b's position-shuffle diagnostic
depends on: for PathOPEN the source order is an artifact of column layout, and for
PathMMU/PatchVQA the published letters bake in one particular arrangement.

`answer_type` distinguishes the three kinds of task this file produces: "mcq" (the
Pillar 3b / 4a MCQ half), "open_ended" (free text, scored with exact-match / BLEU /
BERTScore) and "close_ended" (yes/no, scored as exact match). The runner branches on this
field rather than on dataset name, so a metric is never applied to the wrong answer shape
- scoring a one-word "yes" with a paragraph-similarity metric would be meaningless.
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
# Quilt-VQA ships its images inside HuggingFace Arrow files rather than as files on disk,
# so they are extracted once (see load_quiltvqa_open_ended). They live beside the other
# external datasets rather than inside the repo: they are 217 MB of third-party data, the
# repo has no .gitignore rule that would have caught them, and every other dataset this
# pipeline reads already lives under /data/mn27889/pathology-datasets/.
QUILTVQA_IMAGES_DIR = "/data/mn27889/pathology-datasets/Quilt_VQA/images"

# PathMMU publishes test_tiny as a curated representative subset; using it keeps this
# consistent with the Benchmark 5 scoping in ../vlm_judge/mcq_option_validity_runner.ipynb.
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
    # Open-ended only (sub-pillar 4a): the ground-truth free-text answer that BLEU /
    # BERTScore / exact-match are computed against. Empty for MCQ tasks, where the truth
    # is `correct_index` instead.
    reference: str = ""
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


# ---------------------------------------------------------------------------
# Open-ended tasks (sub-pillar 4a)
#
# Sub-pillar 4a scores every model on all five datasets "using each dataset's native
# metric (accuracy for MCQ/closed-ended; exact-match, BLEU, and BERTScore for
# open-ended)". The MCQ half is above; this is the open-ended half.
#
# Five sources, three different schemas - normalised here so the runner never branches
# on dataset:
#
#   PathOPEN OE   two question/answer columns per row (OE_Question_1/_2)
#   PathOPEN CE   one yes/no pair per row
#   PathVQA       long-form `question`/`answer` with a `question_type` column whose
#                 "close-ended" value marks yes/no items
#   Quilt-VQA     HuggingFace dataset, `answer_type` in {OPEN, CLOSED}, images embedded
#
# `answer_type` on the Task distinguishes "open_ended" from "close_ended": the two need
# different metrics (BLEU/BERTScore vs. exact yes/no match), and conflating them would
# score a one-word "yes" with a text-similarity metric designed for paragraphs.
# ---------------------------------------------------------------------------

PATHVQA_SUBSETS_DIR = PATHOPEN_SUBSETS_DIR  # same folder, different file prefix
PATHVQA_IMAGES_DIR = os.path.join(REPO_ROOT, "pathvqa_data_filtered", "images")


def _clean(value) -> str:
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none") else text


def load_pathopen_open_ended() -> list:
    """PathOPEN open-ended and close-ended, from the same subset CSVs as the MCQs."""
    import pandas as pd
    index = _pathopen_image_index()
    parts = sorted(glob.glob(os.path.join(PATHOPEN_SUBSETS_DIR, "pathopen_vqa_part*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    tasks = []
    for _, row in df.iterrows():
        image_id = row.get("Image_ID")
        if image_id not in index:
            continue
        meta = index[image_id]
        # Two OE question/answer pairs per row; the second is absent on some rows.
        for slot in (1, 2):
            question = _clean(row.get(f"OE_Question_{slot}"))
            answer = _clean(row.get(f"OE_Correct_Answer_{slot}"))
            if not question or not answer:
                continue
            tasks.append(Task(
                item_id=f"PathOPEN_OE::{image_id}::q{slot}",
                dataset="PathOPEN_OE", answer_type="open_ended",
                question=question, image_path=meta["path"],
                reference=answer,
                magnification=meta["magnification"], organ=meta["organ"],
                categorization=meta["categorization"],
            ))
        question = _clean(row.get("CE_Question"))
        answer = _clean(row.get("CE_Correct_Answer"))
        if question and answer:
            tasks.append(Task(
                item_id=f"PathOPEN_CE::{image_id}",
                dataset="PathOPEN_CE", answer_type="close_ended",
                question=question, image_path=meta["path"],
                reference=answer,
                magnification=meta["magnification"], organ=meta["organ"],
                categorization=meta["categorization"],
            ))
    return tasks


def load_pathvqa_open_ended() -> list:
    """Filtered PathVQA - the same 662-row subset the pathologists rated.

    Using the filtered subset rather than raw PathVQA keeps this comparable with the
    judge-side Pillar 1a numbers, which were computed on exactly these rows."""
    import pandas as pd
    parts = sorted(glob.glob(os.path.join(PATHVQA_SUBSETS_DIR, "pathvqa_part*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    tasks = []
    for position, row in df.iterrows():
        question = _clean(row.get("question"))
        answer = _clean(row.get("answer"))
        image_id = _clean(row.get("image_id"))
        if not question or not answer or not image_id:
            continue
        # image_id looks like "train_0476"; the split prefix is also the directory.
        split = image_id.split("_")[0]
        image_path = os.path.join(PATHVQA_IMAGES_DIR, split, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            continue
        close_ended = str(row.get("question_type", "")).strip().lower() == "close-ended"
        # Position is part of the id because image_id repeats - one image carries several
        # questions - so image_id alone would collide.
        tasks.append(Task(
            item_id=f"PathVQA::{position}::{image_id}",
            dataset="PathVQA_CE" if close_ended else "PathVQA_OE",
            answer_type="close_ended" if close_ended else "open_ended",
            question=question, image_path=image_path, reference=answer,
        ))
    return tasks


def load_quiltvqa_open_ended(cache_dir: str | None = None) -> list:
    """Quilt-VQA.

    Unlike every other dataset here, Quilt-VQA ships its images INSIDE the HuggingFace
    Arrow files rather than as files on disk. The answerer API takes an `image_path`, so
    the images have to be materialised once. This is a one-time extraction from the
    dataset the user already downloaded - not a second download.

    Extracted to `/data/mn27889/pathology-datasets/Quilt_VQA/images`, beside the Arrow
    copy and the other external datasets - not into the repo, where 217 MB of third-party
    images would have been committed (nothing in .gitignore covered them).

    The bytes are copied out VERBATIM rather than re-encoded. The source images are JPEG;
    an earlier version of this loader called `.convert("RGB").save(path)` as PNG, which
    turned 216 MB of source data into 1.7 GB of losslessly-re-encoded copies - 8x the
    size, for pixels identical to what was already there. `image["bytes"]` gives the
    original encoded file, so the cache is now the same size and the same quality as the
    source.

    CLOSED answers here are a yes/no verdict followed by an explanation ("Yes,
    hyperchromasia and enlargement are visible..."). The verdict is split off so the
    close-ended metric compares yes/no against yes/no, matching how
    ../vlm_judge/quiltvqa_eval_runner.ipynb treats the same rows.

    **Runs without `datasets` once extracted.** The `path-llava` environment (torch 2.0.1
    / transformers 4.36.2) has no `datasets` package, and installing one risks the same
    dependency cascade that made that environment necessary. So the questions and answers
    are written next to the images as a small JSON sidecar on first extraction, and every
    later call reads that instead of the HuggingFace dataset. Without this, LLaVA-Med and
    Quilt-LLaVA cannot load Quilt-VQA tasks at all - which is a hard failure, not a
    degradation, since `load_all_open_ended` pulls in every subset."""
    if cache_dir is None:
        cache_dir = QUILTVQA_IMAGES_DIR
    os.makedirs(cache_dir, exist_ok=True)
    sidecar = os.path.join(cache_dir, "quiltvqa_qa.json")

    if os.path.exists(sidecar):
        with open(sidecar) as f:
            rows = json.load(f)
    else:
        # First run only: needs `datasets`, so it must happen in the main environment.
        from datasets import Image as HFImage
        from datasets import load_dataset
        # decode=False keeps `image` as {"bytes": ..., "path": ...} so the original
        # encoded file can be written straight out, with no decode/re-encode round trip.
        data = load_dataset("wisdomik/Quilt_VQA")["train"].cast_column(
            "image", HFImage(decode=False))
        rows = []
        for position, row in enumerate(data):
            image_path = os.path.join(cache_dir, f"quilt_{position:05d}.jpg")
            if not os.path.exists(image_path):
                with open(image_path, "wb") as f:
                    f.write(row["image"]["bytes"])
            rows.append({"position": position,
                         "question": _clean(row.get("question")),
                         "answer": _clean(row.get("answer")),
                         "answer_type": str(row.get("answer_type", "")).upper()})
        with open(sidecar, "w") as f:
            json.dump(rows, f)

    yesno = re.compile(r"^\s*(yes|no)\b[\s,.:;!-]*", re.IGNORECASE)
    tasks = []
    for row in rows:
        position = row["position"]
        question = _clean(row.get("question"))
        answer = _clean(row.get("answer"))
        if not question or not answer:
            continue
        image_path = os.path.join(cache_dir, f"quilt_{position:05d}.jpg")
        if not os.path.exists(image_path):
            continue  # image missing from the cache; skip rather than crash mid-run

        if str(row.get("answer_type", "")).upper() == "CLOSED":
            match = yesno.match(answer)
            if match is None:
                continue  # no recoverable verdict - excluded, as on the judge side
            tasks.append(Task(
                item_id=f"QuiltVQA::{position}", dataset="QuiltVQA_CE",
                answer_type="close_ended", question=question,
                image_path=image_path, reference=match.group(1).lower(),
            ))
        else:
            tasks.append(Task(
                item_id=f"QuiltVQA::{position}", dataset="QuiltVQA_OE",
                answer_type="open_ended", question=question,
                image_path=image_path, reference=answer,
            ))
    return tasks


OPEN_ENDED_LOADERS = {
    "PathOPEN": load_pathopen_open_ended,
    "PathVQA": load_pathvqa_open_ended,
    "QuiltVQA": load_quiltvqa_open_ended,
}


def load_all_open_ended() -> list:
    tasks = []
    for loader in OPEN_ENDED_LOADERS.values():
        tasks.extend(loader())
    return tasks
