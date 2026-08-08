"""
Benchmark rubric definitions and judge-prompt builders for PathOPEN VLM-as-judge.

Rubric text is transcribed verbatim from the PathOPEN paper (Tables 5-10) so that
judge scoring is grounded in the exact same criteria given to the human pathologists.
Benchmark 5 (5a/5b) additionally requires scoring MCQ options standalone, i.e.
without the question stem, per the paper's Sub-pillar 3a methodology.
"""

VALID_SCORES = [-1, 0, 1, 2]

# Each benchmark maps to one or two named criteria, each with its own score->reasoning
# rubric. Order matches the paper's tables and the evaluator CSVs' paired
# "Evaluation ... (Benchmark N)" + "Unnamed" sibling columns.

BENCHMARK_1 = {
    "name": "Benchmark 1: Open-Ended Questions, Correct Answers",
    "criteria": {
        "Knowledge Interpretation/Deduction": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Answer did not address the question correctly, or only addressed minor/unrelated parts of the question, or did not use correct deductive reasoning when provided",
            1: "Answer partially addresses the question by leaving out some important details and/or use partially correct deductions",
            2: "Answer addresses the majority details about the question correctly without the inclusion of any unrelated suggestions and/or provides reasonable deductions to achieve the answer",
        },
        "Visual Grounding": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Answer appears to focus on context clues from text of the question without considering the image information at all or considers the inconsequential parts of the image having no connection to question",
            1: "Answer only tangentially considers the image related to question, but still appears to rely on textual context",
            2: "Answer identified and incorporated the image to directly answer the question",
        },
    },
}

BENCHMARK_2 = {
    "name": "Benchmark 2: Open-Ended Questions, Wrong Answers",
    "criteria": {
        "Error Proximity and Deductive Plausibility": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Answer reflects a wrong conceptual frame or unrelated inference, contradicts key constraints, or ignores the question's core; reasoning is largely irrelevant or speculative.",
            1: "Answer shows partial task understanding with mixed correct/incorrect deductions, but misses one or more critical constraints leading to a materially wrong conclusion.",
            2: "Answer is a near-miss: reasoning path is appropriate and mostly correct, with a single fine-grained mistake or omission causing a subtle deviation from the correct answer.",
        },
        "Visual Grounding Error": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Answer is not grounded in the image (relies on textual cues or irrelevant regions) or cites visual elements not present.",
            1: "Answer references some relevant visual content but mislocalizes or misinterprets key cues (e.g., confuses similar objects/regions), leading to a noticeable error.",
            2: "Answer attends to the correct region and most pertinent features; the mistake arises from a fine-grained visual ambiguity (e.g., off-by-one count, subtle attribute/relationship)",
        },
    },
}

BENCHMARK_3 = {
    "name": "Benchmark 3: Close-Ended Questions and Answers",
    "criteria": {
        "Visual Grounding/Reasoning": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Question can be answered by focusing solely on context clues from text without considering the image information at all",
            1: "Question can be answered by tangentially considering the image information, but still the textual context plays a major role",
            2: "Question can be answered by carefully combining contextual clues from text and visual clues from the image",
        },
    },
}

BENCHMARK_4 = {
    "name": "Benchmark 4: Image Augmentation",
    "criteria": {
        "Clinical Relevance": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Image has minimal clinical relevance which makes it highly unlikely to form any sort of diagnosis",
            1: "Image is clinically relevant to form majority of diagnosis but some patches can mislead the user into creating artificial diagnosis",
            2: "Image is clinically relevant to form majority of diagnoses correctly without any misleading patches",
        },
        "Visual Grounding": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Image contains misleading information or does NOT contain enough information to be combined with question text to form the diagnosis given in the answer",
            1: "Image contains appropriate information which complements the question text to form majority of the diagnosis given in the answer, but the answer contains some information which cannot be linked back to image because of corrupt or misleading image patches",
            2: "Image contains appropriate information which along with the question text can be used to form complete diagnosis given in answer without any information NOT linking back to image",
        },
    },
}

BENCHMARK_5A = {
    "name": "Benchmark 5a: MCQ Correct Option (Standalone Validity)",
    "criteria": {
        "Standalone Completeness": {
            -1: "Unable to comprehend the question, option, and/or image, or unable to make the evaluation",
            0: "Option is a bare label, category name, or fragment (e.g., single word/short phrase) that only makes sense when read alongside the other options; conveys no independent clinical claim",
            1: "Option conveys a partial claim but relies on implicit reference to the question stem or option list to be understood as a complete statement",
            2: "Option reads as a complete, standalone clinical statement that would remain fully meaningful if presented without the question's MCQ framing or the other options",
        },
        "Visual Grounding": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Option can be identified as correct using only the question text and/or general medical knowledge, without needing to examine the image",
            1: "Option is only tangentially supported by the image; the question text or option phrasing provides most of the disambiguating information",
            2: "Option can only be confirmed as correct by directly examining and interpreting relevant visual features in the image",
        },
    },
}

BENCHMARK_5B = {
    "name": "Benchmark 5b: MCQ Wrong Options (Distractor Standalone Plausibility)",
    "criteria": {
        "Distractor Standalone Plausibility": {
            -1: "Unable to comprehend the question, option, and/or image, or unable to make the evaluation",
            0: "Option is an implausible or nonsensical standalone claim, easily dismissed without reference to the image (e.g., wrong organ system, internally contradictory, or too vague to evaluate)",
            1: "Option is a plausible standalone claim in general clinical terms but bears little morphological or contextual resemblance to the correct option for this specific case",
            2: "Option is a plausible, well-formed standalone claim that closely parallels the correct option in structure and clinical register, differing only in a specific factual, diagnostic, or morphological detail",
        },
        "Distractor Visual Grounding Error": {
            -1: "Unable to comprehend the question and/or image, or unable to make the evaluation",
            0: "Option shows no reference to plausible visual features of the image; wrong purely on textual/conceptual grounds",
            1: "Option references visual features that are present in the image but misattributes or misinterprets them (e.g., correct structure, wrong finding)",
            2: "Option references visual features precisely enough that distinguishing it from the correct option requires fine-grained visual discrimination, not just textual knowledge",
        },
    },
}

BENCHMARKS = {
    1: BENCHMARK_1,
    2: BENCHMARK_2,
    3: BENCHMARK_3,
    4: BENCHMARK_4,
    "5a": BENCHMARK_5A,
    "5b": BENCHMARK_5B,
}


def _format_rubric(criteria: dict) -> str:
    blocks = []
    for criterion_name, scale in criteria.items():
        lines = [f'Criterion: "{criterion_name}"']
        for score in sorted(scale.keys(), reverse=True):
            lines.append(f"  Score {score}: {scale[score]}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


JUDGE_SYSTEM_PROMPT = """You are an expert pathologist acting as a strict, rubric-driven grader. \
You will be shown a pathology image and a question/answer pair (or a single answer/option), \
along with one or more scoring criteria. For each criterion, assign exactly one integer score \
from {-1, 0, 1, 2} using ONLY the rubric provided. Do not use any other scale or partial scores.

Respond with a single JSON object mapping each criterion name (exactly as given) to its integer \
score, and nothing else. Example format: {"Criterion Name": 2}"""

# For "thinking" models (e.g. Qwen3-VL-Thinking) whose extended <think>...</think>
# reasoning traces can consume the entire token budget before ever reaching the
# JSON answer. Appended to JUDGE_SYSTEM_PROMPT rather than replacing it, so the
# grading instructions stay identical between thinking and non-thinking judges -
# only the reasoning-budget instruction differs.
MINIMIZE_THINKING_SUFFIX = """

Keep any internal reasoning extremely brief - at most 4-5 short sentences noting \
the key observation per criterion. Do not restate the rubric or the question/answer \
text back in your reasoning. As soon as you have enough to decide, stop reasoning and \
output the final JSON object immediately."""

# InternVL3.5's reasoning ("Thinking") mode is OFF by default and is only enabled by
# supplying this R1-style system prompt - see the model card for InternVL3_5-38B-HF.
# Appended to JUDGE_SYSTEM_PROMPT, not a replacement: the grading instructions must stay
# byte-identical across judges so that a judge-vs-judge comparison isolates the model.
#
# Why this is worth the ~7x slower generation (measured ~36s vs ~5s per call): without it
# InternVL emits a score in ~20 tokens with no analysis at all, and reflexively answers 2.
# On PathOPEN Benchmark 1 that looks accurate only because 94.8% of human scores ARE 2 -
# the same constant-rater trap the ceiling effect creates. On Benchmark 2 (wrong answers),
# where human scores genuinely spread, thinking and non-thinking configs disagreed on
# 5 of 10 criteria, with the non-thinking config giving the less discriminating score.
#
# NOTE: the model card recommends do_sample=True, temperature=0.6 alongside this prompt to
# avoid repetition loops. We deliberately use greedy decoding instead. Measured on one
# Benchmark 2 item, temperature 0.6 returned 0/0, 0/0, 1/1 across three identical runs,
# while greedy returned 1/1 three times at an identical 310 tokens. Non-determinism would
# break the checkpoint-resume contract (a resumed run must reproduce pre-crash scores) and
# would contaminate judge-vs-judge kappa with self-disagreement. The repetition the
# temperature guards against does not appear here - the judge's output is short and
# structured, terminating cleanly in JSON at ~310 tokens.
R1_SYSTEM_PROMPT = """

You are an AI pathologist that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."""


def build_benchmark_1_prompt(question: str, correct_answer: str) -> str:
    rubric = _format_rubric(BENCHMARK_1["criteria"])
    return (
        f"{rubric}\n\n"
        f"Question: {question}\n"
        f"Answer: {correct_answer}\n\n"
        'Return a JSON object with keys "Knowledge Interpretation/Deduction" and "Visual Grounding".'
    )


def build_benchmark_2_prompt(question: str, correct_answer: str, wrong_answer: str) -> str:
    rubric = _format_rubric(BENCHMARK_2["criteria"])
    return (
        f"{rubric}\n\n"
        f"Question: {question}\n"
        f"Correct Answer (for reference only): {correct_answer}\n"
        f"Wrong Answer to evaluate: {wrong_answer}\n\n"
        'Return a JSON object with keys "Error Proximity and Deductive Plausibility" and "Visual Grounding Error".'
    )


def build_benchmark_3_prompt(question: str, answer: str) -> str:
    rubric = _format_rubric(BENCHMARK_3["criteria"])
    return (
        f"{rubric}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        'Return a JSON object with key "Visual Grounding/Reasoning".'
    )


def build_benchmark_4_prompt(question: str, correct_answer: str) -> str:
    """Benchmark 4 is scored against the AUGMENTED image (passed separately to the model),
    reusing the same question/correct-answer pair as the original image."""
    rubric = _format_rubric(BENCHMARK_4["criteria"])
    return (
        f"{rubric}\n\n"
        f"Question: {question}\n"
        f"Answer: {correct_answer}\n\n"
        'Return a JSON object with keys "Clinical Relevance" and "Visual Grounding".'
    )


def build_benchmark_5a_prompt(option_text: str) -> str:
    """Benchmark 5a scores the MCQ correct option standalone: option text only, no
    question stem, per the paper's Sub-pillar 3a methodology."""
    rubric = _format_rubric(BENCHMARK_5A["criteria"])
    return (
        f"{rubric}\n\n"
        f"Option (evaluate standalone, without any question context): {option_text}\n\n"
        'Return a JSON object with keys "Standalone Completeness" and "Visual Grounding".'
    )


def build_benchmark_5b_prompt(option_text: str) -> str:
    """Benchmark 5b scores an MCQ wrong option standalone, without the question stem
    or the correct option, per the same standalone methodology as 5a."""
    rubric = _format_rubric(BENCHMARK_5B["criteria"])
    return (
        f"{rubric}\n\n"
        f"Option (evaluate standalone, without any question context): {option_text}\n\n"
        'Return a JSON object with keys "Distractor Standalone Plausibility" and "Distractor Visual Grounding Error".'
    )
