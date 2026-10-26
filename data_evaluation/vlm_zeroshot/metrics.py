"""Parsing model output into a choice, and scoring it.

The hard part of MCQ evaluation is not the accuracy arithmetic - it is deciding what a
model actually answered. A 7B model asked for "a single letter" will variously reply
"B", "B)", "(B)", "The answer is B.", "**B**", "B) Storiform pattern", or restate the
option text with no letter at all. Counting only bare-letter replies would understate
every model, and would understate the chattier ones most - turning a formatting
difference into a fake accuracy gap, which is precisely the kind of artifact Pillar 4's
cross-model ranking must not contain.

So parsing runs a cascade from most to least reliable, and records WHICH rule fired
(`parse_method`) so the paper can report how much of each score rests on lenient
matching. `unparsed` is kept as its own outcome rather than silently scored wrong: a
model that cannot follow the output format is a finding, not a zero.
"""
from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# "B", "B)", "(B)", "B.", "B:" possibly wrapped in markdown emphasis, as the WHOLE reply.
_BARE_LETTER = re.compile(r"^\s*[\*_`]*\(?\s*([A-Z])\s*\)?[\.\):]?[\*_`]*\s*$")
# "The answer is B", "Answer: (B)", "final answer - B", "I would choose option C".
# The optional connective group is what makes "answer IS D" work: without it the letter
# has to sit immediately after the cue word, which misses the most common phrasing of all.
# Connectives are matched non-greedily and capped so this cannot leap across a sentence
# and grab a letter belonging to some later clause.
_STATED = re.compile(
    r"(?:answer|option|choice|select(?:ed)?)\b"          # cue word
    r"(?:\s+(?:is|are|would\s+be|should\s+be|=|:))?"      # optional connective
    r"[^A-Za-z0-9]{0,8}"                                  # punctuation/space/bracket
    r"\(?\s*([A-Z])\s*\)?(?![A-Za-z])",                   # the letter, not mid-word
    re.IGNORECASE)
# A letter-prefixed option echoed back: "B) Storiform pattern"
_LEADING_OPTION = re.compile(r"^\s*[\*_`]*\(?\s*([A-Z])\s*[\)\.\:]\s*\S")


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace. Punctuation is KEPT - the substring rule needs the
    option text to match as written."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# Words that carry no discriminative signal between MCQ distractors. Excluded from the
# coverage score so that two options differing only in their diagnosis are not judged
# similar merely because they share a sentence frame ("in that context, the most likely
# diagnosis is ...").
_STOPWORDS = frozenset("""
a an the this that these those is are was were be been being of in on at to for with by
and or as it its from into their there here which what when where who whom how such
most likely given shows show showing depicted image images context also would could
""".split())


def _word_set(text: str) -> set:
    """Content words, punctuation stripped. Used only by the token-overlap rule."""
    words = re.findall(r"[a-z0-9]+", _normalize(text))
    content = {w for w in words if w not in _STOPWORDS}
    # An option made entirely of stopwords ("It is the same") would otherwise become an
    # empty set and score coverage 0/0; fall back to the raw words in that case.
    return content or set(words)


def parse_choice(response: str, options: list) -> tuple:
    """(index or None, parse_method).

    Cascade, strictest first. Each rule is only consulted if the previous one found
    nothing, so a clean "B" never reaches the fuzzy text matching below."""
    if response is None:
        return None, "empty"
    text = str(response).strip()
    if not text:
        return None, "empty"
    n = len(options)
    valid = set(LETTERS[:n])

    # 1. The whole reply is just a letter.
    m = _BARE_LETTER.match(text)
    if m and m.group(1) in valid:
        return LETTERS.index(m.group(1)), "bare_letter"

    # 2. The reply opens with a letter-prefixed option ("B) Storiform pattern").
    m = _LEADING_OPTION.match(text)
    if m and m.group(1) in valid:
        return LETTERS.index(m.group(1)), "leading_option"

    # 3. An explicit "the answer is X" anywhere. Take the LAST such mention: a model that
    #    reasons aloud often names candidates before committing, and the final statement
    #    is the commitment.
    stated = [g for g in _STATED.findall(text) if g.upper() in valid]
    if stated:
        return LETTERS.index(stated[-1].upper()), "stated_answer"

    # 4. Exact option text somewhere in the reply. Longest option first, so that an option
    #    which is a substring of another ("Necrosis" vs "Necrosis and hemorrhage") cannot
    #    steal the match from the longer one the model actually wrote.
    haystack = _normalize(text)
    by_length = sorted(range(n), key=lambda i: -len(options[i]))
    for i in by_length:
        needle = _normalize(options[i])
        if needle and needle in haystack:
            return i, "option_text"

    # 5. A standalone letter token anywhere. Only accepted if EXACTLY one valid letter
    #    appears - "between B and D" is genuine ambiguity, not an answer.
    loose = {g for g in re.findall(r"\b([A-Z])\b", text) if g in valid}
    if len(loose) == 1:
        return LETTERS.index(next(iter(loose))), "loose_letter"

    # 6. Token overlap, for models that answer the question in their own words without
    #    ever referencing the options. LLaVA-Med does this on most items:
    #
    #      model:    "The pattern of inflammation depicted in this image is
    #                 necrotizing granulomatous inflammation."
    #      option A: "The pattern of inflammation is necrotizing granulomatous
    #                 inflammation"
    #
    #    That is unambiguously option A, but rule 4 misses it because the model inserted
    #    "depicted in this image", breaking the exact substring. Without this rule those
    #    answers are scored `unparsed`, which would report a model that answered correctly
    #    as one that failed to answer - the exact artifact the parse-rate column exists to
    #    prevent. Measured on LLaVA-Med: rules 1-5 parse 7/12, this recovers 4 more.
    #
    #    Scored as the fraction of the OPTION's words present in the response, not the
    #    reverse: a long rambling reply should not be penalised for containing extra words,
    #    but it must cover what the option actually says.
    # Punctuation must be stripped before comparing WORDS. `_normalize` only lowercases
    # and collapses whitespace, which is right for the substring rule above but wrong
    # here: "lymphoma." and "lymphoma" are the same word, and treating them as different
    # silently depresses coverage for every option ending a sentence.
    option_words = [_word_set(o) for o in options]
    response_words = _word_set(text)
    coverage = [len(ow & response_words) / len(ow) if ow else 0.0 for ow in option_words]
    order = sorted(range(n), key=lambda i: -coverage[i])
    best = order[0]
    runner_up = coverage[order[1]] if n > 1 else 0.0
    # Two guards, because MCQ distractors are deliberately similar to the correct option:
    # the winner must cover most of its option, AND beat the next-best by a clear margin.
    # A near-tie means the response is equally consistent with two options, which is not
    # an answer - leaving it unparsed is more honest than picking one.
    if coverage[best] >= 0.75 and (coverage[best] - runner_up) >= 0.10:
        return best, "token_overlap"

    return None, "unparsed"


def score_predictions(records: list) -> dict:
    """Accuracy over a list of {correct_index, predicted_index, parse_method} dicts.

    Two accuracies are reported because they answer different questions:
      accuracy          - over ALL items, unparsed counted wrong. What a user would see.
      accuracy_parsed   - over parsed items only. The model's competence when it does
                          respond in format.
    A large gap between them is a formatting problem, not a knowledge problem, and the
    paper should say which it is rather than blending the two."""
    n = len(records)
    if n == 0:
        return {"n": 0}
    parsed = [r for r in records if r.get("predicted_index") is not None]
    correct = sum(1 for r in parsed if r["predicted_index"] == r["correct_index"])
    n_parsed = len(parsed)

    # Random-chance baseline varies per item because option counts vary (2-16 here), so
    # it is averaged per item rather than assumed to be 1/4.
    chance = sum(1.0 / max(len(r.get("options", [])) or 4, 1) for r in records) / n

    methods = {}
    for r in records:
        methods[r.get("parse_method", "unknown")] = methods.get(r.get("parse_method", "unknown"), 0) + 1

    return {
        "n": n,
        "n_parsed": n_parsed,
        "parse_rate": round(100 * n_parsed / n, 2),
        "n_correct": correct,
        "accuracy": round(100 * correct / n, 2),
        "accuracy_parsed": round(100 * correct / n_parsed, 2) if n_parsed else float("nan"),
        "chance_accuracy": round(100 * chance, 2),
        "parse_methods": methods,
    }


def position_bias(records: list, n_letters: int = 6) -> dict:
    """How often the model picks each letter, regardless of correctness.

    Sub-pillar 3b asks whether a model keys on option POSITION rather than content. If
    predictions pile onto "A" while the ground truth is spread evenly, that is positional
    bias - and it is invisible in an accuracy number alone."""
    counts = {LETTERS[i]: 0 for i in range(n_letters)}
    truth = {LETTERS[i]: 0 for i in range(n_letters)}
    for r in records:
        p, c = r.get("predicted_index"), r.get("correct_index")
        if p is not None and p < n_letters:
            counts[LETTERS[p]] += 1
        if c is not None and c < n_letters:
            truth[LETTERS[c]] += 1
    total = sum(counts.values()) or 1
    return {
        "predicted_pct": {k: round(100 * v / total, 1) for k, v in counts.items()},
        "truth_pct": {k: round(100 * v / (sum(truth.values()) or 1), 1) for k, v in truth.items()},
    }


if __name__ == "__main__":
    opts = ["Storiform pattern", "Necrosis", "Necrosis and hemorrhage", "Fibrosis"]
    cases = [
        ("A", 0, "bare_letter"), ("(C)", 2, "bare_letter"), ("**B**", 1, "bare_letter"),
        ("B) Necrosis", 1, "leading_option"),
        ("Looking at the slide... The answer is D.", 3, "stated_answer"),
        ("I considered B, but the answer is C", 2, "stated_answer"),
        ("Necrosis and hemorrhage", 2, "option_text"),   # must beat the shorter "Necrosis"
        ("The pattern here is storiform pattern.", 0, "option_text"),
        ("It is either B or D", None, "unparsed"),       # genuine ambiguity
        ("", None, "empty"),
    ]
    for text, expected, method in cases:
        idx, got = parse_choice(text, opts)
        status = "ok " if idx == expected and got == method else "FAIL"
        print(f"  {status} {text[:42]!r:46} -> {idx} via {got}")

    # Rule 6, against real LLaVA-Med replies that rules 1-5 could not parse.
    print("\n  token_overlap (real LLaVA-Med responses):")
    inflammation = [
        "The pattern of inflammation is necrotizing granulomatous inflammation",
        "The pattern of inflammation shows hyperplasia with emperipolesis",
        "The pattern of inflammation is neutrophilic suppurative inflammation",
        "The pattern of inflammation is non-necrotizing granulomatous inflammation",
        "The pattern of inflammation is lymphohistiocytic necrotizing inflammation",
    ]
    overlap_cases = [
        ("The pattern of inflammation depicted in this image is necrotizing "
         "granulomatous inflammation.", inflammation, 0, "token_overlap"),
        ("The most likely diagnosis given the histologic pattern from a dural mass in a "
         "teenager is classical Hodgkin lymphoma.",
         ["In that context, the most likely diagnosis is classical Hodgkin lymphoma",
          "In that context, the most likely diagnosis is yolk sac tumor",
          "In that context, the most likely diagnosis is embryonal carcinoma"],
         0, "token_overlap"),
        # Must NOT fire: the reply is generic and covers no option distinctively.
        ("This image shows an inflammatory process of some kind.",
         inflammation, None, "unparsed"),
    ]
    for text, options, expected, method in overlap_cases:
        idx, got = parse_choice(text, options)
        status = "ok " if idx == expected and got == method else "FAIL"
        print(f"  {status} {text[:42]!r:46} -> {idx} via {got}")
