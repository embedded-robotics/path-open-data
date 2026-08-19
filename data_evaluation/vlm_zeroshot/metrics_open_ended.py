"""Metrics for the open-ended half of sub-pillar 4a.

The paper specifies "exact-match, BLEU, and BERTScore for open-ended". Each answers a
different question, which is why all three are reported rather than one blended number:

    exact_match   did the model produce the reference string? Nearly always 0 on
                  free-text pathology answers - a 20-word reference is not going to be
                  reproduced verbatim - but it is the honest floor, and it is the right
                  metric for CLOSE-ended yes/no items.
    BLEU          n-gram overlap. Rewards reusing the reference's wording. Punishing on
                  short answers and blind to synonyms ("neoplasm" vs "tumour").
    BERTScore     embedding similarity. Credits paraphrase, which matters here because a
                  pathologist's answer and a model's can be clinically identical and
                  lexically disjoint. Reported BOTH raw and baseline-rescaled: raw is what
                  the literature quotes, but its floor is ~85 for any fluent same-domain
                  text, so differences look artificially small. Rescaled subtracts that
                  floor, giving a scale where 0 = no better than unrelated text.

## Two things the paper's own reviewer-concerns section demands

"Cross-format metric comparison must be interpreted WITHIN-format, not as one blended
score" - so nothing here averages BLEU against accuracy.

Close-ended answers are scored by yes/no match, NOT by BLEU. A one-word "yes" scored with
a paragraph-similarity metric is meaningless, and PathVQA's close-ended subset would
otherwise report near-zero BLEU while being answered perfectly.

## BERTScore is deliberately deferred

It loads a RoBERTa model and is ~1000x slower per pair than BLEU, so it is computed once
over the whole corpus in `score_open_ended_corpus`, not per item during generation. The
generation run stores raw text; metrics are a separate, re-runnable pass over the
checkpoint - the same split the judge pipeline uses.
"""
from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation/extra whitespace.

    The SQuAD convention. Without it exact-match reports 0 for answers differing only by
    "the", which measures formatting rather than correctness."""
    text = str(text).lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 - the standard partial-credit companion to exact match.

    Not in the paper's list, but it costs nothing and is far more informative than exact
    match on long answers, where EM is almost always 0 and therefore ranks nothing."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = {}
    for token in pred_tokens:
        if token in ref_tokens:
            common[token] = min(pred_tokens.count(token), ref_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_yes_no(text: str) -> str | None:
    """First yes/no token in a reply, or None.

    Close-ended models rarely answer with a bare "yes" - they say "Yes, the lesion shows
    ...". Taking the FIRST verdict matches how ../vlm/quiltvqa_eval_runner.ipynb splits
    Quilt-VQA's close-ended answers, so both pipelines treat these items identically."""
    match = _YES_NO.search(str(text))
    return match.group(1).lower() if match else None


def score_close_ended(prediction: str, reference: str) -> dict:
    """Yes/no accuracy. `parsed` is False when the model gave no verdict at all -
    reported separately so a refusal is never silently counted as a wrong answer."""
    predicted = extract_yes_no(prediction)
    expected = extract_yes_no(reference)
    return {"parsed": predicted is not None,
            "correct": float(predicted is not None and predicted == expected)}


def sentence_bleu(prediction: str, reference: str) -> float:
    """Single-pair BLEU (sacrebleu, 0-100).

    Sentence-level BLEU is noisy by design; the corpus figure below is the one to report.
    This exists so per-item scores can be inspected when a number looks wrong."""
    import sacrebleu
    return sacrebleu.sentence_bleu(str(prediction), [str(reference)]).score


def score_open_ended_corpus(predictions: list, references: list,
                            compute_bertscore: bool = True,
                            bertscore_model: str = "roberta-large",
                            device: str = "cuda:0", batch_size: int = 64,
                            rescale_bertscore: bool = True) -> dict:
    """Corpus-level metrics over aligned prediction/reference lists.

    BLEU is computed at CORPUS level, not averaged over sentences: BLEU's brevity penalty
    and n-gram precisions are defined over a corpus, and averaging sentence scores gives a
    systematically different (and lower) number that is not comparable with published
    BLEU figures."""
    import sacrebleu

    predictions = [str(p) for p in predictions]
    references = [str(r) for r in references]
    if not predictions:
        return {"n": 0}

    result = {
        "n": len(predictions),
        "exact_match": round(100 * sum(map(exact_match, predictions, references)) / len(predictions), 2),
        "token_f1": round(100 * sum(map(token_f1, predictions, references)) / len(predictions), 2),
        "bleu": round(sacrebleu.corpus_bleu(predictions, [references]).score, 2),
        "chrf": round(sacrebleu.corpus_chrf(predictions, [references]).score, 2),
        "mean_pred_words": round(sum(len(p.split()) for p in predictions) / len(predictions), 1),
        "mean_ref_words": round(sum(len(r.split()) for r in references) / len(references), 1),
    }

    if compute_bertscore:
        from bert_score import score as bert_score_fn
        # RAW: comparable with published BERTScore figures, which almost always report
        # the unrescaled number. But it has a high, uninformative floor - any two fluent
        # English sentences on the same topic score ~85, so a 85-vs-88 spread looks tiny
        # when the underlying difference is not.
        _, _, raw_f1 = bert_score_fn(predictions, references, model_type=bertscore_model,
                                     lang="en", device=device, batch_size=batch_size,
                                     verbose=False, rescale_with_baseline=False)
        result["bertscore_f1"] = round(100 * float(raw_f1.mean()), 2)

        if rescale_bertscore:
            # RESCALED: bert-score subtracts an empirical baseline computed on random
            # sentence pairings, so 0 means "no better than unrelated text" and 1 means
            # "identical". That is the scale a reader intuitively assumes, and it is what
            # makes a figure legible - the raw 85-88 band expands into a real spread.
            # Both are reported because they answer different questions; neither alone
            # is honest here.
            _, _, rescaled_f1 = bert_score_fn(
                predictions, references, model_type=bertscore_model, lang="en",
                device=device, batch_size=batch_size, verbose=False,
                rescale_with_baseline=True)
            result["bertscore_f1_rescaled"] = round(100 * float(rescaled_f1.mean()), 2)
    return result


if __name__ == "__main__":
    reference = "The abnormal cells show round large cell morphology with prominent nucleoli"
    cases = [
        ("The abnormal cells show round large cell morphology with prominent nucleoli", "identical"),
        ("the abnormal cells show round, large cell morphology with prominent nucleoli.", "punct/case only"),
        ("Round large cells with prominent nucleoli are seen", "paraphrase"),
        ("Normal squamous epithelium", "wrong"),
    ]
    print(f"{'case':18s} {'EM':>5s} {'F1':>6s} {'BLEU':>7s}")
    for prediction, label in cases:
        print(f"{label:18s} {exact_match(prediction, reference):5.0f} "
              f"{100 * token_f1(prediction, reference):6.1f} "
              f"{sentence_bleu(prediction, reference):7.1f}")

    print("\nclose-ended:")
    for prediction, reference_yn, label in [
        ("Yes", "yes", "bare"),
        ("Yes, the lesion shows necrosis.", "yes", "verdict + prose"),
        ("No, this is benign.", "yes", "wrong verdict"),
        ("The image is unclear.", "yes", "no verdict"),
    ]:
        print(f"  {label:18s} {score_close_ended(prediction, reference_yn)}")
