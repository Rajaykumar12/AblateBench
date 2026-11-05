"""Generative math check (GSM8K-style). MMLU-Pro is scored by answer-letter
log-likelihood, which can hide damage to multi-step reasoning: an abliteration tool
comparison (arXiv 2512.13655) found MMLU moving <1 pp while GSM8K fell up to 18.8 pp.
This scores free-form generations instead: the model works the problem and ends with
"Answer: <number>", and the number is compared to the gold answer."""
from __future__ import annotations

import re

GSM8K_PROMPT = (
    "Solve the following math problem. Reason briefly step by step, then give the "
    "final answer on the last line in the form 'Answer: <number>'.\n\n{question}"
)

_ANSWER_RE = re.compile(r"answer\s*(?:is)?\s*[:=]?\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_answer(solution: str) -> float | None:
    """GSM8K solutions end with '#### <number>'."""
    if "####" not in solution:
        return None
    return _to_float(solution.split("####")[-1].strip())


def extract_answer(response: str) -> float | None:
    """Last 'Answer: N' in the reply, else the last number in it."""
    matches = _ANSWER_RE.findall(response)
    if matches:
        return _to_float(matches[-1])
    numbers = _NUMBER_RE.findall(response)
    return _to_float(numbers[-1]) if numbers else None


def is_correct(response: str, gold: float) -> bool:
    pred = extract_answer(response)
    return pred is not None and abs(pred - gold) < 1e-6 * max(1.0, abs(gold))


def score_responses(responses: list[str], golds: list[float]) -> dict:
    correct = [is_correct(r, g) for r, g in zip(responses, golds)]
    return {
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
        "num_questions": len(correct),
    }
