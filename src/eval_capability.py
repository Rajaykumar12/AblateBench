"""Stage [8]: capability-eval runner over a small multiple-choice benchmark subset.

Scored via log-likelihood comparison across answer options (the standard MMLU/ARC-easy
scoring method), not free-form generation + string matching, so results are
deterministic and cheap to rerun across every ablation variant/coefficient (PRD §6.1,
requirement 9). 50-100 questions per PRD §6.1 — small enough to be a repeatable sanity
check, not a leaderboard-grade run (PRD §4, non-goals).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MCQuestion:
    question: str
    choices: list[str]  # e.g. ["Paris", "London", "Berlin", "Madrid"]
    answer_index: int


# Up to 10 options: MMLU-Pro questions have as many as 10 (A-J); ARC has 3-5.
_LETTERS = "ABCDEFGHIJ"


def _format_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n{options}\nAnswer:"


@torch.no_grad()
def score_choice_loglikelihoods(
    model: nn.Module,
    tokenizer,
    prompt: str,
    choices: list[str],
    device: str = "cuda",
) -> list[float]:
    """Return the total log-likelihood the model assigns to each choice's answer
    letter, conditioned on the formatted prompt. Higher is more preferred.

    When every answer letter is a single token (true for common BPE vocabularies),
    one forward pass over the prompt gives all the scores at once: the log-prob of
    letter token t after the prompt is exactly what the per-choice path computes for
    a 1-token continuation. That's ~N x cheaper for N options - it matters for
    10-option MMLU-Pro on a 27B model. Falls back to one pass per choice otherwise."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    letter_ids = [
        tokenizer(f" {_LETTERS[i]}", return_tensors="pt", add_special_tokens=False)
        .to(device)["input_ids"]
        for i in range(len(choices))
    ]

    if all(ids.shape[1] == 1 for ids in letter_ids):
        out = model(input_ids=prompt_ids["input_ids"],
                    attention_mask=torch.ones_like(prompt_ids["input_ids"]))
        last_log_probs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        return [last_log_probs[ids[0, 0]].item() for ids in letter_ids]

    scores = []
    for ids in letter_ids:
        full_ids = torch.cat([prompt_ids["input_ids"], ids], dim=1)
        attention_mask = torch.ones_like(full_ids)
        out = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = out.logits[0, :-1, :]  # predict each next token from the previous one
        targets = full_ids[0, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        n_letter_tokens = ids.shape[1]
        # sum log-prob over just the answer-letter tokens at the end of the sequence
        token_log_probs = log_probs[torch.arange(logits.shape[0]), targets]
        scores.append(token_log_probs[-n_letter_tokens:].sum().item())
    return scores


@torch.no_grad()
def score_questions(
    model: nn.Module,
    tokenizer,
    questions: list[MCQuestion],
    device: str = "cuda",
) -> list[bool]:
    """Per-question correctness (model's highest-log-likelihood letter == answer),
    in input order - lets callers break accuracy down by category."""
    results = []
    for q in questions:
        prompt = _format_prompt(q.question, q.choices)
        scores = score_choice_loglikelihoods(model, tokenizer, prompt, q.choices, device)
        predicted = max(range(len(scores)), key=lambda i: scores[i])
        results.append(predicted == q.answer_index)
    return results


@torch.no_grad()
def evaluate_capability(
    model: nn.Module,
    tokenizer,
    questions: list[MCQuestion],
    device: str = "cuda",
) -> float:
    """Accuracy over the given multiple-choice subset: fraction of questions where the
    model assigns the highest log-likelihood to the correct answer letter."""
    results = score_questions(model, tokenizer, questions, device)
    return sum(results) / len(results) if results else 0.0
