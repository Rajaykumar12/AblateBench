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


def _format_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCDEFGH"
    options = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
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
    letter, conditioned on the formatted prompt. Higher is more preferred."""
    letters = "ABCDEFGH"
    scores = []
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
    for i in range(len(choices)):
        letter = f" {letters[i]}"
        letter_ids = tokenizer(letter, return_tensors="pt", add_special_tokens=False).to(device)
        full_ids = torch.cat([prompt_ids["input_ids"], letter_ids["input_ids"]], dim=1)
        attention_mask = torch.ones_like(full_ids)
        out = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = out.logits[0, :-1, :]  # predict each next token from the previous one
        targets = full_ids[0, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        n_letter_tokens = letter_ids["input_ids"].shape[1]
        # sum log-prob over just the answer-letter tokens at the end of the sequence
        token_log_probs = log_probs[torch.arange(logits.shape[0]), targets]
        scores.append(token_log_probs[-n_letter_tokens:].sum().item())
    return scores


@torch.no_grad()
def evaluate_capability(
    model: nn.Module,
    tokenizer,
    questions: list[MCQuestion],
    device: str = "cuda",
) -> float:
    """Accuracy over the given multiple-choice subset: fraction of questions where the
    model assigns the highest log-likelihood to the correct answer letter."""
    correct = 0
    for q in questions:
        prompt = _format_prompt(q.question, q.choices)
        scores = score_choice_loglikelihoods(model, tokenizer, prompt, q.choices, device)
        predicted = max(range(len(scores)), key=lambda i: scores[i])
        if predicted == q.answer_index:
            correct += 1
    return correct / len(questions) if questions else 0.0
