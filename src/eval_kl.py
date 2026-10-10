"""KL divergence between the base and ablated model on harmless prompts.

Accuracy benchmarks only register damage once it flips an answer; KL on the
next-token distribution registers any shift in what the model would say. On harmless
prompts - where the refusal direction shouldn't be active - a well-targeted ablation
should leave that distribution nearly unchanged, so KL(base || ablated) near 0 is the
signal that the edit removed refusal and little else.

Measured at the first response position (right after the chat template opens the
assistant turn), the same position extraction uses - it's where the model commits to
how it will answer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model_utils import format_chat_prompt


@torch.no_grad()
def first_token_log_probs(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> torch.Tensor:
    """[num_prompts, vocab] float32 log-probs of the first response token, on CPU.
    Requires left padding so position -1 is each prompt's real last token."""
    if tokenizer.padding_side != "left":
        raise ValueError("first_token_log_probs needs tokenizer.padding_side == 'left'")
    formatted = [format_chat_prompt(tokenizer, p) for p in prompts]
    rows = []
    for i in range(0, len(formatted), batch_size):
        enc = tokenizer(
            formatted[i : i + batch_size], return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(device)
        logits = model(**enc).logits[:, -1, :].float()
        rows.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.cat(rows, dim=0)


def mean_kl_divergence(base_log_probs: torch.Tensor, ablated_log_probs: torch.Tensor) -> float:
    """Mean over prompts of KL(base || ablated) = sum_v p_base(v) (log p_base - log p_abl),
    in nats. Direction: how surprised the ablated model is by what the base model
    would have said."""
    if base_log_probs.shape != ablated_log_probs.shape:
        raise ValueError(
            f"shape mismatch: {tuple(base_log_probs.shape)} vs {tuple(ablated_log_probs.shape)}"
        )
    per_prompt = (base_log_probs.exp() * (base_log_probs - ablated_log_probs)).sum(dim=-1)
    return per_prompt.mean().item()
