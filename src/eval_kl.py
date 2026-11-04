"""KL divergence between the base and ablated model on harmless prompts."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            formatted[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        logits = model(**enc).logits[:, -1, :].float()
        rows.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.cat(rows, dim=0)


def mean_kl_divergence(base_log_probs: torch.Tensor, ablated_log_probs: torch.Tensor) -> float:
    """Mean over prompts of KL(base || ablated) = sum_v p_base(v) (log p_base - log p_abl),
    in nats."""
    if base_log_probs.shape != ablated_log_probs.shape:
        raise ValueError(
            f"shape mismatch: {tuple(base_log_probs.shape)} vs {tuple(ablated_log_probs.shape)}"
        )
    per_prompt = (base_log_probs.exp() * (base_log_probs - ablated_log_probs)).sum(dim=-1)
    return per_prompt.mean().item()


@torch.no_grad()
def sequence_teacher_forced_kl(
    base_model: nn.Module,
    ablated_model: nn.Module,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
    max_new_tokens: int = 100,
    batch_size: int = 4,
) -> float:
    """Strict multi-token teacher-forced KL(base || ablated) across generated responses."""
    if tokenizer.padding_side != "left":
        raise ValueError("sequence_teacher_forced_kl needs tokenizer.padding_side == 'left'")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    formatted = [format_chat_prompt(tokenizer, p) for p in prompts]

    total_kl = 0.0
    total_tokens = 0

    for i in range(0, len(formatted), batch_size):
        enc = tokenizer(
            formatted[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        prompt_len = enc.input_ids.shape[1]
        gen_tokens = base_model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

        gen_len = gen_tokens.shape[1] - prompt_len
        if gen_len <= 0:
            continue

        # Build mask for generated tokens: valid up to and including the first EOS
        gen_slice = gen_tokens[:, prompt_len:]
        if eos_id is not None:
            is_eos = (gen_slice == eos_id).int()
            # Cumulative sum > 1 means tokens *after* the first EOS
            eos_cumsum = is_eos.cumsum(dim=1)
            gen_mask = ((eos_cumsum == 0) | ((eos_cumsum == 1) & (is_eos == 1))).float()
        else:
            gen_mask = torch.ones_like(gen_slice, dtype=torch.float32)

        full_attention_mask = torch.cat([enc.attention_mask.float(), gen_mask], dim=1).long()

        base_logits = base_model(gen_tokens, attention_mask=full_attention_mask).logits
        abl_logits = ablated_model(gen_tokens, attention_mask=full_attention_mask).logits

        # Logits at [prompt_len - 1 : -1] predict tokens at [prompt_len :]
        b_logits = base_logits[:, prompt_len - 1 : -1, :].float()
        a_logits = abl_logits[:, prompt_len - 1 : -1, :].float()

        log_p_base = F.log_softmax(b_logits, dim=-1)
        log_p_abl = F.log_softmax(a_logits, dim=-1)
        p_base = log_p_base.exp()

        token_kl = (p_base * (log_p_base - log_p_abl)).sum(dim=-1)  # [batch, gen_len]
        total_kl += (token_kl * gen_mask).sum().item()
        total_tokens += gen_mask.sum().item()

    return total_kl / max(total_tokens, 1.0)