"""Stage [2]+[3]: activation extraction and per-layer refusal-direction computation.

Difference-in-means technique from Arditi et al., "Refusal in Language Models Is
Mediated by a Single Direction" (arXiv:2406.11717). For each decoder layer, the
direction is the (unit-normalized) difference between the mean last-token
residual-stream activation on harmful prompts and on harmless prompts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model_utils import format_chat_prompt, get_decoder_layers, unwrap_hidden_state


@dataclass
class ActivationSet:
    """Per-layer last-token residual-stream activations for one prompt set.

    activations: tensor of shape [num_layers, num_prompts, hidden_dim].
    """

    activations: torch.Tensor

    @property
    def num_layers(self) -> int:
        return self.activations.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.activations.shape[2]


@torch.no_grad()
def extract_activations(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> ActivationSet:
    """Run a forward pass per prompt (batched), hook every decoder layer, and capture
    the last-token residual-stream activation at each layer.

    Single forward pass per prompt, no backward pass — this is what keeps the whole
    pipeline cheap even at 30B+ scale (see PRD §6.2, Compute placement).
    """
    layers = get_decoder_layers(model)
    num_layers = len(layers)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            hidden = unwrap_hidden_state(output)  # [batch, seq, hidden]
            captured[layer_idx] = hidden[:, -1, :].detach().to("cpu", dtype=torch.float32)

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    formatted_prompts = [format_chat_prompt(tokenizer, p) for p in prompts]

    all_layer_acts: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    try:
        for start in range(0, len(formatted_prompts), batch_size):
            batch = formatted_prompts[start : start + batch_size]
            # add_special_tokens=False: the chat template's output already contains
            # BOS/header tokens - tokenizing again with defaults would double them.
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
            ).to(device)
            model(**enc)
            for layer_idx in range(num_layers):
                all_layer_acts[layer_idx].append(captured[layer_idx])
    finally:
        for h in handles:
            h.remove()

    # [num_layers, num_prompts, hidden_dim]
    stacked = torch.stack([torch.cat(acts, dim=0) for acts in all_layer_acts], dim=0)
    return ActivationSet(activations=stacked)


def _per_layer_means(harmful: ActivationSet, harmless: ActivationSet) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared reduction used by compute_directions, purify_direction, and
    layer_signal_quality: mean last-token activation per layer, per class.

    Returns (harmful_mean, harmless_mean), each [num_layers, hidden_dim].
    """
    if harmful.num_layers != harmless.num_layers:
        raise ValueError(
            f"Layer count mismatch: harmful has {harmful.num_layers}, "
            f"harmless has {harmless.num_layers}"
        )
    return harmful.activations.mean(dim=1), harmless.activations.mean(dim=1)


def _unit_normalize(v: torch.Tensor) -> torch.Tensor:
    """Normalize the last dim to unit length; degenerate (zero) rows stay zero rather
    than producing NaNs, matching compute_directions's existing convention."""
    norms = v.norm(dim=-1, keepdim=True)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    return v / safe_norms


def _project_out(v: torch.Tensor, hat: torch.Tensor) -> torch.Tensor:
    """v with its component along unit vector `hat` removed, magnitude NOT
    renormalized afterward — used both by purify_direction (which does renormalize,
    at the end) and by layer_signal_quality's purity metric (which needs the
    un-renormalized residual norm to measure how much signal survived)."""
    proj = (v * hat).sum(dim=-1, keepdim=True)
    return v - proj * hat


def purify_direction(refusal_dir: torch.Tensor, harmless_mean: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt-orthogonalize `refusal_dir` against the (normalized) harmless-mean
    direction: `refined = refusal_dir - (refusal_dir . harmless_hat) * harmless_hat`.

    Strips out whatever component of the harmful-vs-harmless difference is really just
    "a prompt was given to me at all" / generic topic signal, rather than
    refusal-specific signal. Re-normalized to unit length afterward (or the zero
    vector, if the projection removes it entirely — same degenerate-layer convention
    as compute_directions).

    Both inputs share their last dim (hidden_dim); works for a single layer
    ([hidden_dim]) or a stack of layers ([num_layers, hidden_dim]).
    """
    harmless_hat = _unit_normalize(harmless_mean)
    return _unit_normalize(_project_out(refusal_dir, harmless_hat))


def compute_directions(
    harmful: ActivationSet, harmless: ActivationSet, purified: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-layer difference-in-means direction, unit-normalized.

    Returns tensor of shape [num_layers, hidden_dim], each row a unit vector (or the
    zero vector, in the degenerate case both classes' means coincide exactly).

    If `purified=True`, also Gram-Schmidt-orthogonalizes the direction against the
    harmless-mean direction (see `purify_direction`) and returns `(raw, purified)` so
    callers can save/sweep both instead of only the purified variant.
    """
    harmful_mean, harmless_mean = _per_layer_means(harmful, harmless)
    diff = harmful_mean - harmless_mean  # [layers, hidden]
    raw = _unit_normalize(diff)
    if not purified:
        return raw
    purified_dir = purify_direction(raw, harmless_mean)
    return raw, purified_dir


def layer_signal_quality(harmful: ActivationSet, harmless: ActivationSet) -> torch.Tensor:
    """Per-layer signal-quality score, cheap and purely arithmetic over the same
    per-layer means compute_directions already reduces to — no extra model forward
    passes needed.

    score = SNR * (1 - cosine(harmful_mean, harmless_mean)) * purity, where:
      - SNR = ||refusal_dir|| / max(||harmful_mean||, ||harmless_mean||) — uses the raw
        (pre-normalization) difference norm, since compute_directions's unit-normalized
        output has already discarded magnitude.
      - purity = the fraction of the unit-normalized refusal direction's norm that
        survives Gram-Schmidt against the harmless direction (the un-renormalized
        residual's norm — purify_direction's own output is renormalized back to unit
        length and so can't be used to measure how much signal was removed).

    Returns tensor of shape [num_layers]. Higher is better: a layer where the
    harmful/harmless means are far apart, mostly non-parallel, and mostly refusal-
    specific (not absorbed by generic harmless-direction signal) scores highest.
    """
    harmful_mean, harmless_mean = _per_layer_means(harmful, harmless)
    diff = harmful_mean - harmless_mean
    diff_norm = diff.norm(dim=-1)
    denom = torch.maximum(harmful_mean.norm(dim=-1), harmless_mean.norm(dim=-1))
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    snr = diff_norm / safe_denom

    cos = torch.nn.functional.cosine_similarity(harmful_mean, harmless_mean, dim=-1)

    # purity: fraction of the (unit-normalized) refusal direction's norm that survives
    # Gram-Schmidt against harmless — NOT purify_direction's output, which is itself
    # renormalized back to unit length and so can't be used to measure how much was
    # removed. Uses the un-renormalized residual from _project_out instead.
    refusal_dir = _unit_normalize(diff)
    harmless_hat = _unit_normalize(harmless_mean)
    residual = _project_out(refusal_dir, harmless_hat)
    purity = residual.norm(dim=-1)  # refusal_dir has norm 1, so this is already a fraction

    return snr * (1 - cos) * purity
