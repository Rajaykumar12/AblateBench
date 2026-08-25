"""Stage [5a]: reversible, hook-based runtime ablation.

Projects the refusal direction out of every token position's hidden state at every
(or a chosen subset of) decoder layers, at inference time only. No weight changes —
this is the fast-to-iterate version used during the layer/coefficient sweep, and the
correctness baseline that the permanent weight-level edit (ablate_weights.py) must
reproduce once its hook is removed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model_utils import get_decoder_layers, rewrap_hidden_state, unwrap_hidden_state


def project_out(h: torch.Tensor, d_hat: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    """h' = h - coefficient * (h . d_hat) d_hat, applied along the last dim.

    coefficient=1.0 is full ablation; coefficient<1.0 is the partial-strength variant
    used by the rescaling step (PRD stage [6]).
    """
    d_hat = d_hat.to(dtype=h.dtype, device=h.device)
    proj = (h @ d_hat).unsqueeze(-1) * d_hat  # [..., hidden]
    return h - coefficient * proj


def install_ablation_hooks(
    model: nn.Module,
    directions: torch.Tensor,
    coefficient: float = 1.0,
    layer_indices: list[int] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks that ablate `directions[i]` at layer i's output, for
    every layer in `layer_indices` (default: all layers with a non-degenerate
    direction). Applies to every token position, not just the last one, since the
    ablation must hold throughout generation, not only at the prompt's final token.

    Returns the hook handles; caller is responsible for calling `.remove()` on each
    (see `remove_hooks`) to restore the unmodified model.
    """
    layers = get_decoder_layers(model)
    if directions.shape[0] != len(layers):
        raise ValueError(
            f"directions has {directions.shape[0]} rows but model has {len(layers)} layers"
        )
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    def make_hook(d_hat: torch.Tensor):
        def hook(module, inputs, output):
            hidden = unwrap_hidden_state(output)
            new_hidden = project_out(hidden, d_hat, coefficient)
            return rewrap_hidden_state(output, new_hidden)

        return hook

    handles = []
    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(make_hook(directions[idx])))
    return handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()
