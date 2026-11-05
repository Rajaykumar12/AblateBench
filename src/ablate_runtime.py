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

from model_utils import (
    get_attention_module,
    get_decoder_layers,
    has_sandwich_norms,
    rewrap_hidden_state,
    unwrap_hidden_state,
)


def project_out(
    h: torch.Tensor, d_hat: torch.Tensor, coefficient: float = 1.0, row_strengths=None
) -> torch.Tensor:
    """h' = h - coefficient * (h . d_hat) d_hat, applied along the last dim.

    coefficient=1.0 is full ablation; coefficient<1.0 is the partial-strength variant
    used by the rescaling step (PRD stage [6]).
    """
    d_hat = d_hat.to(dtype=h.dtype, device=h.device)
    if d_hat.dim() == 2:
        # [k, hidden] orthonormal refusal subspace, optionally a strength per row
        coords = h @ d_hat.T
        if row_strengths is not None:
            coords = coords * torch.as_tensor(row_strengths, dtype=h.dtype, device=h.device)
        return h - coefficient * coords @ d_hat
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


def install_component_hooks(
    model: nn.Module,
    directions: torch.Tensor,
    attn_weights: list[float],
    mlp_weights: list[float],
    embed_weight: float = 0.0,
    row_strengths=None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Reversible twin of ablate_weights_(..., attn_weights, mlp_weights,
    embed_weight, preserve_norms=False): project `directions[i]` out of layer i's
    attention-block output and MLP/MoE-block output with separate strengths, and
    out of the token embeddings. Each block's last op is its residual-write
    projection (o_proj / out_proj / down_proj; MoE gates are scalars), so projecting
    the block output equals editing those weights. Used by the optimizer, which
    needs per-component strengths that whole-layer hooks can't express.

    Directions must already be orthonormal per layer ([L, hidden] unit rows or
    [L, k, hidden] orthonormal rows). Sandwich-norm models (Gemma 2/3) are rejected:
    there the post-norm sits between the block output and the residual add."""
    layers = get_decoder_layers(model)
    if directions.shape[0] != len(layers):
        raise ValueError(
            f"directions has {directions.shape[0]} rows but model has {len(layers)} layers"
        )
    if any(has_sandwich_norms(layer) for layer in layers):
        raise NotImplementedError("component hooks do not model sandwich post-norms")

    def make_hook(d_hat: torch.Tensor, strength: float):
        def hook(module, inputs, output):
            hidden = unwrap_hidden_state(output)
            return rewrap_hidden_state(
                output, project_out(hidden, d_hat, strength, row_strengths)
            )

        return hook

    handles = []
    if embed_weight:
        emb = model.get_input_embeddings()
        handles.append(emb.register_forward_hook(make_hook(directions[0], embed_weight)))
    for i, layer in enumerate(layers):
        attn = get_attention_module(layer)
        if attn is not None and attn_weights[i]:
            handles.append(attn.register_forward_hook(make_hook(directions[i], attn_weights[i])))
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and mlp_weights[i]:
            handles.append(mlp.register_forward_hook(make_hook(directions[i], mlp_weights[i])))
    return handles
