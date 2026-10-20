"""Stage [5b]: permanent weight-level ablation ("the real abliteration")."""
from __future__ import annotations

import torch
import torch.nn as nn

from model_utils import get_decoder_layers, get_residual_write_modules_with_post_norms


def post_norm_adjusted_direction(post_norm: nn.Module, d_hat: torch.Tensor) -> torch.Tensor:
    """Direction to orthogonalize a projection against when an RMSNorm sits between
    it and the residual add (Gemma 2/3 sandwich norms)."""
    hidden_dim = d_hat.shape[0]
    param = next(post_norm.parameters())
    ones = torch.ones(1, hidden_dim, dtype=torch.float32, device=param.device)
    g = post_norm(ones).float().squeeze(0)
    u = g * d_hat.to(device=g.device, dtype=torch.float32)
    norm = u.norm()
    return u / norm if norm > 0 else u


def orthogonalize_norm_preserving(
    weight: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project a 2-D matrix out of ``direction`` and restore every row norm.

    The projection is left-sided when ``direction`` matches the output dimension
    and right-sided when it matches the input dimension.  Calculations use
    float32 for stable rescaling and the original dtype is restored on return.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2-D weight, got shape {tuple(weight.shape)}")

    d = direction.to(device=weight.device, dtype=torch.float32).flatten()
    if d.numel() != weight.shape[0] and d.numel() != weight.shape[1]:
        raise ValueError(
            f"Direction length {d.numel()} does not match weight shape {tuple(weight.shape)}"
        )
    d = d / (d.norm() + eps)
    w = weight.float()
    if d.numel() == weight.shape[0]:
        original_norms = w.norm(dim=-1, keepdim=True)
        projected = w - torch.outer(d, d @ w)
    else:
        original_norms = w.norm(dim=-1, keepdim=True)
        projected = w - (w @ torch.outer(d, d))
    restored_norms = projected.norm(dim=-1, keepdim=True)
    return (projected * (original_norms / (restored_norms + eps))).to(weight.dtype)


def orthogonalize_matrix(
    weight: torch.Tensor,
    d_hat: torch.Tensor,
    preserve_norms: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return W' with the component along unit vector `d_hat` removed.

    Supports:
    - 2D Linear weights: shape [out_features (d_model), in_features]
    - 3D MoE weights: shape [n_experts, d_model, d_hidden] or [n_experts, d_hidden, d_model]

    When `preserve_norms=True` (grimjim formulation), rescales each output vector back
    to its original L2 norm to prevent residual stream magnitude decay across layers.
    """
    orig_dtype = weight.dtype
    w_fp32 = weight.float()
    d_fp32 = d_hat.to(device=weight.device, dtype=torch.float32)
    d_fp32 = d_fp32 / (d_fp32.norm() + eps)
    d_model = d_fp32.shape[0]

    if w_fp32.dim() == 2:
        # Standard nn.Linear: [d_model, in_features]
        orig_norms = w_fp32.norm(dim=-1, keepdim=True) if preserve_norms else None
        proj_row = d_fp32 @ w_fp32  # [in_features]
        w_new = w_fp32 - torch.outer(d_fp32, proj_row)
        if preserve_norms and orig_norms is not None:
            new_norms = w_new.norm(dim=-1, keepdim=True)
            w_new = w_new * (orig_norms / (new_norms + eps))
        return w_new.to(dtype=orig_dtype)

    if w_fp32.dim() == 3:
        proj = torch.outer(d_fp32, d_fp32)  # [d_model, d_model]
        if w_fp32.shape[1] == d_model:
            # Shape: [n_experts, d_model, in_features]
            orig_norms = w_fp32.norm(dim=-1, keepdim=True) if preserve_norms else None
            w_new = w_fp32 - torch.einsum("ij,ejk->eik", proj, w_fp32)
            if preserve_norms and orig_norms is not None:
                new_norms = w_new.norm(dim=-1, keepdim=True)
                w_new = w_new * (orig_norms / (new_norms + eps))
            return w_new.to(dtype=orig_dtype)
        elif w_fp32.shape[2] == d_model:
            # Shape: [n_experts, in_features, d_model] (HF packed MoE format)
            orig_norms = w_fp32.norm(dim=1, keepdim=True) if preserve_norms else None
            w_new = w_fp32 - torch.einsum("ekj,ji->eki", w_fp32, proj)
            if preserve_norms and orig_norms is not None:
                new_norms = w_new.norm(dim=1, keepdim=True)
                w_new = w_new * (orig_norms / (new_norms + eps))
            return w_new.to(dtype=orig_dtype)

    raise ValueError(
        f"Unsupported weight shape {tuple(weight.shape)} for d_hat of dim {d_model}"
    )


def orthogonalize_bias(bias: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    d_fp32 = d_hat.to(dtype=torch.float32, device=bias.device)
    b_fp32 = bias.float()
    coeff = d_fp32 @ b_fp32
    return (b_fp32 - coeff * d_fp32).to(dtype=bias.dtype)


@torch.no_grad()
def orthogonalize_embeddings_(model: nn.Module, d_hat: torch.Tensor, coefficient: float = 1.0) -> bool:
    """Remove `d_hat` from every token-embedding row in place. The embedding writes
    the residual stream's initial value, which layer 0's runtime hook also projects
    out, so the weight edit must cover it to match. Skipped (returns False) when the
    embedding is tied to the LM head, since editing it would also change the logits."""
    get_input = getattr(model, "get_input_embeddings", None)
    embedding = get_input() if callable(get_input) else None
    if embedding is None:
        return False
    get_output = getattr(model, "get_output_embeddings", None)
    lm_head = get_output() if callable(get_output) else None
    if lm_head is not None and lm_head.weight is embedding.weight:
        return False
    weight = embedding.weight.data
    d = d_hat.to(device=weight.device, dtype=torch.float32)
    d = d / d.norm()
    # Chunked: a float32 copy of a 250k-row table would cost GBs of extra VRAM.
    for chunk in weight.split(65536):
        rows = chunk.float()
        chunk.copy_((rows - coefficient * torch.outer(rows @ d, d)).to(weight.dtype))
    return True


@torch.no_grad()
def ablate_weights_(
    model: nn.Module,
    directions: torch.Tensor,
    coefficient: float = 1.0,
    layer_indices: list[int] | None = None,
    preserve_norms: bool = False,
) -> None:
    """In-place permanent ablation.

    Note: `preserve_norms` defaults to False so single-layer equivalence unit tests
    against `ablate_runtime` pass out of the box. Pass `preserve_norms=True` in
    full multi-layer runs to prevent capability degradation.
    """
    layers = get_decoder_layers(model)
    if directions.shape[0] != len(layers):
        raise ValueError(
            f"directions has {directions.shape[0]} rows but model has {len(layers)} layers"
        )
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    if 0 in layer_indices and torch.count_nonzero(directions[0]) > 0:
        orthogonalize_embeddings_(model, directions[0], coefficient)

    for idx in layer_indices:
        d_hat = directions[idx]
        if torch.count_nonzero(d_hat) == 0:
            continue
        for proj, post_norm in get_residual_write_modules_with_post_norms(layers[idx]):
            target = (
                d_hat
                if post_norm is None
                else post_norm_adjusted_direction(post_norm, d_hat)
            )
            if isinstance(proj, nn.Parameter):
                new_w = orthogonalize_matrix(proj.data, target, preserve_norms=preserve_norms)
                proj.data = (1 - coefficient) * proj.data + coefficient * new_w
            else:
                new_w = orthogonalize_matrix(
                    proj.weight.data, target, preserve_norms=preserve_norms
                )
                proj.weight.data = (
                    (1 - coefficient) * proj.weight.data + coefficient * new_w
                )
                if getattr(proj, "bias", None) is not None:
                    new_b = orthogonalize_bias(proj.bias.data, target)
                    proj.bias.data = (
                        (1 - coefficient) * proj.bias.data + coefficient * new_b
                    )