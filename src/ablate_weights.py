"""Stage [5b]: permanent weight-level ablation ("the real abliteration").

Directions may be a single unit vector per layer ([hidden]) or a small orthonormal
refusal subspace per layer ([k, hidden]); both are handled through the projector
P = Q^T Q, so rank 1 is just the k = 1 case.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model_utils import get_decoder_layers, get_residual_writes_by_kind


def orthonormal_basis(directions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """[hidden] or [k, hidden] -> [k', hidden] with orthonormal rows, in float32.
    Order is kept (row 0 stays the primary direction, up to sign); near-zero rows
    left after Gram-Schmidt are dropped."""
    q = directions.float()
    if q.dim() == 1:
        q = q.unsqueeze(0)
    rows: list[torch.Tensor] = []
    for v in q:
        for r in rows:
            v = v - (v @ r) * r
        n = v.norm()
        if n > eps:
            rows.append(v / n)
    if not rows:
        return torch.zeros(0, q.shape[-1], dtype=torch.float32, device=q.device)
    return torch.stack(rows)


def post_norm_adjusted_direction(post_norm: nn.Module, d_hat: torch.Tensor) -> torch.Tensor:
    """Direction(s) to orthogonalize a projection against when an RMSNorm sits
    between it and the residual add (Gemma 2/3 sandwich norms)."""
    hidden_dim = d_hat.shape[-1]
    param = next(post_norm.parameters())
    ones = torch.ones(1, hidden_dim, dtype=torch.float32, device=param.device)
    g = post_norm(ones).float().squeeze(0)
    u = g * d_hat.to(device=g.device, dtype=torch.float32)
    if u.dim() == 2:
        return orthonormal_basis(u)
    norm = u.norm()
    return u / norm if norm > 0 else u


def orthogonalize_matrix(
    weight: torch.Tensor,
    d_hat: torch.Tensor,
    preserve_norms: bool = False,
    eps: float = 1e-8,
    strength: float = 1.0,
    row_strengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return W' = W - strength * P W, with P the projector onto `d_hat`'s span
    (a unit vector [d_model] or orthonormal rows [k, d_model]).

    Supports:
    - 2D Linear weights: shape [out_features (d_model), in_features]
    - 3D MoE weights: shape [n_experts, d_model, d_hidden] or [n_experts, d_hidden, d_model]

    When `preserve_norms=True` (grimjim formulation), rescales each *input* column
    (the d_model-sized vector one input feature writes into the residual stream) back
    to its original L2 norm. Scaling columns keeps d_hat^T W' = 0. Scaling the
    d_model-indexed rows instead, as this function used to, multiplies each output
    coordinate by a different factor and re-introduces part of d_hat into every
    output, so the "ablated" weights still write the refusal direction.

    `row_strengths` ([k]): optional per-direction strength, so the projector becomes
    Q^T diag(row_strengths) Q. `d_hat` must then already be orthonormal rows.
    """
    orig_dtype = weight.dtype
    w = weight.float()
    q = orthonormal_basis(d_hat.to(device=weight.device))
    d_model = q.shape[-1]
    proj = _projector(q, row_strengths, d_hat)

    if w.dim() == 2 and w.shape[0] == d_model:
        out_axis = 0
        w_new = w - strength * (proj @ w)
    elif w.dim() == 3 and w.shape[1] == d_model:
        out_axis = 1  # [n_experts, d_model, in_features]
        w_new = w - strength * torch.einsum("ij,ejk->eik", proj, w)
    elif w.dim() == 3 and w.shape[2] == d_model:
        out_axis = 2  # [n_experts, in_features, d_model] (HF packed MoE format)
        w_new = w - strength * torch.einsum("ekj,ji->eki", w, proj)
    else:
        raise ValueError(
            f"Unsupported weight shape {tuple(weight.shape)} for d_hat of dim {d_model}"
        )
    if preserve_norms:
        orig = w.norm(dim=out_axis, keepdim=True)
        new = w_new.norm(dim=out_axis, keepdim=True)
        w_new = w_new * (orig / (new + eps))
    return w_new.to(dtype=orig_dtype)


def _projector(q: torch.Tensor, row_strengths, d_hat: torch.Tensor) -> torch.Tensor:
    if row_strengths is None:
        return q.T @ q
    rs = torch.as_tensor(row_strengths, dtype=torch.float32, device=q.device).flatten()
    expected = 1 if d_hat.dim() == 1 else d_hat.shape[0]
    if q.shape[0] != expected or rs.numel() != q.shape[0]:
        raise ValueError("row_strengths needs orthonormal, non-degenerate rows, one strength each")
    return q.T @ (rs.unsqueeze(1) * q)


def orthogonalize_bias(
    bias: torch.Tensor, d_hat: torch.Tensor, strength: float = 1.0, row_strengths=None
) -> torch.Tensor:
    q = orthonormal_basis(d_hat.to(device=bias.device))
    b = bias.float()
    return (b - strength * (_projector(q, row_strengths, d_hat) @ b)).to(dtype=bias.dtype)


@torch.no_grad()
def orthogonalize_embeddings_(
    model: nn.Module, d_hat: torch.Tensor, coefficient: float = 1.0, row_strengths=None
) -> bool:
    """Remove `d_hat`'s span from every token-embedding row in place. The embedding
    writes the residual stream's initial value, which layer 0's runtime hook also
    projects out, so the weight edit must cover it to match. Skipped (returns False)
    when the embedding is tied to the LM head, since editing it would also change the
    logits."""
    get_input = getattr(model, "get_input_embeddings", None)
    embedding = get_input() if callable(get_input) else None
    if embedding is None:
        return False
    get_output = getattr(model, "get_output_embeddings", None)
    lm_head = get_output() if callable(get_output) else None
    if lm_head is not None and lm_head.weight is embedding.weight:
        return False
    weight = embedding.weight.data
    q = orthonormal_basis(d_hat.to(device=weight.device))
    proj = _projector(q, row_strengths, d_hat)
    # Chunked: a float32 copy of a 250k-row table would cost GBs of extra VRAM.
    for chunk in weight.split(65536):
        rows = chunk.float()
        chunk.copy_((rows - coefficient * rows @ proj).to(weight.dtype))
    return True


def _per_layer(values, n: int, default: float) -> list[float]:
    if values is None:
        return [default] * n
    values = [float(v) for v in values]
    if len(values) != n:
        raise ValueError(f"expected {n} per-layer weights, got {len(values)}")
    return values


@torch.no_grad()
def ablate_weights_(
    model: nn.Module,
    directions: torch.Tensor,
    coefficient: float = 1.0,
    layer_indices: list[int] | None = None,
    preserve_norms: bool = False,
    attn_weights: list[float] | None = None,
    mlp_weights: list[float] | None = None,
    embed_weight: float | None = None,
    row_strengths: torch.Tensor | None = None,
) -> None:
    """In-place permanent ablation.

    `directions`: [num_layers, hidden] or [num_layers, k, hidden] (refusal subspace).
    `attn_weights` / `mlp_weights`: optional per-layer strengths for attention vs
    MLP/expert writes (the optimizer's weight kernels); both default to
    `coefficient`. `embed_weight` defaults to `coefficient` too.
    `row_strengths` ([k]): per-direction strengths within a [L, k, hidden] subspace
    (rows must be orthonormal); the layer strengths multiply them.

    Note: `preserve_norms` defaults to False so single-layer equivalence unit tests
    against `ablate_runtime` pass out of the box.
    """
    layers = get_decoder_layers(model)
    n = len(layers)
    if directions.shape[0] != n:
        raise ValueError(
            f"directions has {directions.shape[0]} rows but model has {n} layers"
        )
    if layer_indices is None:
        layer_indices = list(range(n))
    attn_w = _per_layer(attn_weights, n, coefficient)
    mlp_w = _per_layer(mlp_weights, n, coefficient)
    embed_w = coefficient if embed_weight is None else embed_weight

    if 0 in layer_indices and torch.count_nonzero(directions[0]) > 0 and embed_w != 0:
        orthogonalize_embeddings_(model, directions[0], embed_w, row_strengths)

    for idx in layer_indices:
        d_hat = directions[idx]
        if torch.count_nonzero(d_hat) == 0:
            continue
        for proj, post_norm, kind in get_residual_writes_by_kind(layers[idx]):
            strength = attn_w[idx] if kind == "attn" else mlp_w[idx]
            if strength == 0:
                continue
            if post_norm is not None and row_strengths is not None:
                raise NotImplementedError("per-direction strengths with sandwich post-norms")
            target = (
                d_hat
                if post_norm is None
                else post_norm_adjusted_direction(post_norm, d_hat)
            )
            kw = dict(preserve_norms=preserve_norms, strength=strength, row_strengths=row_strengths)
            if isinstance(proj, nn.Parameter):
                proj.data = orthogonalize_matrix(proj.data, target, **kw)
            else:
                proj.weight.data = orthogonalize_matrix(proj.weight.data, target, **kw)
                if getattr(proj, "bias", None) is not None:
                    proj.bias.data = orthogonalize_bias(
                        proj.bias.data, target, strength, row_strengths
                    )
