"""Stage [5b]: permanent weight-level ablation ("the real abliteration").

Orthogonalizes each residual-stream-writing weight matrix (attention output
projection, MLP down projection — see model_utils._RESIDUAL_WRITE_PROJECTIONS) with
respect to the refusal direction, so the direction can no longer be produced by that
layer at all — no runtime hook required afterward. This is what makes the result
deployable as a plain checkpoint (PRD §6.2, "Portability of the permanent edit").

Math: for a Linear layer computing `y = W x + b`, the component of `y` along a unit
vector `d_hat` is `d_hat . y = (d_hat^T W) x + d_hat . b`. Zeroing that identically for
all `x` requires `d_hat^T W = 0`, which is achieved exactly by:

    W' = W - d_hat (d_hat^T W)      (equivalently: W' = (I - d_hat d_hat^T) W)
    b' = b - (d_hat . b) d_hat

Applying this to every projection that writes into the residual stream at a layer
means that layer can no longer inject any component along d_hat into the stream,
regardless of input — a stronger and more direct claim than "empirically, refusal
rate drops," and the reason ablate_weights must reproduce ablate_runtime's
refusal-rate drop with the hook removed (PRD §5, row 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model_utils import get_decoder_layers, get_residual_write_modules


def orthogonalize_matrix(weight: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    """Return W' = W - d_hat (d_hat^T W), so that d_hat^T W' == 0 (row space of W' is
    orthogonal to d_hat). `weight` has shape [out_features, in_features]; `d_hat` has
    shape [out_features]."""
    d_hat = d_hat.to(dtype=weight.dtype, device=weight.device)
    proj_row = d_hat @ weight  # [in_features] = d_hat^T W
    return weight - torch.outer(d_hat, proj_row)


def orthogonalize_matrix_norm_preserving(weight: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    """Like orthogonalize_matrix (same d_hat^T W' == 0 guarantee), but rescales the
    WHOLE matrix by one global scalar to restore its original Frobenius norm.

    A per-row rescale (decompose each row into magnitude x direction, project out
    d_hat, restore each row's own original norm independently) does NOT preserve
    exactness here: the d_hat^T W' == 0 guarantee is a joint property of all rows
    together (every row shifts by the same correction vector `proj_row`, scaled by
    that row's own d_hat_i). If row i were rescaled by its own scalar c_i, then
    d_hat^T W'' = sum_i d_hat_i * c_i * w_i', which is only zero if every c_i is
    identical — i.e. per-row rescaling reintroduces a nonzero d_hat component and
    undoes the ablation. A single global scalar is the only magnitude-restoration op
    that can't do that in this write-side convention (weight is [out_features,
    in_features], d_hat is output-space) — do not "fix" this into a per-row version.
    """
    projected = orthogonalize_matrix(weight, d_hat)
    original_norm = weight.norm()
    projected_norm = projected.norm()
    if projected_norm == 0:
        return projected
    return projected * (original_norm / projected_norm)


def orthogonalize_bias(bias: torch.Tensor, d_hat: torch.Tensor) -> torch.Tensor:
    d_hat = d_hat.to(dtype=bias.dtype, device=bias.device)
    coeff = (d_hat @ bias)
    return bias - coeff * d_hat


@torch.no_grad()
def ablate_weights_(
    model: nn.Module,
    directions: torch.Tensor,
    coefficient: float = 1.0,
    layer_indices: list[int] | None = None,
    norm_preserving: bool = False,
) -> None:
    """In-place permanent ablation. `coefficient` blends between the original matrix
    (0.0) and the fully orthogonalized one (1.0) — mirrors the runtime hook's
    partial-strength option (PRD stage [6], rescaling) so the same tradeoff curve can
    be produced for the weight-level edit, not just the runtime one.

    `norm_preserving`: when True, use orthogonalize_matrix_norm_preserving instead of
    orthogonalize_matrix, restoring each edited matrix's original Frobenius norm after
    projecting out d_hat (see that function's docstring for why this must be a global
    rescale, not a per-row one).
    """
    layers = get_decoder_layers(model)
    if directions.shape[0] != len(layers):
        raise ValueError(
            f"directions has {directions.shape[0]} rows but model has {len(layers)} layers"
        )
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    for idx in layer_indices:
        d_hat = directions[idx]
        if torch.count_nonzero(d_hat) == 0:
            continue  # degenerate layer (see extract_direction.compute_directions)
        orthogonalize_fn = (
            orthogonalize_matrix_norm_preserving if norm_preserving else orthogonalize_matrix
        )
        for proj in get_residual_write_modules(layers[idx]):
            new_weight = orthogonalize_fn(proj.weight.data, d_hat)
            proj.weight.data = (
                (1 - coefficient) * proj.weight.data + coefficient * new_weight
            )
            if proj.bias is not None:
                new_bias = orthogonalize_bias(proj.bias.data, d_hat)
                proj.bias.data = (1 - coefficient) * proj.bias.data + coefficient * new_bias
