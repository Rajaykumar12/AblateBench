"""Experimental arbitrary-rank ablation (ARA) weight optimizer.

ARA is intentionally a small, dependency-free engine.  Callers provide cached
linear-layer inputs and the corresponding clean outputs; the optimizer preserves
the clean mapping while pulling harmful activations toward the harmless output
manifold.  Row norms are constrained by reparameterizing the optimized matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def knn_distance(left: torch.Tensor, right: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Mean distance from each row in ``left`` to its k nearest rows in ``right``."""
    if left.dim() != 2 or right.dim() != 2:
        raise ValueError("knn_distance expects [samples, features] tensors")
    if left.shape[1] != right.shape[1] or right.shape[0] == 0:
        raise ValueError("kNN inputs must have matching non-empty feature dimensions")
    distances = torch.cdist(left.float(), right.float())
    nearest = distances.topk(min(k, right.shape[0]), largest=False, dim=-1).values
    return nearest.mean()


@dataclass(frozen=True)
class ARAResult:
    loss: float
    iterations: int


def optimize_weight_ara(
    weight: torch.Tensor,
    harmless_inputs: torch.Tensor,
    harmless_targets: torch.Tensor,
    harmful_inputs: torch.Tensor,
    harmful_targets: torch.Tensor,
    *,
    preserve_weight: float = 1.0,
    pull_weight: float = 0.1,
    push_weight: float = 0.05,
    k: int = 8,
    max_iter: int = 5,
) -> tuple[torch.Tensor, ARAResult]:
    """Optimize a 2-D weight with the ARA preserve/pull/push objective."""
    if weight.dim() != 2:
        raise ValueError("ARA currently supports 2-D matrices only")
    if harmless_inputs.shape[1] != weight.shape[1] or harmful_inputs.shape[1] != weight.shape[1]:
        raise ValueError("Input feature dimensions must match weight")
    if harmless_targets.shape[1] != weight.shape[0] or harmful_targets.shape[1] != weight.shape[0]:
        raise ValueError("Target feature dimensions must match weight")

    original = weight.detach().float()
    row_norms = original.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    parameter = nn.Parameter(original.clone())
    optimizer = torch.optim.LBFGS(
        [parameter], max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-9
    )
    iterations = 0
    last_loss = torch.tensor(0.0)

    def closure() -> torch.Tensor:
        nonlocal iterations, last_loss
        optimizer.zero_grad()
        normalized = parameter / parameter.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        candidate = normalized * row_norms
        harmless_pred = harmless_inputs.float() @ candidate.T
        harmful_pred = harmful_inputs.float() @ candidate.T
        loss = preserve_weight * (harmless_pred - harmless_targets.float()).pow(2).mean()
        loss = loss + pull_weight * knn_distance(harmful_pred, harmless_targets, k)
        loss = loss - push_weight * knn_distance(harmful_pred, harmful_targets, k)
        loss.backward()
        iterations += 1
        last_loss = loss.detach()
        return loss

    optimizer.step(closure)
    normalized = parameter.detach() / parameter.detach().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    result = (normalized * row_norms).to(dtype=weight.dtype)
    return result, ARAResult(loss=float(last_loss), iterations=iterations)


def capture_linear_io(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract flattened input/output tensors from a forward-hook invocation."""
    if not inputs:
        raise ValueError("The hooked module produced no input tensor")
    x = inputs[0]
    y = output[0] if isinstance(output, tuple) else output
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("ARA hooks require tensor inputs and outputs")
    return x.detach().reshape(-1, x.shape[-1]), y.detach().reshape(-1, y.shape[-1])
