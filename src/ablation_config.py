"""Parametrized ablation: one config -> one model edit.

Shared by the optimizer (which scores configs reversibly) and the checkpoint builder
(which bakes the winning config into the weights). Both go through `apply_config`, so
the saved model is exactly the configuration that was scored.

The parametrization follows Heretic (p-e-w, 2025):
  - a *fractional* direction index: the refusal direction is interpolated between
    the two neighbouring layers' directions;
  - a per-component weight kernel over depth: strength peaks at `max_weight` at
    layer `max_weight_position` and falls linearly to `min_weight` at
    `min_weight_distance` layers away, and is 0 beyond that. Attention and MLP
    writes get independent kernels.
Plus:
  - per-direction strengths for extra refusal-subspace directions (plan Phase 1, the
    SOM paper's lesson: ablate a chosen subset of directions, not all of them);
  - MoE router orthogonalization and expert damping (plan Phase 2, moe_router.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import torch

from ablate_weights import orthonormal_basis


@dataclass
class Kernel:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float

    def weights(self, num_layers: int) -> list[float]:
        out = []
        for i in range(num_layers):
            dist = abs(i - self.max_weight_position)
            if dist > self.min_weight_distance:
                out.append(0.0)
            else:
                frac = dist / self.min_weight_distance if self.min_weight_distance > 0 else 0.0
                out.append(self.max_weight + frac * (self.min_weight - self.max_weight))
        return out


@dataclass
class AblationConfig:
    direction_index: float
    attn: Kernel
    mlp: Kernel
    embed_weight: float = 0.0
    # Strength of each extra subspace direction (rows 1..k-1); 0 switches one off.
    extra_strengths: list[float] = field(default_factory=list)
    # MoE: refusal direction removed from router / shared-expert-gate inputs, at
    # this strength at every MoE layer (0 = off).
    router_strength: float = 0.0
    # MoE: the top-`damp_experts` refusal-associated experts (global ranking from
    # the routing profile) get their down-projection scaled by (1 - damp_alpha).
    damp_experts: int = 0
    damp_alpha: float = 0.0

    @property
    def n_directions(self) -> int:
        return 1 + sum(1 for s in self.extra_strengths if s != 0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AblationConfig":
        extras = d.get("extra_strengths")
        if extras is None:  # configs written before per-direction strengths
            extras = [1.0] * (int(d.get("n_directions", 1)) - 1)
        return cls(
            direction_index=float(d["direction_index"]),
            attn=Kernel(**d["attn"]),
            mlp=Kernel(**d["mlp"]),
            embed_weight=float(d.get("embed_weight", 0.0)),
            extra_strengths=[float(x) for x in extras],
            router_strength=float(d.get("router_strength", 0.0)),
            damp_experts=int(d.get("damp_experts", 0)),
            damp_alpha=float(d.get("damp_alpha", 0.0)),
        )


def interpolated_direction(directions: torch.Tensor, index: float) -> torch.Tensor:
    """Unit direction at fractional layer `index` of a [num_layers, hidden] stack."""
    lo = int(math.floor(index))
    hi = min(lo + 1, directions.shape[0] - 1)
    t = index - lo
    v = (1 - t) * directions[lo].float() + t * directions[hi].float()
    return v / v.norm()


def config_basis(
    config: AblationConfig,
    directions: torch.Tensor,
    subspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (orthonormal [k, hidden] basis, per-row strengths [k]). Row 0 is the
    interpolated primary direction at strength 1; switched-on extra directions come
    from the subspace at the nearest integer layer, re-orthogonalized against it."""
    primary = interpolated_direction(directions, config.direction_index)
    rows, strengths = [primary], [1.0]
    active = [(i, s) for i, s in enumerate(config.extra_strengths, start=1) if s != 0]
    if active:
        if subspace is None:
            raise ValueError("extra directions need a refusal-subspace file")
        layer = min(round(config.direction_index), subspace.shape[0] - 1)
        for i, s in active:
            if i >= subspace.shape[1]:
                raise ValueError(f"subspace has {subspace.shape[1]} rows, config uses row {i}")
            rows.append(subspace[layer, i].float().to(primary.device))
            strengths.append(s)
    basis = orthonormal_basis(torch.stack(rows))
    if basis.shape[0] != len(rows):
        raise ValueError("subspace rows are degenerate against the primary direction")
    return basis, torch.tensor(strengths)


@dataclass
class Materialized:
    basis: torch.Tensor          # [num_layers, k, hidden], same rows at every layer
    row_strengths: torch.Tensor  # [k]
    attn_weights: list[float]
    mlp_weights: list[float]


def materialize(
    config: AblationConfig,
    directions: torch.Tensor,
    num_layers: int,
    subspace: torch.Tensor | None = None,
) -> Materialized:
    basis, strengths = config_basis(config, directions, subspace)
    per_layer = basis.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
    return Materialized(
        per_layer, strengths, config.attn.weights(num_layers), config.mlp.weights(num_layers)
    )


def apply_config(
    model,
    config: AblationConfig,
    directions: torch.Tensor,
    subspace: torch.Tensor | None = None,
    expert_ranking: list[tuple[int, int]] | None = None,
    reversible: bool = True,
) -> Callable[[], None]:
    """Apply `config` to `model`; returns an undo callable.

    reversible=True (optimizer trials): residual writes are ablated with component
    hooks (exactly equal to the weight edit without norm preservation), and router
    edits are in-place weight edits with a snapshot. Undo removes the hooks and
    restores the router tensors.
    reversible=False (checkpoint): everything is a permanent weight edit; undo is a
    no-op.
    """
    from ablate_runtime import install_component_hooks, remove_hooks
    from ablate_weights import ablate_weights_
    from model_utils import get_decoder_layers
    from moe_router import RouterEdit

    num_layers = len(get_decoder_layers(model))
    m = materialize(config, directions, num_layers, subspace)
    # Router edits read the residual's refusal component *before* any residual edit
    # matters (they act on router input weights), so order doesn't change the result.
    router = RouterEdit(model)
    if config.router_strength:
        router.orthogonalize_routers(m.basis[0], [config.router_strength] * num_layers)
    if config.damp_experts and config.damp_alpha:
        if not expert_ranking:
            raise ValueError("expert damping needs an expert ranking (routing profile)")
        router.damp_experts(expert_ranking[: config.damp_experts], config.damp_alpha)

    if not reversible:
        ablate_weights_(
            model, m.basis, attn_weights=m.attn_weights, mlp_weights=m.mlp_weights,
            embed_weight=config.embed_weight, preserve_norms=False,
            row_strengths=m.row_strengths,
        )
        return lambda: None

    param = next(model.parameters())
    handles = install_component_hooks(
        model, m.basis.to(device=param.device, dtype=param.dtype),
        m.attn_weights, m.mlp_weights, config.embed_weight,
        row_strengths=m.row_strengths.to(param.device),
    )

    def undo():
        remove_hooks(handles)
        router.restore()

    return undo
