"""Parametrized ablation: one config -> per-layer directions + per-layer strengths.

Shared by the optimizer (which scores configs with reversible component hooks) and
the checkpoint builder (which bakes the winning config into the weights), so the
saved model is exactly the configuration that was scored.

The parametrization follows Heretic (p-e-w, 2025):
  - a *fractional* direction index: the refusal direction is interpolated between
    the two neighbouring layers' directions, so the search isn't limited to the
    integer layers whose quality jumps around (0.18 at layer 14, 0.88 at 16 on
    Qwen3.5-35B-A3B);
  - a per-component weight kernel over depth: strength peaks at `max_weight` at
    layer `max_weight_position` and falls linearly to `min_weight` at
    `min_weight_distance` layers away, and is 0 beyond that. Attention and MLP
    writes get independent kernels.
Plus option 2: `n_directions` > 1 also removes extra orthonormal refusal directions
from the subspace file (extract_direction.compute_refusal_subspace).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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
    n_directions: int
    attn: Kernel
    mlp: Kernel
    embed_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AblationConfig":
        return cls(
            direction_index=float(d["direction_index"]),
            n_directions=int(d["n_directions"]),
            attn=Kernel(**d["attn"]),
            mlp=Kernel(**d["mlp"]),
            embed_weight=float(d.get("embed_weight", 0.0)),
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
) -> torch.Tensor:
    """Orthonormal [k, hidden] basis for one config: the interpolated primary
    direction first, then up to n_directions - 1 extra subspace directions from the
    nearest integer layer (re-orthogonalized against the primary)."""
    primary = interpolated_direction(directions, config.direction_index)
    rows = [primary]
    if config.n_directions > 1:
        if subspace is None:
            raise ValueError("n_directions > 1 needs a refusal-subspace file")
        layer = min(round(config.direction_index), subspace.shape[0] - 1)
        extras = subspace[layer, 1 : config.n_directions].float().to(primary.device)
        rows.extend(extras)
    return orthonormal_basis(torch.stack(rows))


def materialize(
    config: AblationConfig,
    directions: torch.Tensor,
    num_layers: int,
    subspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """-> (per-layer basis [num_layers, k, hidden], attn weights, mlp weights).
    The same basis is applied at every layer (broadcast, as in the single-direction
    sweep); only the strengths vary with depth."""
    basis = config_basis(config, directions, subspace)
    per_layer = basis.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
    return per_layer, config.attn.weights(num_layers), config.mlp.weights(num_layers)
