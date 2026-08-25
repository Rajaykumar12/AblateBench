"""Stage [6]: rescaling — measure capability drop, then recover it.

PRD is explicit that capability drop must be measured *before* any rescaling claim is
made (§7 stage 6: "measure capability drop first ... don't skip this"). This module
provides the two recovery mechanisms the plan calls out — a partial-strength ablation
coefficient (handled by the `coefficient` argument already present in
ablate_runtime.project_out / ablate_weights.ablate_weights_) and activation-norm
renormalization — plus the tradeoff-curve bookkeeping used to report results across
several coefficients rather than a single before/after pair (PRD §5, row 6; §11).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def renormalize_to_original_norm(h_ablated: torch.Tensor, h_original: torch.Tensor) -> torch.Tensor:
    """Rescale each ablated hidden-state vector's norm back to its pre-ablation norm,
    direction unchanged. Ablation only removes a component of h, which can shrink its
    norm; renormalizing restores the vector's overall scale so downstream layers (which
    may be sensitive to activation magnitude) see something closer to their expected
    input distribution.

    Both tensors must be broadcastable and share their last dimension (hidden_dim).
    """
    orig_norm = h_original.norm(dim=-1, keepdim=True)
    ablated_norm = h_ablated.norm(dim=-1, keepdim=True)
    safe_ablated_norm = torch.where(
        ablated_norm > 0, ablated_norm, torch.ones_like(ablated_norm)
    )
    return h_ablated * (orig_norm / safe_ablated_norm)


@dataclass
class TradeoffPoint:
    """One row of the refusal-rate-vs-capability tradeoff curve (PRD §11)."""

    coefficient: float
    method: str  # "runtime" or "weight"
    renormalized: bool
    refusal_rate: float
    capability_score: float
    # Fraction of graded responses that needed the judge tier at this point (PRD
    # §9.2/§11: grader validation numbers reported alongside refusal-rate results).
    # Defaults to 0.0 so existing callers that don't track this don't need updating.
    judge_fallback_rate: float = 0.0


@dataclass
class TradeoffCurve:
    points: list[TradeoffPoint]

    def best_meeting_capability_floor(self, capability_floor: float) -> TradeoffPoint | None:
        """Among points whose capability_score >= capability_floor, return the one
        with the lowest refusal_rate. Used to pick the operating point that satisfies
        PRD §5 row 6 (capability within ~5 points of baseline, refusal-rate drop
        retained as much as possible)."""
        candidates = [p for p in self.points if p.capability_score >= capability_floor]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.refusal_rate)

    def as_table_rows(self) -> list[dict]:
        """Plain-dict rows, ready to drop into the README's tradeoff-curve table."""
        return [
            {
                "coefficient": p.coefficient,
                "method": p.method,
                "renormalized": p.renormalized,
                "refusal_rate": p.refusal_rate,
                "capability_score": p.capability_score,
                "judge_fallback_rate": p.judge_fallback_rate,
            }
            for p in self.points
        ]
