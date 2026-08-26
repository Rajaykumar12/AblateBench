"""Stage [4]: layer/coefficient sweep.

For each candidate layer's direction (and a shortlist of ablation coefficients),
install runtime ablation (cheap, reversible — see ablate_runtime.py), measure refusal
rate on a small validation slice, and pick the (layer, coefficient) that clears a
target refusal-rate drop at the lowest capability cost. Mirrors Bahushruth's "how do
you decide where to look" question from the call (PRD source plan, §"Direction
computation & layer sweep").

The grid is deliberately a fixed shortlist, not every layer x every coefficient — PRD
§10 (Risks) calls out sweep cost as the main budget lever, so this module caps it by
construction rather than leaving it open-ended.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from ablate_runtime import install_ablation_hooks, remove_hooks
from pareto import pareto_frontier


@dataclass
class SweepCandidate:
    layer_index: int
    coefficient: float
    refusal_rate: float
    capability_score: float


@dataclass
class SweepResult:
    candidates: list[SweepCandidate]

    def best(self, target_refusal_rate: float) -> SweepCandidate | None:
        """Among candidates whose refusal_rate <= target, return the one with the
        highest capability_score (PRD: "clears a target refusal rate with the least
        capability cost"). Returns None if nothing clears the target — a legitimate
        sweep outcome that should be reported, not silently worked around."""
        passing = [c for c in self.candidates if c.refusal_rate <= target_refusal_rate]
        if not passing:
            return None
        return max(passing, key=lambda c: c.capability_score)

    def pareto_frontier(self) -> list[SweepCandidate]:
        """Full non-dominated subset of candidates on (refusal_rate, capability_score)
        — see pareto.pareto_frontier for the dominance rule. Unlike best(), this isn't
        filtered against a fixed target; it surfaces every candidate worth considering."""
        return pareto_frontier(self.candidates)

    def as_table_rows(self) -> list[dict]:
        return [
            {
                "layer": c.layer_index,
                "coefficient": c.coefficient,
                "refusal_rate": c.refusal_rate,
                "capability_score": c.capability_score,
            }
            for c in self.candidates
        ]


def default_layer_shortlist(num_layers: int, num_candidates: int = 6) -> list[int]:
    """Evenly-spaced candidate layers spanning the model depth (skip layer 0 and the
    final layer, where the direction is typically least concentrated per Arditi et
    al.'s findings). Keeps the sweep grid bounded regardless of model depth."""
    if num_layers <= num_candidates:
        return list(range(num_layers))
    start, end = 1, num_layers - 2
    step = max(1, (end - start) // (num_candidates - 1))
    return sorted(set(range(start, end + 1, step)))[:num_candidates]


DEFAULT_COEFFICIENT_SHORTLIST = [1.0, 0.85, 0.7, 0.5]


def run_ablation_sweep(
    model: nn.Module,
    directions: torch.Tensor,
    measure_refusal_rate: Callable[[], float],
    measure_capability: Callable[[], float],
    layer_shortlist: list[int] | None = None,
    coefficient_shortlist: list[float] | None = None,
) -> SweepResult:
    """Drive the sweep. For each candidate layer, its per-layer direction is treated
    as *the* refusal direction and projected out of **every** decoder layer's output
    (per Arditi et al.: the residual stream is additive, so a direction ablated at
    only its source layer can be reintroduced by later layers' attention/MLP
    computation from other pathways — removing it once is not the same as it never
    being present in the stream). `layer_idx` here selects *which* candidate
    direction to test model-wide, not *where* to intervene.

    `measure_refusal_rate` and `measure_capability` are callbacks with no arguments —
    they're expected to run the eval harnesses (eval_refusal.py / eval_capability.py)
    against the *currently* ablated model, since the ablation hooks are
    installed/removed by this function around each callback pair. Keeping the eval
    harnesses as injected callbacks (rather than importing and calling them directly
    here) avoids coupling the sweep to a specific prompt set or tokenizer setup, which
    differs between the validation slice used here and the full held-out eval run in
    stage [7]/[8].
    """
    num_layers = directions.shape[0]
    layer_shortlist = layer_shortlist or default_layer_shortlist(num_layers)
    coefficient_shortlist = coefficient_shortlist or DEFAULT_COEFFICIENT_SHORTLIST

    candidates = []
    for layer_idx in layer_shortlist:
        # broadcast this candidate layer's direction to every layer of the model —
        # the model-wide intervention the technique actually requires.
        broadcast_directions = directions[layer_idx].unsqueeze(0).expand(num_layers, -1)
        for coeff in coefficient_shortlist:
            handles = install_ablation_hooks(
                model, broadcast_directions, coefficient=coeff, layer_indices=None
            )
            try:
                refusal_rate = measure_refusal_rate()
                capability_score = measure_capability()
            finally:
                remove_hooks(handles)
            candidates.append(
                SweepCandidate(
                    layer_index=layer_idx,
                    coefficient=coeff,
                    refusal_rate=refusal_rate,
                    capability_score=capability_score,
                )
            )
    return SweepResult(candidates=candidates)
