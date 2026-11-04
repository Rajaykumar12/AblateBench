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


@dataclass
class SweepCandidate:
    layer_index: int
    coefficient: float
    refusal_rate: float
    capability_score: float
    band: str | None = None  # "start-end" for a per-layer band candidate


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

    def as_table_rows(self) -> list[dict]:
        return [
            {
                "layer": c.layer_index if c.band is None else f"band:{c.band}",
                "coefficient": c.coefficient,
                "refusal_rate": c.refusal_rate,
                "capability_score": c.capability_score,
            }
            for c in self.candidates
        ]


def parse_band(band: str) -> tuple[int, int]:
    """'24-35' -> (24, 35), inclusive on both ends."""
    start, end = (int(x) for x in band.split("-"))
    if start > end:
        raise ValueError(f"band start {start} is after end {end}")
    return start, end


def band_directions(directions: torch.Tensor, band: str) -> torch.Tensor:
    """Each layer inside `band` keeps its *own* direction; every other layer gets the
    zero vector, which install_ablation_hooks' callers and ablate_weights_ treat as
    "leave this layer alone". The alternative to broadcasting one layer's direction
    model-wide: only layers with a strong refusal signal are touched, each with the
    direction measured there."""
    start, end = parse_band(band)
    if not 0 <= start <= end < directions.shape[0]:
        raise ValueError(f"band {band} outside 0-{directions.shape[0] - 1}")
    out = torch.zeros_like(directions)
    out[start : end + 1] = directions[start : end + 1]
    return out


def run_band_sweep(
    model: nn.Module,
    directions: torch.Tensor,
    bands: list[str],
    measure_refusal_rate: Callable[[], object],
    measure_capability: Callable[[], float],
    coefficient_shortlist: list[float] | None = None,
    grade_refusal: Callable[[object], float] | None = None,
) -> SweepResult:
    """Per-layer band candidates, scored exactly like run_ablation_sweep's (same
    callbacks, grading after the hooks are removed) so the rows are comparable."""
    coefficient_shortlist = coefficient_shortlist or DEFAULT_COEFFICIENT_SHORTLIST
    candidates = []
    for band in bands:
        start, end = parse_band(band)
        per_layer = band_directions(directions, band)
        for coeff in coefficient_shortlist:
            handles = install_ablation_hooks(
                model, per_layer, coefficient=coeff, layer_indices=list(range(start, end + 1))
            )
            try:
                refusal_rate = measure_refusal_rate()
                capability_score = measure_capability()
            finally:
                remove_hooks(handles)
            if grade_refusal is not None:
                refusal_rate = grade_refusal(refusal_rate)
            candidates.append(
                SweepCandidate(
                    layer_index=start,
                    coefficient=coeff,
                    refusal_rate=refusal_rate,
                    capability_score=capability_score,
                    band=band,
                )
            )
    return SweepResult(candidates=candidates)


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
    grade_refusal: Callable[[object], float] | None = None,
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

    `grade_refusal`: when given, `measure_refusal_rate` returns raw generations
    (produced with hooks installed) and `grade_refusal(generations)` turns them into a
    rate *after* the hooks are removed - so a model-based judge grades with the clean
    model rather than the ablated one, which otherwise mislabels its own outputs.
    """
    num_layers = directions.shape[0]
    layer_shortlist = layer_shortlist or default_layer_shortlist(num_layers)
    coefficient_shortlist = coefficient_shortlist or DEFAULT_COEFFICIENT_SHORTLIST

    candidates = []
    for layer_idx in layer_shortlist:
        # broadcast this candidate layer's direction to every layer of the model —
        # the model-wide intervention the technique actually requires.
        # ([L, hidden] single directions or [L, k, hidden] refusal subspaces)
        broadcast_directions = directions[layer_idx].unsqueeze(0).expand(
            num_layers, *directions.shape[1:]
        )
        for coeff in coefficient_shortlist:
            handles = install_ablation_hooks(
                model, broadcast_directions, coefficient=coeff, layer_indices=None
            )
            try:
                refusal_rate = measure_refusal_rate()
                capability_score = measure_capability()
            finally:
                remove_hooks(handles)
            if grade_refusal is not None:
                refusal_rate = grade_refusal(refusal_rate)
            candidates.append(
                SweepCandidate(
                    layer_index=layer_idx,
                    coefficient=coeff,
                    refusal_rate=refusal_rate,
                    capability_score=capability_score,
                )
            )
    return SweepResult(candidates=candidates)
