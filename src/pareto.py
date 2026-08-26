"""Pareto-frontier computation over refusal-rate/capability tradeoff points.

`SweepResult.best()` (layer_sweep.py) and `TradeoffCurve.best_meeting_capability_floor()`
(rescale.py) each collapse a set of candidates down to a single filtered-max pick, which
is the right call when the caller genuinely wants one operating point. Neither exposes
the *full* non-dominated set, though, which is what you actually want when comparing
methods/layers/coefficients against each other or plotting a tradeoff curve — a point
that's second-best on capability but has no other point strictly better on both axes is
still worth showing, and a fixed-target filter throws it away.

Dependency-free and duck-typed: works on `SweepCandidate`, `TradeoffPoint`, or plain
dicts, since all three already use the field names `refusal_rate` (lower is better) and
`capability_score` (higher is better) — whether as dataclass attributes or dict keys,
including as loaded straight from `sweep_result_*.json` / `results_*.json`'s
`"tradeoff_curve"` key.
"""
from __future__ import annotations

from typing import Any


def _get(point: Any, field: str) -> float:
    if isinstance(point, dict):
        return point[field]
    return getattr(point, field)


def pareto_frontier(
    points: list[Any],
    minimize: str = "refusal_rate",
    maximize: str = "capability_score",
) -> list[Any]:
    """Return the full non-dominated subset of `points`.

    A point is dominated (and excluded) if some other point is at least as good on
    both `minimize` and `maximize`, and strictly better on at least one. Ties (two
    points with identical values on both axes) are treated as mutually
    non-dominating, so both survive.

    Unlike `SweepResult.best()` or `TradeoffCurve.best_meeting_capability_floor()`
    (both filtered-max single-point picks), this keeps every point with no other
    point at least as good on both axes and strictly better on one.
    """
    frontier = []
    for candidate in points:
        c_min = _get(candidate, minimize)
        c_max = _get(candidate, maximize)
        dominated = False
        for other in points:
            if other is candidate:
                continue
            o_min = _get(other, minimize)
            o_max = _get(other, maximize)
            at_least_as_good = o_min <= c_min and o_max >= c_max
            strictly_better = o_min < c_min or o_max > c_max
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier
