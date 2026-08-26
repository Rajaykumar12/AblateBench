"""pareto_frontier: full non-dominated subset, as opposed to the filtered-max
single-point picks in SweepResult.best() / TradeoffCurve.best_meeting_capability_floor().
"""
from dataclasses import dataclass

from pareto import pareto_frontier
from rescale import TradeoffPoint
from layer_sweep import SweepCandidate


def test_dominated_point_excluded():
    # point B has lower refusal_rate AND higher capability_score than A on both axes
    # -> A is dominated and must be excluded.
    points = [
        {"refusal_rate": 0.30, "capability_score": 0.70},  # A: dominated by B
        {"refusal_rate": 0.10, "capability_score": 0.90},  # B: dominates A
        {"refusal_rate": 0.05, "capability_score": 0.60},  # C: non-dominated (lower refusal, but lower capability than B)
    ]
    frontier = pareto_frontier(points)
    assert points[0] not in frontier
    assert points[1] in frontier
    assert points[2] in frontier
    assert len(frontier) == 2


def test_all_non_dominated_set_returns_everything():
    # classic staircase of tradeoffs: each point trades refusal for capability, none
    # dominates another.
    points = [
        {"refusal_rate": 0.05, "capability_score": 0.60},
        {"refusal_rate": 0.15, "capability_score": 0.80},
        {"refusal_rate": 0.30, "capability_score": 0.95},
    ]
    frontier = pareto_frontier(points)
    assert len(frontier) == len(points)
    for p in points:
        assert p in frontier


def test_works_on_sweep_candidate_dataclass():
    candidates = [
        SweepCandidate(layer_index=0, coefficient=1.0, refusal_rate=0.30, capability_score=0.70),
        SweepCandidate(layer_index=1, coefficient=0.7, refusal_rate=0.10, capability_score=0.90),
        SweepCandidate(layer_index=2, coefficient=0.5, refusal_rate=0.05, capability_score=0.60),
    ]
    frontier = pareto_frontier(candidates)
    assert candidates[0] not in frontier
    assert candidates[1] in frontier
    assert candidates[2] in frontier


def test_works_on_tradeoff_point_dataclass():
    points = [
        TradeoffPoint(coefficient=1.0, method="runtime", renormalized=False, refusal_rate=0.30, capability_score=0.70),
        TradeoffPoint(coefficient=0.7, method="runtime", renormalized=False, refusal_rate=0.10, capability_score=0.90),
        TradeoffPoint(coefficient=0.5, method="runtime", renormalized=False, refusal_rate=0.05, capability_score=0.60),
    ]
    frontier = pareto_frontier(points)
    assert points[0] not in frontier
    assert points[1] in frontier
    assert points[2] in frontier


def test_identical_points_are_mutually_non_dominating():
    a = {"refusal_rate": 0.1, "capability_score": 0.8}
    b = {"refusal_rate": 0.1, "capability_score": 0.8}
    frontier = pareto_frontier([a, b])
    assert len(frontier) == 2


def test_works_on_arbitrary_dataclass_with_matching_field_names():
    @dataclass
    class Point:
        refusal_rate: float
        capability_score: float

    points = [Point(0.30, 0.70), Point(0.10, 0.90)]
    frontier = pareto_frontier(points)
    assert frontier == [points[1]]
