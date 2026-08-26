"""PRD §9.2: sweep monotonicity sanity check + grid-size bounding (§10 risk: sweep
cost creep)."""
import torch
import torch.nn as nn

from layer_sweep import (
    DEFAULT_COEFFICIENT_SHORTLIST,
    default_layer_shortlist,
    run_ablation_sweep,
)


class _IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class _TinyModel(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_IdentityLayer() for _ in range(num_layers)])


def test_default_layer_shortlist_is_bounded_and_within_range():
    for num_layers in (4, 8, 32, 64):
        shortlist = default_layer_shortlist(num_layers, num_candidates=6)
        assert len(shortlist) <= 6
        assert all(0 <= i < num_layers for i in shortlist)
        assert len(shortlist) == len(set(shortlist))  # no duplicates


def test_default_layer_shortlist_returns_all_layers_when_model_is_shallow():
    assert default_layer_shortlist(3, num_candidates=6) == [0, 1, 2]


def test_sweep_grid_size_matches_layers_times_coefficients():
    model = _TinyModel(num_layers=4)
    directions = torch.eye(4)  # one distinct direction per layer

    call_count = {"n": 0}

    def measure_refusal_rate():
        call_count["n"] += 1
        return 0.5

    def measure_capability():
        return 0.8

    layer_shortlist = [0, 1]
    coeff_shortlist = [1.0, 0.5]
    result = run_ablation_sweep(
        model,
        directions,
        measure_refusal_rate,
        measure_capability,
        layer_shortlist=layer_shortlist,
        coefficient_shortlist=coeff_shortlist,
    )

    assert len(result.candidates) == len(layer_shortlist) * len(coeff_shortlist)
    assert call_count["n"] == len(layer_shortlist) * len(coeff_shortlist)


def test_sweep_removes_hooks_between_candidates():
    """A bug that leaves hooks stacked across candidates would make later candidates'
    ablation effects compound; verify the model is unmodified between sweep calls by
    checking hook count on the layer stays at zero once run_ablation_sweep returns."""
    model = _TinyModel(num_layers=2)
    directions = torch.eye(2)

    run_ablation_sweep(
        model,
        directions,
        measure_refusal_rate=lambda: 0.0,
        measure_capability=lambda: 1.0,
        layer_shortlist=[0, 1],
        coefficient_shortlist=[1.0],
    )

    for layer in model.model.layers:
        assert len(layer._forward_hooks) == 0


def test_best_picks_highest_capability_among_candidates_clearing_target():
    from layer_sweep import SweepCandidate, SweepResult

    candidates = [
        SweepCandidate(layer_index=0, coefficient=1.0, refusal_rate=0.05, capability_score=0.60),
        SweepCandidate(layer_index=1, coefficient=0.7, refusal_rate=0.15, capability_score=0.85),
        SweepCandidate(layer_index=2, coefficient=0.5, refusal_rate=0.40, capability_score=0.95),
    ]
    result = SweepResult(candidates=candidates)

    # target 0.2: candidates 0 and 1 clear it; pick the one with higher capability (1)
    best = result.best(target_refusal_rate=0.2)
    assert best.layer_index == 1

    # impossible target -> None
    assert result.best(target_refusal_rate=0.01) is None


def test_default_coefficient_shortlist_starts_at_full_strength():
    assert DEFAULT_COEFFICIENT_SHORTLIST[0] == 1.0


def test_sweep_ablates_candidate_direction_at_every_layer_not_just_its_source():
    """Regression test for a real bug found during the first live Modal sweep run:
    ablating only at the candidate's source layer left refusal rate almost unchanged
    (74-96% across every layer/coefficient) because downstream layers could still
    reintroduce the direction. The fix broadcasts the chosen layer's direction to
    every layer, matching Arditi et al.'s actual technique (see layer_sweep.py's
    run_ablation_sweep docstring)."""

    class RecordingLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input_seen = None

        def forward(self, x):
            self.last_input_seen = x.clone()
            return x

    class ChainedModel(nn.Module):
        """_TinyModel has no forward() (hooks alone sufficed for the other tests);
        this one actually chains layers sequentially, since this test needs a real
        forward pass to observe cross-layer propagation."""

        def __init__(self, layers):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(layers)

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return x

    model = ChainedModel([RecordingLayer() for _ in range(3)])
    # distinct, non-degenerate direction per layer so a broadcast is observable
    directions = torch.eye(3)

    seen_projected_by_layer = {}

    def measure_refusal_rate():
        # probe with a fixed input aligned with layer 0's direction; record, for each
        # layer, whether that layer's *output* still contains a component along
        # layer 0's direction (it shouldn't, if ablation is applied model-wide).
        x = torch.tensor([[1.0, 1.0, 1.0]])
        model(x)
        for i, layer in enumerate(model.model.layers):
            seen_projected_by_layer[i] = layer.last_input_seen[0, 0].item()
        return 0.0

    run_ablation_sweep(
        model,
        directions,
        measure_refusal_rate,
        measure_capability=lambda: 1.0,
        layer_shortlist=[0],  # candidate direction taken from layer 0
        coefficient_shortlist=[1.0],
    )

    # layer 0's *input* is unmodified (hooks fire on output), but every layer's
    # *output* (== next layer's input) must have had axis 0 (layer 0's direction)
    # projected out if the broadcast is working - so layer 1 and layer 2's recorded
    # inputs must show axis 0 zeroed, not just layer 0's.
    assert seen_projected_by_layer[0] == 1.0  # layer 0 input: pre-hook, untouched
    assert abs(seen_projected_by_layer[1]) < 1e-5  # layer 1 input: post layer-0-hook
    assert abs(seen_projected_by_layer[2]) < 1e-5  # layer 2 input: still ablated


def test_sweep_result_pareto_frontier_delegates_to_pareto_module():
    from layer_sweep import SweepCandidate, SweepResult

    candidates = [
        SweepCandidate(layer_index=0, coefficient=1.0, refusal_rate=0.30, capability_score=0.70),
        SweepCandidate(layer_index=1, coefficient=0.7, refusal_rate=0.10, capability_score=0.90),
        SweepCandidate(layer_index=2, coefficient=0.5, refusal_rate=0.05, capability_score=0.60),
    ]
    result = SweepResult(candidates=candidates)

    frontier = result.pareto_frontier()
    assert candidates[0] not in frontier  # dominated by candidate 1
    assert candidates[1] in frontier
    assert candidates[2] in frontier
