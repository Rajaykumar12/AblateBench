import torch

from rescale import TradeoffCurve, TradeoffPoint, renormalize_to_original_norm


def test_renormalize_restores_original_norm_preserving_direction():
    torch.manual_seed(0)
    h_original = torch.randn(3, 8) * 5.0
    # simulate ablation shrinking the norm
    h_ablated = h_original * 0.4

    h_rescaled = renormalize_to_original_norm(h_ablated, h_original)

    assert torch.allclose(
        h_rescaled.norm(dim=-1), h_original.norm(dim=-1), atol=1e-4
    )
    # direction (unit vector) must be unchanged from the ablated version
    ablated_dir = h_ablated / h_ablated.norm(dim=-1, keepdim=True)
    rescaled_dir = h_rescaled / h_rescaled.norm(dim=-1, keepdim=True)
    assert torch.allclose(ablated_dir, rescaled_dir, atol=1e-4)


def test_renormalize_handles_zero_norm_without_nan():
    h_original = torch.randn(1, 4)
    h_ablated = torch.zeros(1, 4)
    result = renormalize_to_original_norm(h_ablated, h_original)
    assert not torch.isnan(result).any()


def test_tradeoff_curve_best_meeting_capability_floor():
    points = [
        TradeoffPoint(coefficient=1.0, method="runtime", renormalized=False, refusal_rate=0.05, capability_score=0.60),
        TradeoffPoint(coefficient=0.7, method="runtime", renormalized=False, refusal_rate=0.15, capability_score=0.80),
        TradeoffPoint(coefficient=0.5, method="runtime", renormalized=False, refusal_rate=0.30, capability_score=0.90),
    ]
    curve = TradeoffCurve(points=points)

    # floor of 0.75 excludes the first point (0.60 < 0.75); among the remaining two,
    # pick the lowest refusal rate.
    best = curve.best_meeting_capability_floor(0.75)
    assert best.coefficient == 0.7

    # impossible floor -> None
    assert curve.best_meeting_capability_floor(0.99) is None


def test_tradeoff_curve_as_table_rows_shape():
    points = [TradeoffPoint(1.0, "weight", True, 0.1, 0.85, judge_fallback_rate=0.2)]
    rows = TradeoffCurve(points=points).as_table_rows()
    assert rows == [
        {
            "coefficient": 1.0,
            "method": "weight",
            "renormalized": True,
            "refusal_rate": 0.1,
            "capability_score": 0.85,
            "judge_fallback_rate": 0.2,
        }
    ]


def test_tradeoff_point_judge_fallback_rate_defaults_to_zero():
    # existing callers that don't track this shouldn't need updating
    p = TradeoffPoint(1.0, "runtime", False, 0.5, 0.9)
    assert p.judge_fallback_rate == 0.0
