"""PRD §9.1: direction computation correctness — given synthetic activations with a
known mean difference, verify d_layer matches the expected difference-of-means and is
unit-norm."""
import torch

from extract_direction import ActivationSet, compute_directions


def test_direction_matches_known_mean_difference():
    torch.manual_seed(0)
    num_layers, num_prompts, hidden_dim = 3, 50, 8

    # Construct harmful/harmless activations with a known, deliberate mean offset
    # along a fixed axis per layer, plus noise, so the recovered direction should
    # align with that axis after normalization.
    true_axis = torch.zeros(num_layers, hidden_dim)
    true_axis[:, 0] = 1.0  # offset along dim 0 for every layer

    harmless_acts = torch.randn(num_layers, num_prompts, hidden_dim) * 0.01
    harmful_acts = harmless_acts + true_axis.unsqueeze(1) * 5.0

    harmful = ActivationSet(activations=harmful_acts)
    harmless = ActivationSet(activations=harmless_acts)

    directions = compute_directions(harmful, harmless)

    assert directions.shape == (num_layers, hidden_dim)
    for layer in range(num_layers):
        # unit-norm
        assert torch.isclose(directions[layer].norm(), torch.tensor(1.0), atol=1e-5)
        # points along the known offset axis (high cosine similarity)
        cos_sim = torch.dot(directions[layer], true_axis[layer])
        assert cos_sim > 0.99


def test_direction_handles_degenerate_layer_without_nan():
    # harmful and harmless means coincide exactly at layer 0 -> zero diff vector.
    # compute_directions must not divide by zero / produce NaNs.
    acts = torch.zeros(1, 4, 6)
    harmful = ActivationSet(activations=acts.clone())
    harmless = ActivationSet(activations=acts.clone())

    directions = compute_directions(harmful, harmless)

    assert not torch.isnan(directions).any()
    assert torch.allclose(directions, torch.zeros_like(directions))


def test_mismatched_layer_counts_raise():
    harmful = ActivationSet(activations=torch.randn(3, 5, 4))
    harmless = ActivationSet(activations=torch.randn(2, 5, 4))
    try:
        compute_directions(harmful, harmless)
        assert False, "expected ValueError for mismatched layer counts"
    except ValueError:
        pass
