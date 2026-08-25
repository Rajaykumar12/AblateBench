"""PRD §9.1: weight-level orthogonalization correctness — after orthogonalization,
d_hat^T W' ≈ 0, so the edited matrix can no longer produce a component along d_hat for
*any* input. Also PRD §5 row 4: runtime-vs-weight equivalence — a Linear layer edited
by ablate_weights_ must match project_out's effect on its own output, for the same
direction, with no hook installed afterward."""
import torch
import torch.nn as nn

from ablate_runtime import project_out
from ablate_weights import ablate_weights_, orthogonalize_bias, orthogonalize_matrix
from model_utils import get_residual_write_modules


def test_orthogonalize_matrix_zeroes_direction_for_arbitrary_input():
    torch.manual_seed(0)
    out_features, in_features = 16, 10
    W = torch.randn(out_features, in_features)
    d_hat = torch.randn(out_features)
    d_hat = d_hat / d_hat.norm()

    W_prime = orthogonalize_matrix(W, d_hat)

    # d_hat^T W' must be (numerically) zero regardless of input, since W' rows have
    # had their d_hat component removed directly.
    assert torch.allclose(d_hat @ W_prime, torch.zeros(in_features), atol=1e-5)

    # confirm for several random inputs that output has no component along d_hat
    for _ in range(5):
        x = torch.randn(in_features)
        y = W_prime @ x
        assert torch.isclose(torch.dot(d_hat, y), torch.tensor(0.0), atol=1e-4)


def test_orthogonalize_bias_zeroes_direction_component():
    d_hat = torch.zeros(5)
    d_hat[2] = 1.0
    bias = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    bias_prime = orthogonalize_bias(bias, d_hat)

    assert torch.isclose(torch.dot(d_hat, bias_prime), torch.tensor(0.0), atol=1e-6)
    # only the component along d_hat (axis 2) should change
    assert torch.isclose(bias_prime[2], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(bias_prime[[0, 1, 3, 4]], bias[[0, 1, 3, 4]])


class _TinyDecoderLayer(nn.Module):
    """Minimal stand-in matching model_utils' expected attribute paths
    (self_attn.o_proj, mlp.down_proj) so ablate_weights_ can find its projections
    without needing a real HF model."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(hidden_dim, hidden_dim)


class _TinyModel(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_TinyDecoderLayer(hidden_dim) for _ in range(num_layers)]
        )


def test_ablate_weights_matches_runtime_projection_on_output():
    """Runtime-vs-weight equivalence (PRD §5, row 4): orthogonalizing o_proj's weight
    must produce the same output as running the unmodified layer and then applying
    project_out to its output — proving the permanent edit reproduces the runtime
    hook's effect without needing the hook."""
    torch.manual_seed(3)
    hidden_dim = 12
    model = _TinyModel(hidden_dim=hidden_dim, num_layers=1)
    direction = torch.randn(1, hidden_dim)
    direction[0] = direction[0] / direction[0].norm()

    o_proj = model.model.layers[0].self_attn.o_proj
    x = torch.randn(1, hidden_dim)

    with torch.no_grad():
        y_before = o_proj(x)
        y_via_runtime_hook = project_out(y_before, direction[0], coefficient=1.0)

    ablate_weights_(model, direction, coefficient=1.0)

    with torch.no_grad():
        y_via_weight_edit = o_proj(x)

    assert torch.allclose(y_via_runtime_hook, y_via_weight_edit, atol=1e-4)


def test_ablate_weights_skips_degenerate_layer():
    """A zero direction (degenerate layer, per extract_direction.compute_directions)
    must leave that layer's weights untouched rather than corrupting them."""
    model = _TinyModel(hidden_dim=6, num_layers=1)
    o_proj = model.model.layers[0].self_attn.o_proj
    original_weight = o_proj.weight.data.clone()

    zero_direction = torch.zeros(1, 6)
    ablate_weights_(model, zero_direction, coefficient=1.0)

    assert torch.allclose(o_proj.weight.data, original_weight)


def test_get_residual_write_modules_finds_both_projections():
    model = _TinyModel(hidden_dim=4, num_layers=1)
    modules = get_residual_write_modules(model.model.layers[0])
    assert len(modules) == 2
