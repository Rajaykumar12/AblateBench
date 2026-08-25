"""PRD §9.1: runtime ablation hook correctness — h' . d_hat ≈ 0 after projection, and
h' is unchanged along directions orthogonal to d_hat."""
import torch

from ablate_runtime import project_out


def test_projects_out_direction_component_fully():
    torch.manual_seed(0)
    d_hat = torch.randn(16)
    d_hat = d_hat / d_hat.norm()

    h = torch.randn(5, 7, 16)  # [batch, seq, hidden] — applies to every token position
    h_ablated = project_out(h, d_hat, coefficient=1.0)

    residual_component = h_ablated @ d_hat
    assert torch.allclose(residual_component, torch.zeros_like(residual_component), atol=1e-5)


def test_orthogonal_directions_unchanged():
    torch.manual_seed(1)
    hidden_dim = 8
    d_hat = torch.zeros(hidden_dim)
    d_hat[0] = 1.0  # ablate along axis 0 only

    h = torch.randn(3, hidden_dim)
    h_ablated = project_out(h, d_hat, coefficient=1.0)

    # components along axes 1..7 (orthogonal to d_hat) must be untouched
    assert torch.allclose(h[:, 1:], h_ablated[:, 1:])
    # component along axis 0 must be zeroed
    assert torch.allclose(h_ablated[:, 0], torch.zeros(3))


def test_partial_coefficient_scales_projection():
    torch.manual_seed(2)
    d_hat = torch.zeros(4)
    d_hat[0] = 1.0
    h = torch.tensor([[10.0, 1.0, 1.0, 1.0]])

    full = project_out(h, d_hat, coefficient=1.0)
    half = project_out(h, d_hat, coefficient=0.5)

    assert torch.isclose(full[0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(half[0, 0], torch.tensor(5.0), atol=1e-5)


def test_install_and_remove_hooks_round_trip():
    from ablate_runtime import install_ablation_hooks, remove_hooks

    class TinyLayer(torch.nn.Module):
        def forward(self, x):
            return x  # identity layer, output is just the hidden state tensor

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return x

    model = TinyModel()
    directions = torch.eye(4)[:2]  # 2 layers, hidden_dim=4, ablate axis 0 and axis 1
    x = torch.ones(1, 4)

    handles = install_ablation_hooks(model, directions, coefficient=1.0)
    out_ablated = model(x)
    remove_hooks(handles)
    out_restored = model(x)

    # with hooks installed, both dim0 (layer0's direction) and dim1 (layer1's
    # direction) should be projected out
    assert torch.isclose(out_ablated[0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(out_ablated[0, 1], torch.tensor(0.0), atol=1e-5)
    # after removing hooks, the model must be back to its unmodified behavior
    assert torch.allclose(out_restored, x)
