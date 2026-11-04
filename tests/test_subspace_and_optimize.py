"""Refusal subspace (option 2) and optimizer parametrization (option 1): the
component hooks scored by `--stage optimize` must equal the weight edit that
`--stage full --ablation-config` bakes into the checkpoint, for a rank-k subspace
and per-layer attention / MLP strengths, on a toy MoE with packed experts and a
shared expert."""
import copy

import torch
import torch.nn as nn

from ablate_runtime import install_component_hooks, remove_hooks
from ablate_weights import ablate_weights_, orthogonalize_matrix, orthonormal_basis
from ablation_config import AblationConfig, Kernel, config_basis, interpolated_direction, materialize
from extract_direction import ActivationSet, compute_refusal_subspace

H, INNER, E = 12, 6, 4


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(H, H)
        self.o_proj = nn.Linear(H, H)

    def forward(self, x):
        return (self.o_proj(torch.tanh(self.q(x))), None)  # HF-style tuple output


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.randn(E, H, INNER) * 0.3)
        self.down_proj = nn.Parameter(torch.randn(E, INNER, H) * 0.3)  # [E, in, d_model]

    def forward(self, x, gates):
        h = torch.relu(torch.einsum("bsh,ehi->bsei", x, self.up))
        out = torch.einsum("bsei,eid->bsed", h, self.down_proj)
        return (out * gates.unsqueeze(-1)).sum(-2)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, E)
        self.experts = _Experts()
        self.shared_expert = nn.Module()
        self.shared_expert.up = nn.Linear(H, INNER)
        self.shared_expert.down_proj = nn.Linear(INNER, H)

    def forward(self, x):
        gates = torch.softmax(self.gate(x), -1)
        shared = self.shared_expert.down_proj(torch.relu(self.shared_expert.up(x)))
        return self.experts(x, gates) + torch.sigmoid(x.sum(-1, keepdim=True)) * shared


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MoE()

    def forward(self, x):
        x = x + self.self_attn(x)[0]
        return x + self.mlp(x)


class _ToyMoE(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(20, H)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer() for _ in range(num_layers)])

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return None

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.model.layers:
            x = layer(x)
        return x


def _basis(k, seed=0):
    torch.manual_seed(seed)
    return orthonormal_basis(torch.randn(k, H))


def test_subspace_orthogonalization_removes_every_direction_for_all_layouts():
    q = _basis(3)
    torch.manual_seed(1)
    for w, out in [
        (torch.randn(H, INNER), lambda w: q @ w),
        (torch.randn(E, H, INNER), lambda w: torch.einsum("kh,ehi->eki", q, w)),
        (torch.randn(E, INNER, H), lambda w: w @ q.T),
    ]:
        for preserve in (False, True):
            assert out(orthogonalize_matrix(w, q, preserve_norms=preserve)).abs().max() < 1e-5


def test_component_hooks_match_weighted_weight_edit():
    torch.manual_seed(0)
    model = _ToyMoE()
    edited = copy.deepcopy(model)
    q = _basis(2)
    directions = q.unsqueeze(0).expand(4, -1, -1).contiguous()
    attn_w, mlp_w, embed_w = [1.2, 0.0, 0.7, 1.0], [0.5, 1.0, 0.0, 1.3], 0.8
    ids = torch.randint(0, 20, (2, 5))

    handles = install_component_hooks(model, directions, attn_w, mlp_w, embed_w)
    with torch.no_grad():
        hooked = model(ids)
    remove_hooks(handles)

    ablate_weights_(edited, directions, attn_weights=attn_w, mlp_weights=mlp_w, embed_weight=embed_w)
    with torch.no_grad():
        baked = edited(ids)
    assert torch.allclose(hooked, baked, atol=1e-5)


def test_kernel_shape():
    k = Kernel(max_weight=1.2, max_weight_position=5, min_weight=0.2, min_weight_distance=2)
    w = k.weights(9)
    assert w[5] == 1.2
    assert abs(w[4] - 0.7) < 1e-9 and abs(w[7] - 0.2) < 1e-9
    assert w[2] == 0.0 and w[8] == 0.0


def test_fractional_direction_interpolates_and_is_unit():
    d = torch.eye(3)
    v = interpolated_direction(d, 0.5)
    assert torch.allclose(v, torch.tensor([1.0, 1.0, 0.0]) / 2**0.5)
    assert torch.allclose(interpolated_direction(d, 2.0), d[2])


def test_config_round_trip_and_materialize():
    flat = Kernel(1.0, 1.0, 1.0, 10.0)
    cfg = AblationConfig(direction_index=1.3, n_directions=2, attn=flat, mlp=flat, embed_weight=0.5)
    assert AblationConfig.from_dict(cfg.to_dict()) == cfg
    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(4, H), dim=-1)
    subspace = torch.stack([orthonormal_basis(torch.randn(3, H)) for _ in range(4)])
    basis = config_basis(cfg, directions, subspace)
    assert basis.shape == (2, H)
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-5)
    per_layer, aw, mw = materialize(cfg, directions, 4, subspace)
    assert per_layer.shape == (4, 2, H) and aw == [1.0] * 4 and mw == [1.0] * 4


def test_refusal_subspace_recovers_a_second_cluster_direction():
    """Two harmful clusters refusing along different axes: row 0 is the mean
    difference, row 1 must pick up the axis the mean under-represents."""
    torch.manual_seed(0)
    n, d = 40, 16
    harmless = torch.randn(1, n, d) * 0.1
    a = torch.zeros(d); a[0] = 3.0
    b = torch.zeros(d); b[1] = 3.0
    harmful = torch.cat([torch.randn(n // 2, d) * 0.1 + a, torch.randn(n // 2, d) * 0.1 + 0.3 * b + a])
    harmful = torch.cat([harmful, torch.randn(n // 2, d) * 0.1 + b])[None]
    sub = compute_refusal_subspace(
        ActivationSet(harmful), ActivationSet(harmless), k=2, n_clusters=3, purified=False
    )
    assert sub.shape == (1, 2, d)
    assert torch.allclose(sub[0] @ sub[0].T, torch.eye(2), atol=1e-5)
    # span of the two rows contains both refusal axes
    proj = sub[0].T @ sub[0]
    for axis in (a, b):
        u = axis / axis.norm()
        assert (proj @ u).norm() > 0.95
