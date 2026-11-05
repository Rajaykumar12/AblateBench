"""Refusal subspace (option 2) and optimizer parametrization (option 1): the
component hooks scored by `--stage optimize` must equal the weight edit that
`--stage full --ablation-config` bakes into the checkpoint, for a rank-k subspace
and per-layer attention / MLP strengths, on a toy MoE with packed experts and a
shared expert."""
import copy

import pytest

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
    cfg = AblationConfig(
        direction_index=1.3, attn=flat, mlp=flat, embed_weight=0.5, extra_strengths=[0.0, 0.7]
    )
    assert AblationConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.n_directions == 2
    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(4, H), dim=-1)
    subspace = torch.stack([orthonormal_basis(torch.randn(3, H)) for _ in range(4)])
    basis, strengths = config_basis(cfg, directions, subspace)
    assert basis.shape == (2, H) and strengths.tolist() == pytest.approx([1.0, 0.7])
    assert torch.allclose(basis @ basis.T, torch.eye(2), atol=1e-5)
    m = materialize(cfg, directions, 4, subspace)
    assert m.basis.shape == (4, 2, H) and m.attn_weights == [1.0] * 4 and m.mlp_weights == [1.0] * 4


def test_old_configs_with_n_directions_still_load():
    flat = {"max_weight": 1.0, "max_weight_position": 1.0, "min_weight": 1.0, "min_weight_distance": 9.0}
    cfg = AblationConfig.from_dict(
        {"direction_index": 2.0, "n_directions": 3, "attn": flat, "mlp": flat, "embed_weight": 0.3}
    )
    assert cfg.extra_strengths == [1.0, 1.0] and cfg.router_strength == 0.0


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


def test_per_direction_strengths_match_between_hooks_and_weights():
    torch.manual_seed(2)
    model = _ToyMoE()
    edited = copy.deepcopy(model)
    q = _basis(3)
    directions = q.unsqueeze(0).expand(4, -1, -1).contiguous()
    rs = torch.tensor([1.0, 0.4, 1.3])
    aw, mw = [1.0, 0.5, 1.2, 0.0], [0.8, 1.0, 0.3, 1.1]
    ids = torch.randint(0, 20, (2, 5))
    handles = install_component_hooks(model, directions, aw, mw, 0.6, row_strengths=rs)
    with torch.no_grad():
        hooked = model(ids)
    remove_hooks(handles)
    ablate_weights_(edited, directions, attn_weights=aw, mlp_weights=mw, embed_weight=0.6, row_strengths=rs)
    with torch.no_grad():
        assert torch.allclose(hooked, edited(ids), atol=1e-5)


def test_zero_strength_row_is_a_no_op_for_that_direction():
    q = _basis(2)
    w = torch.randn(H, INNER)
    only_first = orthogonalize_matrix(w, q[:1])
    with_zero = orthogonalize_matrix(w, q, row_strengths=torch.tensor([1.0, 0.0]))
    assert torch.allclose(only_first, with_zero, atol=1e-6)


def _flat_cfg(**kw):
    flat = Kernel(1.0, 1.5, 0.6, 3.0)
    return AblationConfig(direction_index=1.4, attn=flat, mlp=flat, embed_weight=0.5, **kw)


def test_apply_config_reversible_equals_permanent_with_router_edits():
    from ablation_config import apply_config

    torch.manual_seed(3)
    model = _ToyMoE()
    permanent = copy.deepcopy(model)
    original = copy.deepcopy(model)
    directions = torch.nn.functional.normalize(torch.randn(4, H), dim=-1)
    subspace = torch.stack([orthonormal_basis(torch.randn(3, H)) for _ in range(4)])
    ranking = [(0, 1), (2, 3), (1, 0)]
    cfg = _flat_cfg(extra_strengths=[0.8, 0.0], router_strength=0.9, damp_experts=2, damp_alpha=0.5)
    ids = torch.randint(0, 20, (2, 6))

    undo = apply_config(model, cfg, directions, subspace, ranking, reversible=True)
    with torch.no_grad():
        trial = model(ids)
    undo()
    apply_config(permanent, cfg, directions, subspace, ranking, reversible=False)
    with torch.no_grad():
        assert torch.allclose(trial, permanent(ids), atol=1e-5)
        # undo restores the model exactly
        assert torch.equal(model(ids), original(ids))


def test_router_orthogonalization_makes_routing_blind_to_the_direction():
    from moe_router import RouterEdit

    torch.manual_seed(4)
    model = _ToyMoE()
    d = torch.nn.functional.normalize(torch.randn(H), dim=0)
    RouterEdit(model).orthogonalize_routers(d.unsqueeze(0), [1.0] * 4)
    gate = model.model.layers[0].mlp.gate
    x = torch.randn(5, H)
    with torch.no_grad():
        assert torch.allclose(gate(x), gate(x + 3.0 * d), atol=1e-5)


def test_router_orthogonalization_accounts_for_the_norm_scale():
    """With a zero-centered RMSNorm (x / rms(x) * (1 + w)) in front of the router, the
    edited router must give the same logits as the original router fed the normalized
    input with the refusal component removed *before* the per-channel scale:
    W'(g * x/rms) == W(g * P(x/rms)), P = I - d d^T."""
    from moe_router import RouterEdit

    class _Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(H) * 0.5)

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1 + self.weight)

    torch.manual_seed(5)
    model = _ToyMoE()
    layer = model.model.layers[0]
    layer.post_attention_layernorm = _Norm()
    original_w = layer.mlp.gate.weight.detach().clone()
    d = torch.nn.functional.normalize(torch.randn(H), dim=0)
    RouterEdit(model).orthogonalize_routers(d.unsqueeze(0), [1.0] * 4)
    x = torch.randn(5, H) + 2.0 * d
    with torch.no_grad():
        g = 1 + layer.post_attention_layernorm.weight
        x_hat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_hat_p = x_hat - (x_hat @ d).unsqueeze(1) * d
        edited = (g * x_hat) @ layer.mlp.gate.weight.T
    # equality holds for the d-induced signal; the edit also drops whatever else lies
    # along normalize(g*d), so compare on inputs where that is the only difference
    u = torch.nn.functional.normalize(g * d, dim=0)
    delta = (g * x_hat - g * x_hat_p)
    assert torch.allclose(delta - (delta @ u).unsqueeze(1) * u, torch.zeros_like(delta), atol=1e-5)
    with torch.no_grad():
        resid = (g * x_hat_p) - ((g * x_hat_p) @ u).unsqueeze(1) * u
        assert torch.allclose(edited, resid @ original_w.T, atol=1e-4)
        assert not torch.allclose(edited, (g * x_hat) @ original_w.T, atol=1e-3)


def test_damping_scales_only_the_listed_experts_and_restores():
    from moe_router import RouterEdit

    torch.manual_seed(6)
    model = _ToyMoE()
    before = model.model.layers[1].mlp.experts.down_proj.detach().clone()
    edit = RouterEdit(model)
    edit.damp_experts([(1, 2)], 0.25)
    after = model.model.layers[1].mlp.experts.down_proj
    assert torch.allclose(after[2], 0.75 * before[2])
    assert torch.equal(after[0], before[0])
    edit.restore()
    assert torch.equal(model.model.layers[1].mlp.experts.down_proj, before)


def test_routing_profile_and_ranking():
    from moe_router import profile_routing, rank_refusal_experts

    class _Tok:
        chat_template = None

        def __call__(self, texts, return_tensors=None, padding=True, add_special_tokens=False):
            ids = [[len(w) % 20 for w in t.split()] for t in texts]
            n = max(map(len, ids))
            pad = [[0] * (n - len(i)) + i for i in ids]
            mask = [[0] * (n - len(i)) + [1] * len(i) for i in ids]

            class _Enc(dict):
                def to(self, device):
                    return self

            return _Enc(ids=torch.tensor(pad), attention_mask=torch.tensor(mask))

    class _Wrapped(_ToyMoE):
        config = type("C", (), {"num_experts_per_tok": 2})()

        def forward(self, ids, attention_mask=None):
            return super().forward(ids)

    torch.manual_seed(7)
    model = _Wrapped()
    freq = profile_routing(model, _Tok(), ["a bb ccc", "dddd e"], batch_size=2, device="cpu")
    assert freq.shape == (4, E)
    assert torch.allclose(freq.sum(dim=1), torch.ones(4), atol=1e-5)  # top-k slots sum to 1
    harm = torch.tensor([[0.5, 0.1, 0.2, 0.2]])
    safe = torch.tensor([[0.2, 0.1, 0.4, 0.3]])
    assert rank_refusal_experts(harm, safe) == [(0, 0)]
