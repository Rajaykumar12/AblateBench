"""Gemma 2/3 support: decoder-layer discovery under Gemma 3's multimodal wrapper, and
weight-level ablation that stays exact through sandwich norms (an RMSNorm between
each residual-write projection and the residual add)."""
import pytest
import torch
import torch.nn as nn

from ablate_weights import ablate_weights_, post_norm_adjusted_direction
from model_utils import get_decoder_layers, get_residual_write_modules_with_post_norms


class _GemmaStyleRMSNorm(nn.Module):
    """Same math as transformers' Gemma RMSNorm: scale stored as `1 + weight`."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class _SandwichLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.post_attention_layernorm = _GemmaStyleRMSNorm(hidden_dim)
        self.pre_feedforward_layernorm = _GemmaStyleRMSNorm(hidden_dim)
        self.post_feedforward_layernorm = _GemmaStyleRMSNorm(hidden_dim)


class _PreNormLayer(nn.Module):
    """Llama/Qwen layout: has a `post_attention_layernorm`, but it's the pre-MLP norm."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_layernorm = _GemmaStyleRMSNorm(hidden_dim)
        self.post_attention_layernorm = _GemmaStyleRMSNorm(hidden_dim)


def _model_of(layer: nn.Module) -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    return model


def _unit(dim: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    d = torch.randn(dim)
    return d / d.norm()


def test_sandwich_layer_pairs_each_projection_with_its_post_norm():
    layer = _SandwichLayer(8)
    pairs = get_residual_write_modules_with_post_norms(layer)
    assert pairs == [
        (layer.self_attn.o_proj, layer.post_attention_layernorm),
        (layer.mlp.down_proj, layer.post_feedforward_layernorm),
    ]


def test_pre_norm_layer_is_not_mistaken_for_sandwich():
    """Llama/Qwen's `post_attention_layernorm` reads *from* the stream (pre-MLP norm);
    treating it as a post-norm would wrongly rewrite every Qwen/Llama edit."""
    layer = _PreNormLayer(8)
    pairs = get_residual_write_modules_with_post_norms(layer)
    assert all(post_norm is None for _, post_norm in pairs)


def test_post_norm_adjusted_direction_matches_scale_times_direction():
    dim = 10
    norm = _GemmaStyleRMSNorm(dim)
    d_hat = _unit(dim, seed=1)
    expected = (1.0 + norm.weight.detach()) * d_hat
    expected = expected / expected.norm()
    assert torch.allclose(post_norm_adjusted_direction(norm, d_hat), expected, atol=1e-5)


def test_sandwich_weight_ablation_removes_direction_after_the_norm():
    """The property that matters: what actually reaches the residual stream - the
    post-norm output - has no d_hat component, for arbitrary inputs."""
    torch.manual_seed(0)
    dim = 16
    layer = _SandwichLayer(dim)
    d_hat = _unit(dim, seed=2)

    ablate_weights_(_model_of(layer), d_hat.unsqueeze(0), coefficient=1.0)

    with torch.no_grad():
        x = torch.randn(32, dim)
        attn_write = layer.post_attention_layernorm(layer.self_attn.o_proj(x))
        mlp_write = layer.post_feedforward_layernorm(layer.mlp.down_proj(x))
    assert torch.allclose(attn_write @ d_hat, torch.zeros(32), atol=1e-4)
    assert torch.allclose(mlp_write @ d_hat, torch.zeros(32), atol=1e-4)


def test_naive_ablation_would_leak_through_sandwich_norm():
    """Guard that the test above is meaningful: orthogonalizing against d_hat itself
    (the pre-norm-model math) leaves a clear d_hat component after a non-trivial norm."""
    from ablate_weights import orthogonalize_matrix

    torch.manual_seed(0)
    dim = 16
    layer = _SandwichLayer(dim)
    d_hat = _unit(dim, seed=2)
    o_proj = layer.self_attn.o_proj
    o_proj.weight.data = orthogonalize_matrix(o_proj.weight.data, d_hat)

    with torch.no_grad():
        x = torch.randn(32, dim)
        leaked = layer.post_attention_layernorm(o_proj(x)) @ d_hat
    assert leaked.abs().max() > 1e-2


def test_pre_norm_ablation_unchanged_by_sandwich_support():
    """Qwen/Llama path must be byte-for-byte the old math: projection output itself
    is orthogonal to d_hat."""
    torch.manual_seed(0)
    dim = 12
    layer = _PreNormLayer(dim)
    d_hat = _unit(dim, seed=4)

    ablate_weights_(_model_of(layer), d_hat.unsqueeze(0), coefficient=1.0)

    assert torch.allclose(d_hat @ layer.self_attn.o_proj.weight, torch.zeros(dim), atol=1e-5)
    assert torch.allclose(d_hat @ layer.mlp.down_proj.weight, torch.zeros(dim), atol=1e-5)


# --- real transformers Gemma 3 classes (tiny random configs, CPU) ---

gemma3 = pytest.importorskip("transformers.models.gemma3")


def _tiny_gemma3_text_config():
    from transformers import Gemma3TextConfig

    return Gemma3TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, sliding_window=16,
    )


def test_gemma3_multimodal_wrapper_layers_are_found():
    """google/gemma-3-27b-it loads as Gemma3ForConditionalGeneration, whose decoder
    lives at model.language_model.layers."""
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    config = Gemma3Config(text_config=_tiny_gemma3_text_config().to_dict())
    with torch.device("meta"):
        model = Gemma3ForConditionalGeneration(config)
    layers = get_decoder_layers(model)
    assert len(layers) == 2
    assert all(post_norm is not None for _, post_norm in
               get_residual_write_modules_with_post_norms(layers[0]))


def test_gemma3_real_model_post_norm_outputs_lose_direction():
    """End to end on the real Gemma3 decoder layer: after ablate_weights_, the
    tensors each layer adds into the residual stream (post-norm outputs) carry no
    d_hat component during an actual forward pass."""
    from transformers import Gemma3ForCausalLM

    torch.manual_seed(0)
    config = _tiny_gemma3_text_config()
    model = Gemma3ForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:  # Gemma inits norm weights to 0 (scale 1) - randomize
                p.normal_(0.0, 0.5)

    d_hat = _unit(config.hidden_size, seed=5)
    directions = d_hat.unsqueeze(0).expand(config.num_hidden_layers, -1)
    ablate_weights_(model, directions, coefficient=1.0)

    writes = []
    hooks = []
    for layer in get_decoder_layers(model):
        for norm in (layer.post_attention_layernorm, layer.post_feedforward_layernorm):
            hooks.append(norm.register_forward_hook(lambda m, i, o: writes.append(o)))
    with torch.no_grad():
        model(input_ids=torch.randint(0, config.vocab_size, (2, 10)))
    for h in hooks:
        h.remove()

    assert len(writes) == 2 * config.num_hidden_layers
    for w in writes:
        assert torch.allclose(w @ d_hat, torch.zeros(w.shape[:-1]), atol=1e-4)
