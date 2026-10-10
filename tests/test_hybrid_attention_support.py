"""Hybrid linear/full-attention models (Qwen3.5 / 3.6 / 3.8): 3 of every 4 layers use
Gated DeltaNet linear attention, which writes into the residual stream through
`linear_attn.out_proj` rather than `self_attn.o_proj`. Weight ablation must cover it,
or those layers silently get an MLP-only edit."""
import pytest
import torch

from ablate_weights import ablate_weights_
from model_utils import get_decoder_layers, get_residual_write_modules

qwen3_5 = pytest.importorskip("transformers.models.qwen3_5")


def _tiny_qwen3_5_text_config():
    from transformers import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )


def test_linear_attention_layers_expose_out_proj_as_residual_write():
    from transformers import Qwen3_5ForCausalLM

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(_tiny_qwen3_5_text_config())
    for layer in get_decoder_layers(model):
        attn = getattr(layer, "linear_attn", None) or layer.self_attn
        attn_out = getattr(attn, "out_proj", None) or attn.o_proj
        assert get_residual_write_modules(layer) == [attn_out, layer.mlp.down_proj]


def test_hybrid_model_attention_and_mlp_writes_lose_direction():
    """Real forward pass: after ablate_weights_, every sub-block output that gets
    added into the residual stream - linear attention, full attention, MLP - has no
    d_hat component."""
    from transformers import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    config = _tiny_qwen3_5_text_config()
    model = Qwen3_5ForCausalLM(config).eval()

    d_hat = torch.randn(config.hidden_size)
    d_hat = d_hat / d_hat.norm()
    directions = d_hat.unsqueeze(0).expand(config.num_hidden_layers, -1)
    ablate_weights_(model, directions, coefficient=1.0)

    writes = {}
    hooks = []
    for i, layer in enumerate(get_decoder_layers(model)):
        attn = getattr(layer, "linear_attn", None) or layer.self_attn
        for name, module in (("attn", attn), ("mlp", layer.mlp)):
            def hook(m, inputs, output, key=(i, name)):
                writes[key] = output[0] if isinstance(output, tuple) else output
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(input_ids=torch.randint(0, config.vocab_size, (2, 10)))
    for h in hooks:
        h.remove()

    assert len(writes) == 2 * config.num_hidden_layers
    for key, w in writes.items():
        assert torch.allclose(w @ d_hat, torch.zeros(w.shape[:-1]), atol=1e-4), key
