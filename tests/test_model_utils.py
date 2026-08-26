"""Regression tests for model_utils.format_chat_prompt - added after a live
Llama-3.1-8B-Instruct run showed a ~90% judge_fallback_rate (vs. ~20-27% for Qwen on
the identical code path) traced to generation and extraction both tokenizing raw
prompt text instead of applying the model's chat template.

Also covers snapshot_residual_write_weights / restore_residual_write_weights, added
alongside norm-preserving weight ablation (ablate_weights_ mutates in place with no
other undo path)."""
import torch
import torch.nn as nn


class _FakeTokenizerWithTemplate:
    chat_template = "some non-none template string"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages == [{"role": "user", "content": "hello"}]
        return "<|template_wrapped|>hello<|assistant|>"


class _FakeTokenizerNoTemplate:
    chat_template = None


def test_format_chat_prompt_wraps_when_template_exists():
    from model_utils import format_chat_prompt

    result = format_chat_prompt(_FakeTokenizerWithTemplate(), "hello")
    assert result == "<|template_wrapped|>hello<|assistant|>"


def test_format_chat_prompt_falls_back_to_raw_when_no_template():
    from model_utils import format_chat_prompt

    result = format_chat_prompt(_FakeTokenizerNoTemplate(), "hello")
    assert result == "hello"


def test_format_chat_prompt_falls_back_when_attribute_missing_entirely():
    from model_utils import format_chat_prompt

    class _NoAttrAtAll:
        pass

    result = format_chat_prompt(_NoAttrAtAll(), "hello")
    assert result == "hello"


class _TinyDecoderLayer(nn.Module):
    """Matches model_utils' expected attribute paths (self_attn.o_proj,
    mlp.down_proj), mirroring tests/test_ablate_weights.py's fixture."""

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


def test_snapshot_restore_roundtrips_to_exact_original_tensors():
    from model_utils import (
        restore_residual_write_weights,
        snapshot_residual_write_weights,
    )

    torch.manual_seed(5)
    model = _TinyModel(hidden_dim=6, num_layers=2)

    original_weights = {
        (i, name): getattr_by_path(model.model.layers[i], name).weight.data.clone()
        for i in range(2)
        for name in ("self_attn.o_proj", "mlp.down_proj")
    }
    original_biases = {
        (i, name): getattr_by_path(model.model.layers[i], name).bias.data.clone()
        for i in range(2)
        for name in ("self_attn.o_proj", "mlp.down_proj")
    }

    snapshot = snapshot_residual_write_weights(model)

    # mutate every residual-write module's weight and bias
    for i in range(2):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            proj = getattr_by_path(model.model.layers[i], name)
            proj.weight.data.add_(1.0)
            proj.bias.data.add_(1.0)

    restore_residual_write_weights(model, snapshot)

    for i in range(2):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            proj = getattr_by_path(model.model.layers[i], name)
            assert torch.equal(proj.weight.data, original_weights[(i, name)])
            assert torch.equal(proj.bias.data, original_biases[(i, name)])


def getattr_by_path(obj, dotted_name: str):
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj
