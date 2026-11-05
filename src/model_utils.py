"""Shared model-introspection helpers used across the pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn

_DECODER_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "model.model.language_model.layers",
    "language_model.model.layers",
    "transformer.h",
    "gpt_neox.layers",
)

_RESIDUAL_WRITE_PROJECTIONS = (
    "self_attn.o_proj",
    "linear_attn.out_proj",
    "mlp.down_proj",
    "mlp.shared_expert.down_proj",
    "mlp.shared_experts.down_proj",
)

_POST_NORM_FOR_PROJECTION = {
    "self_attn.o_proj": "post_attention_layernorm",
    "linear_attn.out_proj": "post_attention_layernorm",
    "mlp.down_proj": "post_feedforward_layernorm",
    "mlp.shared_expert.down_proj": "post_feedforward_layernorm",
    "mlp.shared_experts.down_proj": "post_feedforward_layernorm",
}

_SANDWICH_NORM_MARKER = "post_feedforward_layernorm"

# Which sub-block each residual write belongs to, so attention and MLP/expert writes
# can be ablated with separate per-layer weights (ablate_weights_ attn/mlp weights).
_ATTENTION_PROJECTIONS = {"self_attn.o_proj", "linear_attn.out_proj"}


def get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return the ordered list of decoder-layer modules for a HF causal LM."""
    for path in _DECODER_LAYER_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            layers = list(obj)
            if layers:
                return layers
        except AttributeError:
            continue
    raise ValueError(
        f"Could not locate decoder layers on {type(model).__name__}; "
        f"tried paths: {_DECODER_LAYER_PATHS}"
    )


def get_residual_write_modules(layer: nn.Module) -> list[nn.Module | nn.Parameter]:
    """Return the sub-modules/parameters of one decoder layer whose output feeds
    directly into the residual stream."""
    return [proj for proj, _ in get_residual_write_modules_with_post_norms(layer)]


def get_residual_write_modules_with_post_norms(
    layer: nn.Module,
) -> list[tuple[nn.Module | nn.Parameter, nn.Module | None]]:
    """Pairs each residual-write projection with the norm that sits between it and
    the residual add (Gemma 2/3 sandwich norms), or None for pre-norm architectures.
    Handles standard 2D projections, linear_attn, ModuleList MoEs, and 3D packed MoEs.
    """
    return [(proj, norm) for proj, norm, _ in get_residual_writes_by_kind(layer)]


def get_residual_writes_by_kind(
    layer: nn.Module,
) -> list[tuple[nn.Module | nn.Parameter, nn.Module | None, str]]:
    """Like get_residual_write_modules_with_post_norms, plus each write's kind:
    "attn" (attention / linear-attention output) or "mlp" (dense MLP, shared expert,
    routed experts)."""
    sandwich = hasattr(layer, _SANDWICH_NORM_MARKER)
    pairs: list[tuple[nn.Module | nn.Parameter, nn.Module | None, str]] = []

    for dotted_name in _RESIDUAL_WRITE_PROJECTIONS:
        obj = layer
        try:
            for part in dotted_name.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        post_norm = (
            getattr(layer, _POST_NORM_FOR_PROJECTION[dotted_name], None)
            if sandwich
            else None
        )
        kind = "attn" if dotted_name in _ATTENTION_PROJECTIONS else "mlp"
        pairs.append((obj, post_norm, kind))

    # Handle MoE routed experts (both ModuleList of Linear and packed 3D Parameter)
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        experts = layer.mlp.experts
        mlp_post_norm = getattr(layer, "post_feedforward_layernorm", None) if sandwich else None

        if isinstance(experts, nn.ModuleList):
            for exp in experts:
                if hasattr(exp, "down_proj") and isinstance(exp.down_proj, nn.Module):
                    pairs.append((exp.down_proj, mlp_post_norm, "mlp"))
        else:
            for name, param in experts.named_parameters(recurse=False):
                if "down_proj" in name and param.dim() == 3:
                    pairs.append((param, mlp_post_norm, "mlp"))

    if not pairs:
        raise ValueError(
            f"Could not locate any residual-write projections on layer "
            f"{type(layer).__name__}; tried: {_RESIDUAL_WRITE_PROJECTIONS}"
        )
    return pairs


def get_attention_module(layer: nn.Module) -> nn.Module | None:
    """The attention block (full or linear) whose output is that layer's attention
    residual write, or None."""
    for name in ("self_attn", "linear_attn"):
        mod = getattr(layer, name, None)
        if isinstance(mod, nn.Module):
            return mod
    return None


def has_sandwich_norms(layer: nn.Module) -> bool:
    return hasattr(layer, _SANDWICH_NORM_MARKER)


def unwrap_hidden_state(output) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def rewrap_hidden_state(output, new_hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


def format_chat_prompt(tokenizer, prompt: str, enable_thinking: bool = False) -> str:
    """Chat-template one user turn. Thinking is off by default everywhere (extraction,
    sweep, optimizer, compare); `enable_thinking=True` is only for the compare stage's
    separate thinking-mode refusal check."""
    if getattr(tokenizer, "chat_template", None) is None:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )