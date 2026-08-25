"""Shared model-introspection helpers used across the pipeline.

Not one of the named pipeline-stage modules in the PRD's repo layout, but factored out
because every stage (extraction, both ablation methods, the sweep) needs to find a
model's decoder-layer list and normalize its forward-hook output shape, and duplicating
that per-file would violate the "reuse what exists" convention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Attribute paths where common causal-LM architectures (Llama, Qwen2, Mistral, Gemma,
# ...) keep their stack of decoder layers. Extend this list if a future model doesn't
# match any of these.
_DECODER_LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers")

# Sub-module names, relative to each decoder layer, whose output is written directly
# into the residual stream and is therefore a valid target for weight-level ablation.
# (embed_tokens also writes into the stream at layer 0; PRD scopes weight-level
# ablation to "each layer's output projection", so it's deliberately excluded here.)
_RESIDUAL_WRITE_PROJECTIONS = (
    "self_attn.o_proj",
    "mlp.down_proj",
)


def get_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return the ordered list of decoder-layer modules for a HF causal LM.

    Raises ValueError if none of the known attribute paths match — that means the
    model's architecture isn't one of the ones this pipeline has been validated
    against, and a new path needs to be added to _DECODER_LAYER_PATHS.
    """
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


def get_residual_write_modules(layer: nn.Module) -> list[nn.Linear]:
    """Return the Linear sub-modules of one decoder layer whose output feeds directly
    into the residual stream (attention output projection, MLP down projection)."""
    modules = []
    for dotted_name in _RESIDUAL_WRITE_PROJECTIONS:
        obj = layer
        try:
            for part in dotted_name.split("."):
                obj = getattr(obj, part)
            modules.append(obj)
        except AttributeError:
            continue
    if not modules:
        raise ValueError(
            f"Could not locate any residual-write projections on layer "
            f"{type(layer).__name__}; tried: {_RESIDUAL_WRITE_PROJECTIONS}"
        )
    return modules


def unwrap_hidden_state(output) -> torch.Tensor:
    """Decoder layers in HF transformers return either a bare tensor or a tuple whose
    first element is the hidden state (subsequent elements are attention weights /
    cache, depending on config). Normalize to just the tensor."""
    if isinstance(output, tuple):
        return output[0]
    return output


def rewrap_hidden_state(output, new_hidden: torch.Tensor):
    """Inverse of unwrap_hidden_state — put a modified hidden state back into whatever
    shape the layer originally returned, so a forward hook can swap it in place."""
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden
