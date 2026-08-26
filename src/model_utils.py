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


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Wrap a raw prompt in the model's chat template so an instruct model is asked
    as an actual chat turn, not asked to continue raw text as if it were a base
    model. This matters for *every* stage that feeds a prompt to the model:

    - extraction (extract_direction.py): the "last token" residual-stream activation
      should be captured right after the chat template opens the assistant turn -
      the position where the model is about to decide how to respond, which is
      where a refusal-mediating direction should be most salient. Capturing it at
      the end of raw, un-templated instruction text is a different (and weaker)
      signal.
    - generation and judging (modal_app.py): an un-templated prompt can produce text
      that doesn't look like a normal assistant response at all, especially for
      models with a strict chat format (found via a live Llama-3.1-8B-Instruct run
      where judge_fallback_rate hit ~90%, vs. ~20-27% for Qwen on the identical
      code path, before this fix).

    Falls back to the raw prompt if the tokenizer has no chat template (e.g. a base
    model). Callers must tokenize the result with `add_special_tokens=False` - the
    template's output already contains BOS/header tokens.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def snapshot_residual_write_weights(model: nn.Module) -> dict[int, dict]:
    """Clone every decoder layer's residual-write module weights/biases (see
    _RESIDUAL_WRITE_PROJECTIONS / get_residual_write_modules), so ablate_weights_'s
    in-place edit can later be undone with restore_residual_write_weights.

    ablate_weights_ mutates `proj.weight.data` (and `.bias.data`) in place with no
    other undo path — this exists so run_full_pipeline can test plain vs.
    norm-preserving weight ablation against the same originally-loaded checkpoint
    without an expensive reload. Keyed by `id(module)` rather than a name/path so
    restore doesn't depend on re-deriving the same lookup path (it must be applied to
    the *same* model instance the snapshot was taken from).
    """
    snapshot = {}
    for layer in get_decoder_layers(model):
        for proj in get_residual_write_modules(layer):
            snapshot[id(proj)] = {
                "weight": proj.weight.data.clone(),
                "bias": proj.bias.data.clone() if proj.bias is not None else None,
            }
    return snapshot


def restore_residual_write_weights(model: nn.Module, snapshot: dict[int, dict]) -> None:
    """Inverse of snapshot_residual_write_weights: copy the snapshotted tensors back
    in place. Must be called on the same model instance the snapshot was taken from."""
    for layer in get_decoder_layers(model):
        for proj in get_residual_write_modules(layer):
            saved = snapshot[id(proj)]
            proj.weight.data.copy_(saved["weight"])
            if proj.bias is not None:
                proj.bias.data.copy_(saved["bias"])


def rewrap_hidden_state(output, new_hidden: torch.Tensor):
    """Inverse of unwrap_hidden_state — put a modified hidden state back into whatever
    shape the layer originally returned, so a forward hook can swap it in place."""
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden
