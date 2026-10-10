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
# match any of these. `model.language_model.layers` is Gemma 3's multimodal wrapper
# (Gemma3ForConditionalGeneration, what AutoModelForCausalLM returns for
# google/gemma-3-27b-it) - its text decoder sits one level deeper than usual.
_DECODER_LAYER_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "model.language_model.layers",
)

# Sub-module names, relative to each decoder layer, whose output is written directly
# into the residual stream and is therefore a valid target for weight-level ablation.
# (embed_tokens also writes into the stream at layer 0; PRD scopes weight-level
# ablation to "each layer's output projection", so it's deliberately excluded here.)
#
# `linear_attn.out_proj`: hybrid models (Qwen3.5 / 3.6 / 3.8) use Gated DeltaNet linear
# attention in 3 of every 4 layers, which writes to the stream through `out_proj`
# instead of `self_attn.o_proj`. Without it those layers would silently get an
# MLP-only edit, since down_proj alone is enough to pass the "found something" check.
_RESIDUAL_WRITE_PROJECTIONS = (
    "self_attn.o_proj",
    "linear_attn.out_proj",
    "mlp.down_proj",
)

# "Sandwich-norm" architectures (Gemma 2, Gemma 3) apply an RMSNorm to each
# sub-block's output *before* the residual add, so the projection is not the last
# thing to touch what gets written into the stream. Maps each residual-write
# projection to the norm that follows it in those architectures.
_POST_NORM_FOR_PROJECTION = {
    "self_attn.o_proj": "post_attention_layernorm",
    "linear_attn.out_proj": "post_attention_layernorm",
    "mlp.down_proj": "post_feedforward_layernorm",
}
# Marker for a sandwich-norm layer. NOT `post_attention_layernorm` alone: Llama/Qwen
# also have a module by that name, but there it's the *pre*-MLP norm and reads from
# the stream rather than sitting between o_proj and the residual add.
_SANDWICH_NORM_MARKER = "post_feedforward_layernorm"


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
    return [proj for proj, _ in get_residual_write_modules_with_post_norms(layer)]


def get_residual_write_modules_with_post_norms(
    layer: nn.Module,
) -> list[tuple[nn.Linear, nn.Module | None]]:
    """Like get_residual_write_modules, but pairs each projection with the norm that
    sits between it and the residual add (Gemma 2/3 sandwich norms), or None for
    pre-norm architectures (Llama, Qwen, Mistral) where the projection writes into
    the stream directly. ablate_weights needs this: with a post-norm in between,
    orthogonalizing the projection against d_hat alone doesn't stop the norm's
    per-channel scale from reintroducing a d_hat component."""
    sandwich = hasattr(layer, _SANDWICH_NORM_MARKER)
    pairs = []
    for dotted_name in _RESIDUAL_WRITE_PROJECTIONS:
        obj = layer
        try:
            for part in dotted_name.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        post_norm = getattr(layer, _POST_NORM_FOR_PROJECTION[dotted_name]) if sandwich else None
        pairs.append((obj, post_norm))
    if not pairs:
        raise ValueError(
            f"Could not locate any residual-write projections on layer "
            f"{type(layer).__name__}; tried: {_RESIDUAL_WRITE_PROJECTIONS}"
        )
    return pairs


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

    `enable_thinking=False` puts hybrid-reasoning models (Qwen3 / Qwen3.5+, thinking
    on by default) in direct-answer mode. With thinking on, the 128-token generation
    budget is spent inside `<think>...</think>`, so the grader would score reasoning
    text instead of the answer, and extraction's last-token activation would sit at
    the start of a think block instead of where the model commits to a response.
    Templates that don't reference the variable ignore it (Jinja drops unused
    kwargs), so this is a no-op for Llama, Qwen2.5, Gemma.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def rewrap_hidden_state(output, new_hidden: torch.Tensor):
    """Inverse of unwrap_hidden_state — put a modified hidden state back into whatever
    shape the layer originally returned, so a forward hook can swap it in place."""
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden
