"""MoE router-level abliteration (plan Phase 2).

Residual-stream ablation edits what experts *write*. Two lines of evidence say MoE
refusal also lives in *routing*: safety-critical experts are concentrated in a few
experts per model (L3, arXiv 2602.08741; SAFEx, arXiv 2506.17368), and per-expert
edits get undone when the router sends tokens to unedited experts (Flay, on
Qwen3-30B-A3B).

The Qwen3.5-MoE router is a bias-free `F.linear(x, W_gate)` on the
post-attention-norm output, followed by softmax and top-k. A constant per-expert
logit penalty (SteerMoE/L3-style masking) therefore cannot be stored in this
checkpoint format, since there is no bias to put it in. Two edits *can* be stored exactly:

1. Router orthogonalization: remove the refusal direction from what the router (and
   the shared-expert gate) reads, so routing no longer reacts to it. The router input
   is `g * x / rms(x)` with g the norm's effective per-channel scale, so a component
   `c * d` of the residual reaches the router as `c * (g * d) / rms`. Projecting
   normalize(g * d) out of W_gate's input side removes exactly that signal.
2. Expert damping: scale down the `down_proj` of the experts whose routing frequency
   rises most on harmful vs. harmless prompts. The router still selects them, so
   their routing mass is not handed to other experts; it just contributes less.

Both are plain weight edits. The optimizer applies them per trial with a snapshot
and restore, so what it scores is exactly what the checkpoint gets.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model_utils import get_decoder_layers


def _router(layer: nn.Module):
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate", None) if mlp is not None else None
    weight = getattr(gate, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return None, None
    return gate, weight


def is_moe(model: nn.Module) -> bool:
    return any(_router(layer)[1] is not None for layer in get_decoder_layers(model))


def _top_k(model: nn.Module, gate: nn.Module) -> int:
    k = getattr(gate, "top_k", None)
    if k is None:
        cfg = getattr(model.config, "text_config", model.config)
        k = getattr(cfg, "num_experts_per_tok", None)
    if k is None:
        raise ValueError("cannot determine the router's top-k")
    return int(k)


@torch.no_grad()
def profile_routing(
    model: nn.Module, tokenizer, prompts: list[str], batch_size: int = 8, device: str = "cuda"
) -> torch.Tensor:
    """Expert selection frequency [num_layers, num_experts]: the fraction of top-k
    slots (over all non-padding prompt tokens) that went to each expert. Uses the
    same chat formatting as direction extraction."""
    from model_utils import format_chat_prompt

    layers = get_decoder_layers(model)
    routers = [_router(layer) for layer in layers]
    if not any(w is not None for _, w in routers):
        raise ValueError("model has no MoE routers")
    num_experts = next(w.shape[0] for _, w in routers if w is not None)
    counts = torch.zeros(len(layers), num_experts, dtype=torch.float64)
    state = {"mask": None, "tokens": 0}

    def make_hook(idx: int, k: int):
        def hook(module, inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.reshape(-1, logits.shape[-1]).float()
            chosen = logits.topk(k, dim=-1).indices  # softmax is monotonic
            chosen = chosen[state["mask"]]
            counts[idx] += torch.bincount(
                chosen.flatten().cpu(), minlength=num_experts
            ).double()

        return hook

    handles = []
    k = None
    for i, (gate, weight) in enumerate(routers):
        if weight is not None:
            k = _top_k(model, gate)
            handles.append(gate.register_forward_hook(make_hook(i, k)))
    try:
        formatted = [format_chat_prompt(tokenizer, p) for p in prompts]
        for start in range(0, len(formatted), batch_size):
            enc = tokenizer(
                formatted[start : start + batch_size], return_tensors="pt",
                padding=True, add_special_tokens=False,
            ).to(device)
            mask = enc["attention_mask"].reshape(-1).bool()
            state["mask"] = mask
            state["tokens"] += int(mask.sum())
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    return (counts / (state["tokens"] * k)).float()


def rank_refusal_experts(freq_harmful: torch.Tensor, freq_harmless: torch.Tensor) -> list[tuple[int, int]]:
    """(layer, expert) pairs sorted by how much more often harmful prompts route to
    them than harmless ones (absolute frequency difference, largest first)."""
    diff = (freq_harmful - freq_harmless).flatten()
    order = torch.argsort(diff, descending=True)
    num_experts = freq_harmful.shape[1]
    return [(int(i) // num_experts, int(i) % num_experts) for i in order if diff[i] > 0]


def _router_input_basis(layer: nn.Module, basis: torch.Tensor) -> torch.Tensor | None:
    """normalize(g * q) for each basis row, with g the post-attention norm's
    effective per-channel scale (norm applied to a ones vector)."""
    from ablate_weights import orthonormal_basis

    norm = getattr(layer, "post_attention_layernorm", None)
    q = basis.float()
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if norm is not None:
        param = next(norm.parameters())
        g = norm(torch.ones(1, q.shape[-1], dtype=param.dtype, device=param.device)).float().squeeze(0)
        q = q.to(g.device) * g
    return orthonormal_basis(q)


class RouterEdit:
    """Applies router orthogonalization and expert damping in place, keeping only
    the touched tensors (router weights, shared-expert gates, damped expert slices)
    for `restore()`. Router tensors are ~0.5M params per layer, so a full snapshot
    costs ~40 MB for a 40-layer, 256-expert model."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = get_decoder_layers(model)
        self._saved: list[tuple[torch.Tensor, object, torch.Tensor]] = []

    def _save(self, tensor: torch.Tensor, index=None):
        data = tensor.data
        self._saved.append((data, index, (data if index is None else data[index]).clone()))

    @torch.no_grad()
    def orthogonalize_routers(self, basis: torch.Tensor, strengths: list[float]) -> int:
        """basis: [k, hidden] (same at every layer) or [L, k, hidden];
        strengths: per layer. Returns the number of edited router matrices."""
        edited = 0
        for i, layer in enumerate(self.layers):
            s = strengths[i]
            gate, weight = _router(layer)
            if weight is None or s == 0:
                continue
            layer_basis = basis[i] if basis.dim() == 3 else basis
            u = _router_input_basis(layer, layer_basis).to(weight.device)
            targets = [weight]
            shared = getattr(layer.mlp, "shared_expert_gate", None)
            if isinstance(getattr(shared, "weight", None), torch.Tensor):
                targets.append(shared.weight)
            for w in targets:
                self._save(w)
                wf = w.data.float()
                w.data.copy_((wf - s * (wf @ u.T) @ u).to(w.dtype))
            edited += 1
        return edited

    @torch.no_grad()
    def damp_experts(self, pairs: list[tuple[int, int]], alpha: float) -> int:
        """Scale each listed expert's down-projection by (1 - alpha)."""
        if alpha == 0:
            return 0
        done = 0
        for layer_idx, expert in pairs:
            experts = getattr(self.layers[layer_idx].mlp, "experts", None)
            if experts is None:
                continue
            if isinstance(experts, nn.ModuleList):
                w = experts[expert].down_proj.weight
                self._save(w)
                w.data.mul_(1 - alpha)
            else:
                w = experts.down_proj
                self._save(w, expert)
                w.data[expert].mul_(1 - alpha)
            done += 1
        return done

    @torch.no_grad()
    def restore(self) -> None:
        for data, index, saved in reversed(self._saved):
            if index is None:
                data.copy_(saved)
            else:
                data[index].copy_(saved)
        self._saved.clear()
