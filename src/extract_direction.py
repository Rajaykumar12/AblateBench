"""Stage [2]+[3]: activation extraction and per-layer refusal-direction computation.

Difference-in-means technique from Arditi et al., "Refusal in Language Models Is
Mediated by a Single Direction" (arXiv:2406.11717). For each decoder layer, the
direction is the (unit-normalized) difference between the mean last-token
residual-stream activation on harmful prompts and on harmless prompts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model_utils import get_decoder_layers, unwrap_hidden_state


@dataclass
class ActivationSet:
    """Per-layer last-token residual-stream activations for one prompt set.

    activations: tensor of shape [num_layers, num_prompts, hidden_dim].
    """

    activations: torch.Tensor

    @property
    def num_layers(self) -> int:
        return self.activations.shape[0]

    @property
    def hidden_dim(self) -> int:
        return self.activations.shape[2]


@torch.no_grad()
def extract_activations(
    model: nn.Module,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 8,
) -> ActivationSet:
    """Run a forward pass per prompt (batched), hook every decoder layer, and capture
    the last-token residual-stream activation at each layer.

    Single forward pass per prompt, no backward pass — this is what keeps the whole
    pipeline cheap even at 30B+ scale (see PRD §6.2, Compute placement).
    """
    layers = get_decoder_layers(model)
    num_layers = len(layers)

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            hidden = unwrap_hidden_state(output)  # [batch, seq, hidden]
            captured[layer_idx] = hidden[:, -1, :].detach().to("cpu", dtype=torch.float32)

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]

    all_layer_acts: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            model(**enc)
            for layer_idx in range(num_layers):
                all_layer_acts[layer_idx].append(captured[layer_idx])
    finally:
        for h in handles:
            h.remove()

    # [num_layers, num_prompts, hidden_dim]
    stacked = torch.stack([torch.cat(acts, dim=0) for acts in all_layer_acts], dim=0)
    return ActivationSet(activations=stacked)


def compute_directions(harmful: ActivationSet, harmless: ActivationSet) -> torch.Tensor:
    """Per-layer difference-in-means direction, unit-normalized.

    Returns tensor of shape [num_layers, hidden_dim], each row a unit vector (or the
    zero vector, in the degenerate case both classes' means coincide exactly).
    """
    if harmful.num_layers != harmless.num_layers:
        raise ValueError(
            f"Layer count mismatch: harmful has {harmful.num_layers}, "
            f"harmless has {harmless.num_layers}"
        )
    diff = harmful.activations.mean(dim=1) - harmless.activations.mean(dim=1)  # [layers, hidden]
    norms = diff.norm(dim=-1, keepdim=True)
    # Avoid divide-by-zero on a degenerate layer; leave it as the zero vector so
    # downstream ablation is a no-op there rather than producing NaNs.
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    return diff / safe_norms
