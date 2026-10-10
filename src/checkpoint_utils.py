"""Checkpoint file handling for the publish step (push_checkpoint_to_hub).

Not a PRD pipeline stage - factored into src/ so the resharding logic is unit-testable
on CPU like the rest of the pure logic, while modal_app.py only wires it to the volume.
"""
from __future__ import annotations

import json
import os

# HF Hub rejects files over 50 GB, and multi-GB single-file uploads are the ones that
# fail partway; 5 GB matches the size transformers itself has historically used.
DEFAULT_MAX_SHARD_BYTES = 5 * 1024**3


def reshard_safetensors(path: str, max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES) -> bool:
    """Rewrite a sharded safetensors checkpoint at `path` so no shard exceeds
    `max_shard_bytes`, streaming one tensor at a time (peak RAM ~ one output shard,
    never the whole model). Returns False and touches nothing if every shard already
    fits.

    Old shards are only deleted after every new shard has been written (under temp
    names), so a crash during the slow streaming phase leaves the original checkpoint
    loadable."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = os.path.join(path, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    old_files = sorted(set(index["weight_map"].values()))
    if all(os.path.getsize(os.path.join(path, f)) <= max_shard_bytes for f in old_files):
        return False

    tmp_names: list[str] = []
    tmp_weight_map: dict[str, str] = {}
    pending: dict = {}
    pending_bytes = 0

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        name = f"reshard-tmp-{len(tmp_names):05d}.safetensors"
        save_file(pending, os.path.join(path, name), metadata={"format": "pt"})
        for key in pending:
            tmp_weight_map[key] = name
        tmp_names.append(name)
        pending, pending_bytes = {}, 0

    for old in old_files:
        with safe_open(os.path.join(path, old), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                size = tensor.numel() * tensor.element_size()
                if pending and pending_bytes + size > max_shard_bytes:
                    flush()
                pending[key] = tensor
                pending_bytes += size
    flush()

    total = len(tmp_names)
    final_names = {
        tmp: f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i, tmp in enumerate(tmp_names)
    }
    for old in old_files:
        if old not in final_names.values():
            os.remove(os.path.join(path, old))
    for tmp, final in final_names.items():
        os.replace(os.path.join(path, tmp), os.path.join(path, final))

    index["weight_map"] = {k: final_names[v] for k, v in tmp_weight_map.items()}
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    return True
