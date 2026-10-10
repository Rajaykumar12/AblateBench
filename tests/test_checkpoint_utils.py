"""reshard_safetensors: an oversized-shard checkpoint is rewritten into bounded shards
with identical tensors and a consistent index, and a well-sharded one is untouched."""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from checkpoint_utils import reshard_safetensors


def _write_checkpoint(path, shards: dict[str, dict[str, torch.Tensor]]):
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, os.path.join(path, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in tensors})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)


def _load_all(path) -> dict[str, torch.Tensor]:
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out = {}
    for key, fname in weight_map.items():
        with safe_open(os.path.join(path, fname), framework="pt") as f:
            out[key] = f.get_tensor(key)
    return out


def test_oversized_shard_is_split_with_identical_tensors(tmp_path):
    torch.manual_seed(0)
    big = {f"layer.{i}.weight": torch.randn(64, 64) for i in range(10)}  # 16 KiB each
    small = {"lm_head.weight": torch.randn(8, 64)}
    _write_checkpoint(tmp_path, {
        "model-00001-of-00002.safetensors": small,
        "model-00002-of-00002.safetensors": big,
    })
    expected = {**big, **small}

    assert reshard_safetensors(str(tmp_path), max_shard_bytes=40 * 1024) is True

    files = sorted(f for f in os.listdir(tmp_path) if f.endswith(".safetensors"))
    assert all(os.path.getsize(tmp_path / f) <= 40 * 1024 + 4096 for f in files)  # + header
    assert len(files) > 2
    assert all(f.endswith(f"-of-{len(files):05d}.safetensors") for f in files)
    assert not any(f.startswith("reshard-tmp") for f in files)
    loaded = _load_all(tmp_path)
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[k], expected[k]) for k in expected)


def test_already_small_shards_are_left_alone(tmp_path):
    tensors = {"a": torch.randn(4, 4)}
    _write_checkpoint(tmp_path, {"model-00001-of-00001.safetensors": tensors})
    before = os.path.getmtime(tmp_path / "model-00001-of-00001.safetensors")

    assert reshard_safetensors(str(tmp_path), max_shard_bytes=1024**2) is False
    assert os.path.getmtime(tmp_path / "model-00001-of-00001.safetensors") == before
