"""Builds the harmful/harmless train/held-out prompt splits (PRD §6.1, requirement 1)."""
from __future__ import annotations

import json
import random
from pathlib import Path

TRAIN_FRACTION = 0.7
SEED = 0


def load_advbench_harmful(limit: int | None = None) -> list[str]:
    """AdvBench harmful-behaviors set via `walledai/AdvBench`."""
    from datasets import load_dataset

    ds = load_dataset("walledai/AdvBench", split="train")
    prompts = [row["prompt"] for row in ds if row.get("prompt")]
    if limit:
        prompts = prompts[:limit]
    return prompts


def load_enriched_harmful(limit: int | None = None) -> list[str]:
    """Multi-style harmful set (`Bahushruth/abliteration-harmful-enriched`, 35 categories
    x 10 styles). Falls back to AdvBench if unavailable."""
    from datasets import load_dataset

    try:
        ds = load_dataset("Bahushruth/abliteration-harmful-enriched", split="train")
        prompts = [row["text"] for row in ds if row.get("text")]
    except (ConnectionError, ImportError, OSError, ValueError) as exc:
        print(
            "WARNING: could not load the enriched harmful dataset; "
            f"falling back to AdvBench ({exc})"
        )
        prompts = load_advbench_harmful(limit=None)

    if limit:
        prompts = prompts[:limit]
    return prompts


def load_alpaca_harmless(limit: int | None = None) -> list[str]:
    """Alpaca instructions with no input field."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = [row["instruction"] for row in ds if not row.get("input")]
    if limit:
        prompts = prompts[:limit]
    return prompts


def split_train_heldout(prompts: list[str], seed: int = SEED) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = prompts[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * TRAIN_FRACTION)
    return shuffled[:cut], shuffled[cut:]


def build_and_save(
    output_dir: str | Path,
    num_harmful: int = 300,
    num_harmless: int = 300,
    use_enriched: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harmful = (
        load_enriched_harmful(limit=num_harmful)
        if use_enriched
        else load_advbench_harmful(limit=num_harmful)
    )
    harmless = load_alpaca_harmless(limit=num_harmless)

    harmful_train, harmful_heldout = split_train_heldout(harmful)
    harmless_train, harmless_heldout = split_train_heldout(harmless)

    assert not set(harmful_train) & set(harmful_heldout)
    assert not set(harmless_train) & set(harmless_heldout)

    (output_dir / "harmful_prompts_train.json").write_text(json.dumps(harmful_train, indent=2))
    (output_dir / "harmful_prompts_heldout.json").write_text(json.dumps(harmful_heldout, indent=2))
    (output_dir / "harmless_prompts_train.json").write_text(json.dumps(harmless_train, indent=2))
    (output_dir / "harmless_prompts_heldout.json").write_text(json.dumps(harmless_heldout, indent=2))


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "."
    build_and_save(out)
    print(f"Wrote prompt splits to {out}")