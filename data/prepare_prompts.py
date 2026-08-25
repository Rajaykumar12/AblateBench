"""Builds the harmful/harmless train/held-out prompt splits (PRD §6.1, requirement 1).

Requires network access to HuggingFace Hub — run this on Modal (see modal_app.py's
`prepare_data` function), not in a sandboxed/offline environment. Harmful prompts come
from AdvBench; harmless contrast prompts from Alpaca, per the source technique
(Arditi et al.) and PRD §6.1/§7.

The harmful prompt set exists solely to compute/validate the refusal direction and
measure refusal rate — see PRD §4 (Non-Goals) and §6.2 (safety handling).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TRAIN_FRACTION = 0.7
SEED = 0


def load_advbench_harmful(limit: int | None = None) -> list[str]:
    """AdvBench harmful-behaviors set, via the `walledai/AdvBench` HF dataset (a
    commonly-used mirror of the original llm-attacks harmful_behaviors.csv)."""
    from datasets import load_dataset

    ds = load_dataset("walledai/AdvBench", split="train")
    prompts = [row["prompt"] for row in ds]
    if limit:
        prompts = prompts[:limit]
    return prompts


def load_alpaca_harmless(limit: int | None = None) -> list[str]:
    """Alpaca instructions with no input field, so each is a self-contained prompt
    comparable in form to the AdvBench harmful instructions."""
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


def build_and_save(output_dir: str | Path, num_harmful: int = 300, num_harmless: int = 300) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harmful = load_advbench_harmful(limit=num_harmful)
    harmless = load_alpaca_harmless(limit=num_harmless)

    harmful_train, harmful_heldout = split_train_heldout(harmful)
    harmless_train, harmless_heldout = split_train_heldout(harmless)

    # Assert-no-overlap check per PRD §6.1 requirement 1 ("never overlap").
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
