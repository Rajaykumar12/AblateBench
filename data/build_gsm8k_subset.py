"""Fixed-seed GSM8K test sample for the generative math check (src/eval_math.py).
Requires network (HF Hub) - run on Modal."""
from __future__ import annotations

import json
from pathlib import Path


def build_and_save(output_path: str | Path, num_questions: int = 200, seed: int = 0) -> None:
    import random

    from datasets import load_dataset

    from eval_math import gold_answer

    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    rows = []
    for idx in indices:
        gold = gold_answer(ds[idx]["answer"])
        if gold is None:
            continue
        rows.append({"question": ds[idx]["question"], "answer": gold})
        if len(rows) == num_questions:
            break
    Path(output_path).write_text(
        json.dumps({"source": "openai/gsm8k:main:test", "questions": rows}, indent=2)
    )
