"""Builds a category-stratified MMLU-Pro sample for the harder capability check.

ARC-Easy saturates at 27B (1.000 before and after ablation), so it can only rule out
gross breakage. MMLU-Pro (TIGER-Lab/MMLU-Pro) has up to 10 options per question and
reasoning-heavy items, leaving headroom for an ablation-induced drop to show up.

Stratified by category (fixed seed) so all 14 subjects are represented equally and
the same questions are used for the base and ablated model. Requires network (HF
Hub) - run on Modal. No static fallback: a fallback would silently replace a hard
benchmark with an easy one, which is exactly the failure this check exists to avoid.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_mmlu_pro_subset(num_questions: int = 500, seed: int = 0) -> list[dict]:
    import random
    from collections import defaultdict

    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_category: dict[str, list[int]] = defaultdict(list)
    for idx, category in enumerate(ds["category"]):
        by_category[category].append(idx)

    rng = random.Random(seed)
    categories = sorted(by_category)
    per_category = num_questions // len(categories)
    picked: list[int] = []
    for category in categories:
        indices = by_category[category][:]
        rng.shuffle(indices)
        picked.extend(indices[:per_category])

    questions = []
    for idx in picked:
        row = ds[idx]
        questions.append({
            "question": row["question"],
            "choices": row["options"],
            "answer_index": row["answer_index"],
            "category": row["category"],
        })
    return questions


def build_and_save(output_path: str | Path, num_questions: int = 500) -> None:
    questions = load_mmlu_pro_subset(num_questions)
    Path(output_path).write_text(
        json.dumps({"source": "TIGER-Lab/MMLU-Pro:test", "questions": questions}, indent=2)
    )


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "mmlu_pro_subset.json"
    build_and_save(out)
    print(f"Wrote MMLU-Pro subset to {out}")
