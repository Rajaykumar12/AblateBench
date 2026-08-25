"""Builds the small capability-eval subset (PRD §6.1 requirement 9: 50-100 questions).

Prefers pulling a real ARC-easy sample from HF (`allenai/ai2_arc`, config
"ARC-Easy") when network is available (i.e. on Modal); falls back to a small static
seed set otherwise so the eval_capability.py harness has something to run against in
an offline/sandboxed dev environment. The static fallback is clearly insufficient for
the real reported run — swap in the real ARC-easy sample before then.
"""
from __future__ import annotations

import json
from pathlib import Path

# Static fallback: 10 general-knowledge MCQs in the same {question, choices,
# answer_index} shape eval_capability.MCQuestion expects. Enough to smoke-test the
# harness offline; not a substitute for the real 50-100 question ARC-easy sample.
_STATIC_FALLBACK = [
    {"question": "What is the chemical symbol for water?", "choices": ["H2O", "O2", "CO2", "NaCl"], "answer_index": 0},
    {"question": "What planet is known as the Red Planet?", "choices": ["Venus", "Mars", "Jupiter", "Saturn"], "answer_index": 1},
    {"question": "How many continents are there on Earth?", "choices": ["5", "6", "7", "8"], "answer_index": 2},
    {"question": "What is the boiling point of water at sea level in Celsius?", "choices": ["90", "100", "110", "120"], "answer_index": 1},
    {"question": "Which gas do plants primarily absorb for photosynthesis?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "answer_index": 2},
    {"question": "What is the largest mammal in the world?", "choices": ["Elephant", "Blue whale", "Giraffe", "Polar bear"], "answer_index": 1},
    {"question": "What is the freezing point of water in Fahrenheit?", "choices": ["0", "32", "100", "212"], "answer_index": 1},
    {"question": "Which organ pumps blood through the human body?", "choices": ["Lungs", "Liver", "Heart", "Kidney"], "answer_index": 2},
    {"question": "What is the speed of light approximately, in km/s?", "choices": ["300", "3,000", "30,000", "300,000"], "answer_index": 3},
    {"question": "Which element has the atomic number 1?", "choices": ["Helium", "Hydrogen", "Oxygen", "Carbon"], "answer_index": 1},
]


def load_arc_easy_subset(num_questions: int = 100, seed: int = 0) -> list[dict]:
    """Real ARC-easy sample — requires network (HF Hub). Run on Modal."""
    import random

    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    questions = []
    for idx in indices:
        row = ds[idx]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        if row["answerKey"] not in labels:
            continue  # skip malformed rows rather than crash the whole build
        answer_index = labels.index(row["answerKey"])
        questions.append(
            {"question": row["question"], "choices": choices, "answer_index": answer_index}
        )
        if len(questions) >= num_questions:
            break
    return questions


def build_and_save(output_path: str | Path, num_questions: int = 100) -> None:
    output_path = Path(output_path)
    try:
        questions = load_arc_easy_subset(num_questions)
        source = "allenai/ai2_arc:ARC-Easy"
    except Exception as e:  # network unavailable, dataset load failed, etc.
        print(f"Falling back to static capability subset ({e!r})")
        questions = _STATIC_FALLBACK
        source = "static_fallback"
    output_path.write_text(json.dumps({"source": source, "questions": questions}, indent=2))


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "capability_subset.json"
    build_and_save(out)
    print(f"Wrote capability subset to {out}")
