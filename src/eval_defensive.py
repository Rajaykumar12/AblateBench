"""Defensive-security scorers for the baseline-vs-abliterated comparison
(modal_app.compare_models). All three tasks are defensive — no exploit construction.

  * security MCQ            -> reuses eval_capability.MCQuestion / score_questions as-is
  * vulnerability detection -> classify code as vulnerable/safe, scored as a binary MCQ
  * patch synthesis         -> model rewrites vulnerable code; a judge grades fix/no-fix

The two MCQ-style tasks reuse eval_capability's deterministic log-likelihood path
verbatim (this module only adapts labeled code rows into MCQuestion form). Patch
synthesis needs free-form generation plus a judge: this module holds the pure,
CPU-testable pieces (prompt construction + grade aggregation), while the generation
and the judge model live in modal_app (per the self-judge design used elsewhere).
"""
from __future__ import annotations

from typing import Callable, Iterable

from eval_capability import MCQuestion

# answer_index 0 == vulnerable, 1 == safe. Kept as a module constant so the scorer and
# any analysis of its outputs agree on the labeling.
VULN_CHOICES = [
    "The code contains a security vulnerability",
    "The code is secure",
]
VULN_ANSWER_VULNERABLE = 0
VULN_ANSWER_SAFE = 1


def vuln_detection_questions(rows: Iterable[dict]) -> list[MCQuestion]:
    """Adapt labeled code rows into binary-MCQ questions so vulnerability detection is
    scored by the exact same log-likelihood path as the capability/MMLU evals.

    Accepts either raw rows ({"code", "vulnerable"}) or rows already in MCQuestion
    shape ({"question", "choices", "answer_index"})."""
    questions: list[MCQuestion] = []
    for row in rows:
        if "choices" in row and "answer_index" in row:
            questions.append(MCQuestion(row["question"], list(row["choices"]), row["answer_index"]))
            continue
        code = row["code"]
        vulnerable = bool(row["vulnerable"])
        question = (
            "Analyze the following code for security vulnerabilities.\n\n"
            f"```\n{code}\n```\n\n"
            "Which statement is correct?"
        )
        answer = VULN_ANSWER_VULNERABLE if vulnerable else VULN_ANSWER_SAFE
        questions.append(MCQuestion(question, list(VULN_CHOICES), answer))
    return questions


# Free-form fix-writing prompt. Defensive: asks for a corrected version of code that is
# already known-vulnerable, not for an exploit.
PATCH_PROMPT = (
    "The following code contains a security vulnerability: {description}.\n\n"
    "```\n{code}\n```\n\n"
    "Rewrite the code to fix the vulnerability while keeping its intended behavior. "
    "Return only the corrected code."
)


def build_patch_prompts(tasks: Iterable[dict]) -> list[str]:
    """One fix-writing prompt per task ({"code", "description"})."""
    return [
        PATCH_PROMPT.format(
            description=t.get("description", "an unspecified security weakness"),
            code=t["code"],
        )
        for t in tasks
    ]


def grade_patches(
    tasks: list[dict],
    patches: list[str],
    judge_fn: Callable[[dict, str], str],
) -> dict:
    """Grade each generated patch with judge_fn(task, patch) -> "fixed" | "not_fixed".
    Returns the fix rate plus the raw per-task labels so the generations can be
    inspected for coherence (a near-0 or near-1 rate is meaningless if the output is
    degenerate)."""
    if len(tasks) != len(patches):
        raise ValueError(f"tasks/patches length mismatch: {len(tasks)} vs {len(patches)}")
    labels: list[str] = []
    for task, patch in zip(tasks, patches):
        label = judge_fn(task, patch)
        if label not in ("fixed", "not_fixed"):
            raise ValueError(f"judge_fn must return 'fixed' or 'not_fixed', got {label!r}")
        labels.append(label)
    fixed = sum(1 for lbl in labels if lbl == "fixed")
    total = len(labels)
    return {
        "total": total,
        "fixed": fixed,
        "fix_rate": fixed / total if total else 0.0,
        "labels": labels,
    }
