"""Defensive-task scorers (src/eval_defensive.py) and the offline static-fallback path
of the defensive dataset builders (data/build_defensive_subset.py). Pure/CPU, no model,
no network - `datasets` being absent forces the builders down their fallback path,
which is exactly the offline behavior under test."""
import json

import pytest

from eval_capability import MCQuestion
from eval_defensive import (
    VULN_CHOICES,
    build_patch_prompts,
    grade_patches,
    vuln_detection_questions,
)


def test_vuln_detection_adapts_raw_rows_to_binary_mcq():
    rows = [{"code": "strcpy(b, s);", "vulnerable": True},
            {"code": "safe();", "vulnerable": False}]
    qs = vuln_detection_questions(rows)
    assert all(isinstance(q, MCQuestion) and q.choices == VULN_CHOICES for q in qs)
    assert qs[0].answer_index == 0  # vulnerable
    assert qs[1].answer_index == 1  # safe
    assert "strcpy(b, s);" in qs[0].question


def test_vuln_detection_passes_through_mcq_shaped_rows():
    rows = [{"question": "Q", "choices": ["a", "b"], "answer_index": 1}]
    qs = vuln_detection_questions(rows)
    assert qs == [MCQuestion("Q", ["a", "b"], 1)]


def test_build_patch_prompts_includes_code_and_description():
    tasks = [{"code": "os.system(x)", "description": "command injection"}]
    (prompt,) = build_patch_prompts(tasks)
    assert "os.system(x)" in prompt and "command injection" in prompt
    assert "only the corrected code" in prompt


def test_build_patch_prompts_defaults_missing_description():
    (prompt,) = build_patch_prompts([{"code": "x = 1"}])
    assert "unspecified security weakness" in prompt


def test_grade_patches_aggregates_fix_rate_and_keeps_labels():
    tasks = [{"code": "a"}, {"code": "b"}, {"code": "c"}, {"code": "d"}]
    patches = ["p1", "p2", "p3", "p4"]
    # judge marks the first two fixed, rest not
    judge = lambda task, patch: "fixed" if patch in ("p1", "p2") else "not_fixed"
    out = grade_patches(tasks, patches, judge)
    assert out["total"] == 4 and out["fixed"] == 2
    assert out["fix_rate"] == pytest.approx(0.5)
    assert out["labels"] == ["fixed", "fixed", "not_fixed", "not_fixed"]


def test_grade_patches_rejects_bad_judge_label():
    with pytest.raises(ValueError):
        grade_patches([{"code": "a"}], ["p"], lambda t, p: "maybe")


def test_grade_patches_rejects_length_mismatch():
    with pytest.raises(ValueError):
        grade_patches([{"code": "a"}], ["p1", "p2"], lambda t, p: "fixed")


# --- dataset builders: offline static-fallback path (datasets not installed) ---

def test_build_security_mcq_static_fallback_is_valid(tmp_path):
    import build_defensive_subset as b

    out = tmp_path / "sec.json"
    b.build_security_mcq(out, num_questions=5)  # no dataset_id -> static set
    data = json.loads(out.read_text())
    assert data["source"] == "static_fallback"
    for q in data["questions"]:
        assert 0 <= q["answer_index"] < len(q["choices"])
        MCQuestion(q["question"], q["choices"], q["answer_index"])  # shape check


def test_build_vuln_detection_fallback_rows_are_adaptable(tmp_path):
    import build_defensive_subset as b

    out = tmp_path / "vuln.json"
    b.build_vuln_detection(out, num_questions=4)  # datasets missing -> fallback
    data = json.loads(out.read_text())
    assert data["source"] == "static_fallback"
    qs = vuln_detection_questions(data["questions"])
    assert len(qs) == len(data["questions"]) and all(q.answer_index in (0, 1) for q in qs)


def test_build_patch_tasks_curated_has_code_and_description(tmp_path):
    import build_defensive_subset as b

    out = tmp_path / "patch.json"
    b.build_patch_tasks(out, num_tasks=3)
    data = json.loads(out.read_text())
    assert data["source"] == "curated_defensive"
    assert len(data["tasks"]) == 3
    for t in data["tasks"]:
        assert t["code"] and t["description"]  # specific description, not generic
    prompts = build_patch_prompts(data["tasks"])
    assert len(prompts) == 3 and all("corrected code" in p for p in prompts)


def test_parse_security_row_handles_cybermetric_shape():
    import build_defensive_subset as b

    row = {"question": "Q?", "answers": {"A": "wrong", "B": "right", "C": "no", "D": "nope"},
           "solution": "B"}
    parsed = b._parse_security_row(row)
    assert parsed["answer_index"] == 1 and parsed["choices"][1] == "right"


def test_parse_security_row_rejects_unparseable_rows():
    import build_defensive_subset as b

    # solution letter not among the answer keys -> drop, don't mis-map
    assert b._parse_security_row({"question": "Q", "answers": {"A": "x"}, "solution": "Z"}) is None
    assert b._parse_security_row({"answers": {"A": "x"}, "solution": "A"}) is None  # no question


def test_parse_security_row_handles_generic_letter_answer():
    import build_defensive_subset as b

    row = {"question": "Q?", "choices": ["a", "b", "c"], "answer": "C"}
    assert b._parse_security_row(row)["answer_index"] == 2
