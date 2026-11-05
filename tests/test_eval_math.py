"""GSM8K-style generative math scoring (plan Phase 0)."""
from eval_math import extract_answer, gold_answer, is_correct, score_responses


def test_gold_answer_parses_gsm8k_solution():
    assert gold_answer("She has 3 + 4 = 7 apples.\n#### 7") == 7.0
    assert gold_answer("... #### 1,250") == 1250.0
    assert gold_answer("no marker") is None


def test_extract_answer_prefers_the_last_answer_line():
    assert extract_answer("First 3, then 5.\nAnswer: 12") == 12.0
    assert extract_answer("Answer: 4. Wait, recheck. Answer: $1,024") == 1024.0
    assert extract_answer("so the total is 42") == 42.0
    assert extract_answer("no digits") is None


def test_scoring():
    assert is_correct("Answer: 7.0", 7.0)
    assert not is_correct("Answer: 8", 7.0)
    assert score_responses(["Answer: 1", "Answer: 3"], [1.0, 2.0]) == {"accuracy": 0.5, "num_questions": 2}
