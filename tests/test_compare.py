"""Wiring + aggregation for the two-model comparison (modal_app._eval_battery and the
cross-model KL/top-1 computation compare_models does). The model-dependent calls
(_generate, the judges, score_questions, first_token_log_probs) are monkeypatched so
this runs on CPU with no checkpoint load and no network - the logic under test is how
the battery assembles its metrics dict and how KL compares two models."""
import pytest
import torch

pytest.importorskip("modal")  # modal_app builds a modal.App at import time

import modal_app  # noqa: E402
from eval_capability import MCQuestion  # noqa: E402
from eval_defensive import vuln_detection_questions  # noqa: E402


@pytest.fixture
def battery_kwargs():
    return dict(
        category_prompts={"cyber": ["p1", "p2"], "malware": ["p3"]},
        mmlu_questions=[MCQuestion("q", ["a", "b"], 0) for _ in range(4)],
        mmlu_categories=["math", "math", "bio", "bio"],
        security_questions=[MCQuestion("q", ["a", "b"], 0) for _ in range(2)],
        security_categories=["sec", "sec"],
        vuln_questions=vuln_detection_questions(
            [{"code": "x", "vulnerable": True}, {"code": "y", "vulnerable": False}]
        ),
        patch_tasks=[{"code": "a", "description": "d"}, {"code": "b"}],
        harmless=["h1", "h2", "h3"],
        max_new_tokens=16,
    )


def _patch_model_calls(monkeypatch, log_probs):
    import eval_capability
    import eval_kl

    # refusals: a clear rule-based refusal so no judge is needed
    monkeypatch.setattr(modal_app, "_generate",
                        lambda m, t, prompts, max_new_tokens=128: ["I cannot help with that."] * len(prompts))
    monkeypatch.setattr(modal_app, "_make_self_judge", lambda m, t: (lambda text: "comply"))
    monkeypatch.setattr(modal_app, "_make_patch_fix_judge", lambda m, t: (lambda task, patch: "fixed"))
    # deterministic MCQ correctness: alternating T/F -> 0.5 accuracy everywhere
    monkeypatch.setattr(eval_capability, "score_questions",
                        lambda m, t, questions, device="cuda": [i % 2 == 0 for i in range(len(questions))])
    monkeypatch.setattr(eval_kl, "first_token_log_probs",
                        lambda m, t, prompts, device="cuda": log_probs)


def test_eval_battery_assembles_all_metric_blocks(monkeypatch, battery_kwargs):
    torch.manual_seed(0)
    log_probs = torch.log_softmax(torch.randn(3, 8), dim=-1)
    _patch_model_calls(monkeypatch, log_probs)

    metrics, returned_lp = modal_app._eval_battery(model=None, tokenizer=None, **battery_kwargs)

    assert metrics["refusal_by_category"] == {"cyber": 1.0, "malware": 1.0}
    assert metrics["refusal_aggregate"] == 1.0
    assert metrics["mmlu_pro"]["accuracy"] == pytest.approx(0.5)
    assert metrics["mmlu_pro"]["by_category"] == {"bio": pytest.approx(0.5), "math": pytest.approx(0.5)}
    assert metrics["defensive"]["security_mcq"]["accuracy"] == pytest.approx(0.5)
    assert metrics["defensive"]["vuln_detection"] == {"accuracy": pytest.approx(0.5), "num_questions": 2}
    assert metrics["defensive"]["patch_synthesis"] == {"fix_rate": 1.0, "fixed": 2, "total": 2}
    assert len(metrics["sample_patches"]) == 2
    assert set(metrics["sample_patches"][0]) == {"description", "patch", "grade"}
    assert torch.equal(returned_lp, log_probs)


def test_eval_battery_uses_injected_external_judges(monkeypatch, battery_kwargs):
    """When refusal_judge/patch_judge are passed, the battery must use them and never
    fall back to the model's own self-judge (the whole point of --judge-id)."""
    import eval_capability
    import eval_kl

    torch.manual_seed(0)
    log_probs = torch.log_softmax(torch.randn(3, 8), dim=-1)
    # Ambiguous generations so refusal grading escalates to the judge tier.
    monkeypatch.setattr(modal_app, "_generate",
                        lambda m, t, prompts, max_new_tokens=128: ["The weather is mild today."] * len(prompts))
    monkeypatch.setattr(eval_capability, "score_questions",
                        lambda m, t, questions, device="cuda": [True] * len(questions))
    monkeypatch.setattr(eval_kl, "first_token_log_probs",
                        lambda m, t, prompts, device="cuda": log_probs)
    # Self-judge factories must NOT be called when external judges are supplied.
    def _boom(*a, **k):
        raise AssertionError("self-judge factory called despite injected external judge")
    monkeypatch.setattr(modal_app, "_make_self_judge", _boom)
    monkeypatch.setattr(modal_app, "_make_patch_fix_judge", _boom)

    ext_refusal = lambda text: "refuse"
    ext_patch = lambda task, patch: "not_fixed"
    metrics, _ = modal_app._eval_battery(
        model=None, tokenizer=None, refusal_judge=ext_refusal, patch_judge=ext_patch,
        **battery_kwargs,
    )
    # External refusal judge marked everything "refuse"; external patch judge "not_fixed".
    assert metrics["refusal_aggregate"] == 1.0
    assert metrics["judge_fallback_rate"] == 1.0  # all ambiguous -> judge tier
    assert metrics["defensive"]["patch_synthesis"]["fix_rate"] == 0.0


def test_mmlu_report_overall_and_per_category():
    report = modal_app._mmlu_report([True, False, True, True], ["x", "x", "y", "y"])
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["by_category"] == {"x": pytest.approx(0.5), "y": pytest.approx(1.0)}


def test_mmlu_report_handles_empty():
    assert modal_app._mmlu_report([], [])["accuracy"] == 0.0


def test_cross_model_kl_and_top1_agreement():
    """The compare_models KL/top-1 step: identical distributions -> KL 0, full top-1
    agreement; a reordered distribution -> positive KL and reduced agreement."""
    from eval_kl import mean_kl_divergence

    torch.manual_seed(1)
    base = torch.log_softmax(torch.randn(6, 16), dim=-1)
    assert mean_kl_divergence(base, base) == pytest.approx(0.0, abs=1e-7)
    top1_same = (base.argmax(-1) == base.argmax(-1)).float().mean().item()
    assert top1_same == 1.0

    shuffled = base[:, torch.randperm(16)]
    assert mean_kl_divergence(base, shuffled) > 0
    top1_diff = (base.argmax(-1) == shuffled.argmax(-1)).float().mean().item()
    assert top1_diff < 1.0
