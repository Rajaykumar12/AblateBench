import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_model_card import render_card


def _results():
    return {
        "baseline_id": "Qwen/Qwen3.5-35B-A3B",
        "abliterated_id": "you/Qwen3.5-35B-A3B-abliterated",
        "refusal_by_category": {
            "baseline": {"malware": 1.0, "fraud": 0.9, "aggregate": 0.95},
            "abliterated": {"malware": 0.2, "fraud": 0.1, "aggregate": 0.15},
            "judge": "fixed external judge (Qwen/Qwen2.5-7B-Instruct)",
            "caveat": "Comparable across the A/B.",
        },
        "mmlu_pro": {
            "num_questions": 500,
            "baseline": {"accuracy": 0.7, "by_category": {"math": 0.6}},
            "abliterated": {"accuracy": 0.68, "by_category": {"math": 0.58}},
        },
        "defensive": {
            "security_mcq": {"baseline": {"accuracy": 0.8}, "abliterated": {"accuracy": 0.79}},
            "vuln_detection": {"baseline": {"accuracy": 0.6}, "abliterated": {"accuracy": 0.61}},
            "patch_synthesis": {
                "baseline": {"fix_rate": 0.5},
                "abliterated": {"fix_rate": 0.48},
                "judge": "fixed external judge",
            },
        },
        "kl_harmless": {
            "num_prompts": 100,
            "mean_kl_base_to_abliterated_nats": 0.0123,
            "first_token_top1_agreement": 0.97,
        },
    }


def test_card_has_frontmatter_and_all_benchmark_rows():
    card = render_card(_results(), "you/Qwen3.5-35B-A3B-abliterated", 36, 0.7, purified=True)
    assert card.startswith("---\nbase_model: Qwen/Qwen3.5-35B-A3B\n")
    assert "| malware | 100.0% | 20.0% | -80.0 pp |" in card
    assert "| **aggregate** | 95.0% | 15.0% | -80.0 pp |" in card
    assert "| **overall** | 70.0% | 68.0% | -2.0 pp |" in card
    assert "| Patch synthesis (fix rate) | 50.0% | 48.0% | -2.0 pp |" in card
    assert "0.0123 nats" in card
    assert "purified" in card


def test_missing_category_renders_na():
    results = _results()
    del results["refusal_by_category"]["abliterated"]["fraud"]
    card = render_card(results, "you/m", 36, 0.7, purified=False)
    assert "| fraud | 90.0% | n/a | n/a |" in card
