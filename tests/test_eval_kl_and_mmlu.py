"""MMLU-Pro support in eval_capability (10 options, single-pass scoring path) and the
harmless-prompt KL check in eval_kl."""
import math

import pytest
import torch

from eval_capability import MCQuestion, score_choice_loglikelihoods, score_questions
from eval_kl import first_token_log_probs, mean_kl_divergence

transformers = pytest.importorskip("transformers")


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _CharTokenizer:
    """Char-level tokenizer, except " X" (space + letter) is one token - the shape
    of a real BPE vocab, where answer letters are single tokens."""

    chat_template = None
    vocab_size = 64

    def __init__(self, padding_side="left", multi_token_letters=False):
        self.padding_side = padding_side
        self.multi_token_letters = multi_token_letters
        self.pad_token_id = 0

    def _encode(self, text):
        if not self.multi_token_letters and len(text) == 2 and text[0] == " " and text[1].isupper():
            return [1 + (ord(text[1]) - ord("A"))]
        return [11 + (ord(ch) % 50) for ch in text]

    def __call__(self, text, return_tensors="pt", padding=False, add_special_tokens=True,
                 truncation=False):
        texts = [text] if isinstance(text, str) else text
        encoded = [self._encode(t) for t in texts]
        width = max(len(e) for e in encoded)
        ids, mask = [], []
        for e in encoded:
            pad = [self.pad_token_id] * (width - len(e))
            ids.append(pad + e if self.padding_side == "left" else e + pad)
            mask.append([0] * len(pad) + [1] * len(e) if self.padding_side == "left"
                        else [1] * len(e) + [0] * len(pad))
        return _FakeBatchEncoding(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))


def _tiny_model(seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256)
    return Qwen2ForCausalLM(config).eval()


def test_ten_option_question_formats_and_scores_all_letters():
    model, tok = _tiny_model(), _CharTokenizer()
    choices = [f"option {i}" for i in range(10)]
    scores = score_choice_loglikelihoods(model, tok, "Q?\nA. x\nAnswer:", choices, device="cpu")
    assert len(scores) == 10
    assert all(math.isfinite(s) and s <= 0 for s in scores)


def test_single_pass_path_matches_per_choice_path():
    """The single-forward fast path must give the same scores the original
    one-pass-per-choice computation gives for 1-token letters."""
    model, tok = _tiny_model(), _CharTokenizer()
    prompt, choices = "Which is right?\nAnswer:", ["a", "b", "c", "d"]
    fast = score_choice_loglikelihoods(model, tok, prompt, choices, device="cpu")

    prompt_ids = tok(prompt)["input_ids"]
    expected = []
    with torch.no_grad():
        for i in range(len(choices)):
            letter = tok(f" {'ABCD'[i]}")["input_ids"]
            full = torch.cat([prompt_ids, letter], dim=1)
            log_probs = torch.log_softmax(model(input_ids=full).logits[0, :-1].float(), dim=-1)
            expected.append(log_probs[-1, letter[0, 0]].item())
    assert fast == pytest.approx(expected, abs=1e-5)


def test_multi_token_letters_fall_back_to_per_choice_path():
    model = _tiny_model()
    scores = score_choice_loglikelihoods(
        model, _CharTokenizer(multi_token_letters=True), "Q?\nAnswer:", ["a", "b", "c"],
        device="cpu",
    )
    assert len(scores) == 3 and all(math.isfinite(s) for s in scores)


def test_score_questions_reports_correctness_per_question_in_order():
    model, tok = _tiny_model(), _CharTokenizer()
    prompt_choices = ["p", "q", "r", "s"]
    scores = score_choice_loglikelihoods(
        model, tok, "Pick one\nA. p\nB. q\nC. r\nD. s\nAnswer:", prompt_choices, device="cpu"
    )
    best = max(range(4), key=lambda i: scores[i])
    worst = min(range(4), key=lambda i: scores[i])
    questions = [MCQuestion("Pick one", prompt_choices, best),
                 MCQuestion("Pick one", prompt_choices, worst)]
    assert score_questions(model, tok, questions, device="cpu") == [True, False]


def test_kl_is_zero_for_identical_distributions_and_positive_otherwise():
    torch.manual_seed(0)
    a = torch.log_softmax(torch.randn(5, 64), dim=-1)
    b = torch.log_softmax(torch.randn(5, 64), dim=-1)
    assert mean_kl_divergence(a, a) == pytest.approx(0.0, abs=1e-7)
    assert mean_kl_divergence(a, b) > 0


def test_kl_matches_hand_computed_value():
    p = torch.tensor([[0.5, 0.5]]).log()
    q = torch.tensor([[0.9, 0.1]]).log()
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert mean_kl_divergence(p, q) == pytest.approx(expected, rel=1e-6)


def test_kl_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        mean_kl_divergence(torch.zeros(2, 4), torch.zeros(3, 4))


def test_first_token_log_probs_batched_with_left_padding_matches_unbatched():
    """Different-length prompts in one batch must give each prompt's own last-token
    distribution - the property left padding exists to guarantee."""
    model, tok = _tiny_model(), _CharTokenizer()
    prompts = ["hi", "a much longer prompt here", "mid length"]
    batched = first_token_log_probs(model, tok, prompts, device="cpu", batch_size=3)
    single = torch.cat([first_token_log_probs(model, tok, [p], device="cpu") for p in prompts])
    assert batched.shape == (3, 64)
    assert torch.allclose(batched, single, atol=1e-4)


def test_first_token_log_probs_rejects_right_padding():
    with pytest.raises(ValueError):
        first_token_log_probs(_tiny_model(), _CharTokenizer(padding_side="right"), ["x"], device="cpu")
