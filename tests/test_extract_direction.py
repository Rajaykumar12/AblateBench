"""PRD §9.1: direction computation correctness — given synthetic activations with a
known mean difference, verify d_layer matches the expected difference-of-means and is
unit-norm."""
import torch
import torch.nn as nn

from extract_direction import (
    ActivationSet,
    compute_directions,
    extract_activations,
    layer_signal_quality,
    purify_direction,
)


def test_direction_matches_known_mean_difference():
    torch.manual_seed(0)
    num_layers, num_prompts, hidden_dim = 3, 50, 8

    # Construct harmful/harmless activations with a known, deliberate mean offset
    # along a fixed axis per layer, plus noise, so the recovered direction should
    # align with that axis after normalization.
    true_axis = torch.zeros(num_layers, hidden_dim)
    true_axis[:, 0] = 1.0  # offset along dim 0 for every layer

    harmless_acts = torch.randn(num_layers, num_prompts, hidden_dim) * 0.01
    harmful_acts = harmless_acts + true_axis.unsqueeze(1) * 5.0

    harmful = ActivationSet(activations=harmful_acts)
    harmless = ActivationSet(activations=harmless_acts)

    directions = compute_directions(harmful, harmless)

    assert directions.shape == (num_layers, hidden_dim)
    for layer in range(num_layers):
        # unit-norm
        assert torch.isclose(directions[layer].norm(), torch.tensor(1.0), atol=1e-5)
        # points along the known offset axis (high cosine similarity)
        cos_sim = torch.dot(directions[layer], true_axis[layer])
        assert cos_sim > 0.99


def test_direction_handles_degenerate_layer_without_nan():
    # harmful and harmless means coincide exactly at layer 0 -> zero diff vector.
    # compute_directions must not divide by zero / produce NaNs.
    acts = torch.zeros(1, 4, 6)
    harmful = ActivationSet(activations=acts.clone())
    harmless = ActivationSet(activations=acts.clone())

    directions = compute_directions(harmful, harmless)

    assert not torch.isnan(directions).any()
    assert torch.allclose(directions, torch.zeros_like(directions))


def test_mismatched_layer_counts_raise():
    harmful = ActivationSet(activations=torch.randn(3, 5, 4))
    harmless = ActivationSet(activations=torch.randn(2, 5, 4))
    try:
        compute_directions(harmful, harmless)
        assert False, "expected ValueError for mismatched layer counts"
    except ValueError:
        pass


def test_purify_direction_orthogonal_input_is_noop():
    # refusal_dir already has zero component along harmless_mean -> purification
    # should leave it (up to sign/normalization) unchanged.
    harmless_mean = torch.tensor([[1.0, 0.0, 0.0]])
    refusal_dir = torch.tensor([[0.0, 1.0, 0.0]])

    purified = purify_direction(refusal_dir, harmless_mean)

    assert torch.allclose(purified, refusal_dir, atol=1e-5)


def test_purify_direction_parallel_input_is_zeroed():
    # refusal_dir points exactly along harmless_mean -> nothing survives
    # Gram-Schmidt against it, so the purified direction should be the zero vector
    # (compute_directions's existing degenerate-layer convention).
    harmless_mean = torch.tensor([[2.0, 0.0, 0.0]])
    refusal_dir = torch.tensor([[1.0, 0.0, 0.0]])

    purified = purify_direction(refusal_dir, harmless_mean)

    assert torch.allclose(purified, torch.zeros_like(purified), atol=1e-5)


def test_compute_directions_purified_returns_raw_and_purified():
    torch.manual_seed(1)
    num_layers, num_prompts, hidden_dim = 2, 40, 6

    # harmless mean offset along dim 0; refusal signal mixes in a dim-0 component
    # (which purification should strip) alongside a genuine dim-1 signal.
    harmless_acts = torch.randn(num_layers, num_prompts, hidden_dim) * 0.01
    harmless_acts[:, :, 0] += 3.0
    harmful_acts = harmless_acts.clone()
    harmful_acts[:, :, 0] += 2.0  # shares harmless's dim-0 axis
    harmful_acts[:, :, 1] += 4.0  # genuine refusal-specific signal

    harmful = ActivationSet(activations=harmful_acts)
    harmless = ActivationSet(activations=harmless_acts)

    raw, purified = compute_directions(harmful, harmless, purified=True)

    assert raw.shape == (num_layers, hidden_dim)
    assert purified.shape == (num_layers, hidden_dim)
    # unpurified default behavior is unaffected
    assert torch.allclose(raw, compute_directions(harmful, harmless))
    # purified direction has (near) zero dot product against the harmless direction
    harmless_hat = harmless.activations.mean(dim=1)
    harmless_hat = harmless_hat / harmless_hat.norm(dim=-1, keepdim=True)
    dots = (purified * harmless_hat).sum(dim=-1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-4)


def test_layer_signal_quality_ranks_high_snr_layer_above_low_snr_layer():
    torch.manual_seed(2)
    num_prompts, hidden_dim = 40, 6

    # Layer 0: large, orthogonal harmful/harmless means -> high SNR, low cosine,
    # high purity -> should score highest.
    good_harmless = torch.randn(num_prompts, hidden_dim) * 0.01
    good_harmless[:, 0] += 3.0
    good_harmful = good_harmless.clone()
    good_harmful[:, 1] += 5.0

    # Layer 1: harmful/harmless means nearly coincide (small, noisy diff) -> low SNR
    # -> should score lowest.
    bad_harmless = torch.randn(num_prompts, hidden_dim) * 0.01
    bad_harmless[:, 0] += 3.0
    bad_harmful = bad_harmless + torch.randn(num_prompts, hidden_dim) * 0.001

    harmful = ActivationSet(activations=torch.stack([good_harmful, bad_harmful]))
    harmless = ActivationSet(activations=torch.stack([good_harmless, bad_harmless]))

    scores = layer_signal_quality(harmful, harmless)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _RecordingTokenizer:
    """Records exactly what text and add_special_tokens value extract_activations
    actually tokenizes, so the chat-template regression below can assert on it."""

    chat_template = "present"

    def __init__(self):
        self.calls: list[tuple[list[str], bool]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True):
        self.calls.append((list(batch), add_special_tokens))
        seqs = [[(ord(c) % 50) + 1 for c in item][:5] or [1] for item in batch]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s)
        return _FakeBatchEncoding(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        )


class _IdentityLayer(nn.Module):
    def forward(self, x):
        return x


class _TinyModelForExtraction(nn.Module):
    """Doesn't need to be a meaningful language model - extract_activations only
    needs decoder layers to hook and a forward() that runs input through them."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_IdentityLayer(), _IdentityLayer()])

    def forward(self, input_ids, attention_mask=None):
        x = input_ids.unsqueeze(-1).float().expand(-1, -1, self.hidden_dim)
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_extract_activations_applies_chat_template_before_tokenizing():
    """Regression test for a real bug found from a live Llama-3.1-8B-Instruct run:
    extract_activations tokenized raw prompt text, so the captured 'last token'
    activation was at the end of the raw instruction rather than at the position
    right after the chat template opens the assistant turn - a different (weaker)
    signal for exactly the reason model_utils.format_chat_prompt's docstring
    explains. This asserts the tokenizer actually receives templated text, with
    add_special_tokens=False (since the template already embeds them)."""
    model = _TinyModelForExtraction(hidden_dim=4)
    tokenizer = _RecordingTokenizer()

    extract_activations(model, tokenizer, ["hello", "world"], device="cpu", batch_size=2)

    assert len(tokenizer.calls) == 1
    batch, add_special_tokens = tokenizer.calls[0]
    assert batch == ["<chat>hello</chat>", "<chat>world</chat>"]
    assert add_special_tokens is False
