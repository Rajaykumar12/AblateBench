"""PRD §9.1: direction computation correctness — given synthetic activations with a
known mean difference, verify d_layer matches the expected difference-of-means and is
unit-norm."""
import torch
import torch.nn as nn

from extract_direction import ActivationSet, compute_directions, extract_activations


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


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _RecordingTokenizer:
    """Records exactly what text and add_special_tokens value extract_activations
    actually tokenizes, so the chat-template regression below can assert on it."""

    chat_template = "present"

    def __init__(self):
        self.calls: list[tuple[list[str], bool]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
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
