"""PRD §9.1: direction computation correctness — given synthetic activations with a
known mean difference, verify d_layer matches the expected difference-of-means and is
unit-norm."""
import torch
import torch.nn as nn

from extract_direction import (
    ActivationSet,
    compute_directions,
    compute_multidomain_directions,
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


def test_multidomain_pooling_concatenates_correctly():
    """Pooled activations must be the concatenation of every category's activations
    along the prompt dim, so a category with more prompts gets proportionally more
    weight in the pooled mean (same reduction a single undifferentiated pool would
    give a larger category)."""
    torch.manual_seed(10)
    num_layers, hidden_dim = 2, 5

    cat_a = ActivationSet(activations=torch.randn(num_layers, 30, hidden_dim))
    cat_b = ActivationSet(activations=torch.randn(num_layers, 10, hidden_dim))
    harmless = ActivationSet(activations=torch.randn(num_layers, 20, hidden_dim) * 0.01)

    result = compute_multidomain_directions({"cat_a": cat_a, "cat_b": cat_b}, harmless)

    assert result.raw.shape == (num_layers, hidden_dim)
    assert result.purified.shape == (num_layers, hidden_dim)
    assert result.universal.shape == (num_layers, hidden_dim)
    assert set(result.per_category.keys()) == {"cat_a", "cat_b"}

    # raw must match compute_directions on the manually pooled (concatenated) set.
    pooled = ActivationSet(
        activations=torch.cat([cat_a.activations, cat_b.activations], dim=1)
    )
    expected_raw = compute_directions(pooled, harmless)
    assert torch.allclose(result.raw, expected_raw, atol=1e-5)


def test_multidomain_universal_isolates_shared_axis_while_per_category_retains_idiosyncrasies():
    """Synthetic 4-axis setup: axis 0 is the harmless axis, axis 1 is a "common
    refusal" signal shared by every category, axes 2/3 are idiosyncratic to
    category A/B respectively (plus a third category C on axis 4's-worth of its own
    idiosyncrasy, so there's more than one independent category-difference direction
    to Gram-Schmidt against — with only 2 categories, their residuals are always
    exactly anti-parallel, so a 2nd projection is a no-op and only the *difference*
    between the two categories' idiosyncrasies gets removed, not each one fully).

    Checks the two guarantees compute_multidomain_directions actually makes:
    (1) universal is (near) orthogonal to every category's own residual direction —
    the literal Gram-Schmidt invariant, same style of check as
    test_purify_direction_*'s orthogonality assertions; and
    (2) per_category directions retain a strong component along their own
    idiosyncratic axis, much more so than universal does.
    """
    torch.manual_seed(11)
    num_layers, hidden_dim = 1, 6
    noise = 0.01

    axes = list(torch.eye(hidden_dim))
    axis0, axis1, axis2, axis3, axis4 = axes[0], axes[1], axes[2], axes[3], axes[4]

    def _acts(mean_offset: torch.Tensor, num_prompts: int) -> ActivationSet:
        base = torch.randn(num_layers, num_prompts, hidden_dim) * noise
        return ActivationSet(activations=base + mean_offset.view(1, 1, hidden_dim))

    harmless_offset = 3.0 * axis0
    harmless = _acts(harmless_offset, num_prompts=200)

    # all three categories share axis1 (the "common refusal" signal); each also
    # carries its own idiosyncratic axis (2 for A, 3 for B, 4 for C).
    cat_a_offset = harmless_offset + 2.0 * axis1 + 1.0 * axis2
    cat_b_offset = harmless_offset + 2.0 * axis1 + 1.0 * axis3
    cat_c_offset = harmless_offset + 2.0 * axis1 + 1.0 * axis4
    cat_a = _acts(cat_a_offset, num_prompts=200)
    cat_b = _acts(cat_b_offset, num_prompts=150)
    cat_c = _acts(cat_c_offset, num_prompts=100)

    harmful_by_category = {"cat_a": cat_a, "cat_b": cat_b, "cat_c": cat_c}
    result = compute_multidomain_directions(harmful_by_category, harmless)

    # (1) recompute each category's residual against the pooled mean exactly as
    # compute_multidomain_directions does. The *last* category processed (sorted by
    # name -> "cat_c") is exactly orthogonal to universal, since nothing runs after
    # it to reintroduce a component; earlier categories (single-pass, not iterated to
    # convergence, unlike modified Gram-Schmidt) end up only near-orthogonal, since a
    # later projection can partially reintroduce a small component along an earlier,
    # non-orthogonal residual direction.
    pooled = torch.cat([acts.activations for acts in harmful_by_category.values()], dim=1)
    pooled_mean = pooled.mean(dim=1)  # [num_layers, hidden_dim]
    for category_name, acts in harmful_by_category.items():
        category_mean = acts.activations.mean(dim=1)
        residual_hat = category_mean - pooled_mean
        residual_hat = residual_hat / residual_hat.norm(dim=-1, keepdim=True)
        dot = (result.universal * residual_hat).sum(dim=-1)
        tol = 1e-4 if category_name == "cat_c" else 1e-2
        assert torch.allclose(dot, torch.zeros_like(dot), atol=tol)

    def cos_to_axis(vec: torch.Tensor, axis: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(
            vec[0].unsqueeze(0), axis.unsqueeze(0)
        ).item()

    # (2) per_category directions retain their own category's idiosyncratic axis far
    # more strongly than universal does.
    assert abs(cos_to_axis(result.per_category["cat_a"], axis2)) > abs(
        cos_to_axis(result.universal, axis2)
    )
    assert abs(cos_to_axis(result.per_category["cat_b"], axis3)) > abs(
        cos_to_axis(result.universal, axis3)
    )
    assert abs(cos_to_axis(result.per_category["cat_c"], axis4)) > abs(
        cos_to_axis(result.universal, axis4)
    )


def test_multidomain_single_category_matches_compute_directions_purified():
    """Degenerate case: with one category, its residual against the pooled mean is
    the zero vector, so the universal Gram-Schmidt pass is a no-op and universal
    should exactly equal purified, and raw/purified should exactly match
    compute_directions(purified=True) on that same pool."""
    torch.manual_seed(12)
    num_layers, num_prompts, hidden_dim = 2, 40, 6

    harmless_acts = torch.randn(num_layers, num_prompts, hidden_dim) * 0.01
    harmless_acts[:, :, 0] += 3.0
    harmful_acts = harmless_acts.clone()
    harmful_acts[:, :, 0] += 2.0
    harmful_acts[:, :, 1] += 4.0

    harmful = ActivationSet(activations=harmful_acts)
    harmless = ActivationSet(activations=harmless_acts)

    expected_raw, expected_purified = compute_directions(harmful, harmless, purified=True)

    result = compute_multidomain_directions({"only_category": harmful}, harmless)

    assert torch.allclose(result.raw, expected_raw, atol=1e-5)
    assert torch.allclose(result.purified, expected_purified, atol=1e-5)
    assert torch.allclose(result.universal, expected_purified, atol=1e-5)


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
