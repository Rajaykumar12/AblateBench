# AblateBench

Finds the internal "refuse this" direction in an open-weight LLM, removes it two ways (reversible hook / permanent weight edit), and proves the before/after tradeoff with a self-built refusal-rate + capability eval harness — all running on rented **NVIDIA Blackwell (B200)** GPUs via [Modal](https://modal.com).

`Python 3.11` · `PyTorch` · `Transformers` · `Modal` · `NVIDIA Blackwell (B200)` · `pytest`

## Why this project exists

This reproduces, at small-to-medium scale, the **difference-in-means refusal
direction** technique from Arditi et al., ["Refusal in Language Models Is Mediated
by a Single Direction"](https://arxiv.org/abs/2406.11717) (arXiv:2406.11717) — the
technique the open-source community calls "abliteration." It's explicitly a
reproduction/extension exercise, not a claim of novel technique.

The interesting part isn't producing a de-refused checkpoint (plenty of tools do
that already) — it's the **eval harness around it**: a two-tier refusal grader, an
independent capability-regression guard, a layer/coefficient sweep, and a tradeoff
curve, all self-built and reported honestly, including the failures and dead ends
along the way. If a result looks too good (e.g. a layer that "suppresses" 98% of
refusals but only because it broke the model's coherence), that gets caught and
reported as a finding, not quietly dropped.

## Architecture

```
prepare_data()                              (Modal, CPU only)
  -> data/{harmful,harmless}_prompts_{train,heldout}.json
  -> data/capability_subset.json
       written to /cache/data on a shared Modal Volume
            |
            v
extract_and_save_direction(model, purified=False, multidomain=False, compile_model=False)
                                                       (Modal, Blackwell B200)
  extract_activations() on train prompts (single forward pass per prompt)
  -> compute_directions() -> per-layer unit directions [num_layers, hidden_dim]
  -> layer_signal_quality() printed (free — same activations, no extra GPU cost)
  saved to /cache/directions_<model>.pt
  if purified=True: also purify_direction() (Gram-Schmidt against the
    harmless-mean direction) -> /cache/directions_purified_<model>.pt
    (raw AND purified both saved, so both can be swept/compared)
  if multidomain=True: also extracts activations per-category (disjoint prompt
    set from both training and eval) -> compute_multidomain_directions()
    -> /cache/directions_multidomain_<model>.pt (raw/purified/universal/per_category)
  if compile_model=True: wraps the model in torch.compile for this stage only
            |
            v
run_sweep(model, directions_path)           (Modal, Blackwell B200)
  for each candidate layer:
    broadcast that ONE layer's direction to every layer of the model
    for each coefficient in [1.0, 0.85, 0.7, 0.5]:
      install runtime ablation hooks -> generate -> grade -> measure capability
  -> sweep_result_<model>.json (pick the winning (layer, coefficient) by hand,
     or SweepResult.pareto_frontier() / scripts/plot_pareto.py for the full curve)
            |
            v
run_full_pipeline(model, best_layer, best_coefficient)  (Modal, Blackwell B200)
  baseline (no ablation): generate + grade + capability, across 12 eval categories
  for each coefficient: runtime-ablate -> generate -> capability -> grade
  permanent weight-level ablation at best_coefficient -> generate -> capability -> grade
  snapshot -> restore -> repeat weight ablation with norm_preserving=True -> grade
    (2nd row, method="weight_norm_preserved", same coefficient, no reload needed)
  -> results_<model>.json (refusal-by-category, capability, tradeoff curve)
            |
            v
flagship_run(30B+ model)                    (Modal, Blackwell B200)
  independent of the above, no eval sweep — confirms the technique still works
  and reports peak GPU memory at flagship scale
```

**Key design decisions:**

- **Per-layer directions, broadcast model-wide.** `compute_directions` computes a
  distinct direction at every layer, but the actual technique picks *one* winning
  layer's direction and removes that same vector from the residual stream at every
  layer — not just its source layer (residual streams are additive, so a
  single-layer edit can be reintroduced downstream).
- **Two ablation methods, proven equivalent.** A reversible forward-hook
  (`ablate_runtime.py`) and a permanent weight-level orthogonalization
  (`ablate_weights.py`) are both implemented; `tests/test_ablate_weights.py` proves
  they produce numerically identical output for a single layer, which is the basis
  for trusting that the permanent edit reproduces what the reversible hook
  demonstrated.
- **Direction purification + signal-quality scoring.** `purify_direction`
  Gram-Schmidt-orthogonalizes a computed direction against the harmless-prompt
  direction, stripping out generic "a prompt was given at all" signal from the
  true refusal-specific signal — motivated by an early finding that the raw
  direction suppressed one prompt-phrasing style far more than another.
  `layer_signal_quality` scores every layer's SNR/purity from the same activations
  already extracted, as a free, cheap pre-filter before spending sweep GPU-time.
- **Two-tier refusal grading.** A cheap rule-based pass classifies obvious
  refusals/compliances; only genuinely ambiguous responses escalate to an
  LLM-judge fallback (the model judging itself), keeping grading fast without
  sacrificing accuracy on hard cases.
- **Sweep grid is a fixed shortlist, not exhaustive.** Layer and coefficient
  candidates are capped by construction (a handful of evenly-spaced layers × 4
  coefficients) so sweep cost stays bounded regardless of model depth.
- **Norm-preserving weight ablation.** `orthogonalize_matrix_norm_preserving`
  restores the whole edited weight matrix's Frobenius norm with a single global
  scalar after orthogonalizing — a per-row rescale would silently reintroduce a
  nonzero refusal-direction component in this write-side weight convention (see
  `docs/01-architecture.md` for the proof). A snapshot/restore helper lets the
  pipeline compare plain vs. norm-preserving ablation against the same loaded
  checkpoint without a second model reload.
- **Multi-domain direction extraction.** Extraction can pool activations across
  several disjoint-from-eval prompt categories and separate a `universal` direction
  (common across categories) from each category's own idiosyncratic signal, rather
  than fitting the direction to one undifferentiated harmful-prompt pool.
- **Pareto frontier over filtered-max picks.** `SweepResult.best()` and
  `TradeoffCurve.best_meeting_capability_floor()` each return one operating point;
  `pareto_frontier()` (duck-typed, dependency-free) surfaces the full non-dominated
  set instead, plottable locally via `scripts/plot_pareto.py`.
- **torch.compile scoped to extraction only.** Opt-in on the stable,
  hooks-installed-once extraction forward pass; deliberately not enabled on the
  sweep/generation path, where hooks are reinstalled per candidate and would fight
  `reduce-overhead`'s CUDA graphs.

## Requirements

- **Python 3.11**
- A [Modal](https://modal.com) account — all GPU compute runs on Modal-hosted
  **NVIDIA Blackwell (B200)** GPUs, not locally; budgeted at roughly $30/month for
  this project's scale
- A [Hugging Face](https://huggingface.co) account + access token — some models and
  datasets used here are gated
- Package versions pinned in `requirements.txt`: `modal>=1.5.0`, `torch>=2.4.0`,
  `transformers>=4.44.0`, `accelerate>=0.33.0`, `datasets>=2.20.0`,
  `numpy>=1.26.0`, `pytest>=8.0.0`
- `matplotlib>=3.8.0` — optional, local-only, only needed for `scripts/plot_pareto.py`
  (not installed on the Modal image)

## Quickstart

```bash
git clone <this-repo-url>
cd AblateBench

pip install -r requirements.txt

# One-time Modal auth (opens a browser)
modal setup

# Store your HF token as a Modal secret — see Environment Variables below
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

## Usage

```bash
# 1. Pull real AdvBench/Alpaca/ARC-Easy data onto the shared volume (CPU only)
modal run modal_app.py --stage prepare_data

# 2. Extract the refusal direction for a model (first GPU spend)
modal run modal_app.py --stage extract --model-name Qwen/Qwen2.5-7B-Instruct
# ...or extract both raw and purified directions in one pass:
modal run modal_app.py::extract_and_save_direction \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-name Qwen_Qwen2.5-7B-Instruct \
    --purified

# 3. Sweep layers/coefficients to find the best ablation operating point
modal run modal_app.py::run_sweep --model-name Qwen/Qwen2.5-7B-Instruct \
    --directions-path /cache/directions_Qwen_Qwen2.5-7B-Instruct.pt

# 4. Run the full pipeline at the sweep's winning (layer, coefficient)
modal run modal_app.py::run_full_pipeline \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --best-layer 16 --best-coefficient 1.0

# 5. Confirm the technique holds at 30B+ scale (no eval sweep, just a smoke check)
modal run modal_app.py --stage flagship --model-name Qwen/Qwen2.5-32B-Instruct

# Optional: multi-domain extraction (separate universal vs. per-category signal)
python3.11 data/build_extraction_categories.py data/extraction_categories
modal run modal_app.py::extract_and_save_direction \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-name Qwen_Qwen2.5-7B-Instruct \
    --multidomain

# Optional: plot the Pareto frontier of a sweep or full-pipeline result (local only)
modal volume get abliteration-eval-cache sweep_result_Qwen_Qwen2.5-7B-Instruct.json .
python3.11 scripts/plot_pareto.py sweep_result_Qwen_Qwen2.5-7B-Instruct.json --out pareto.png
```

See [docs/07-verification-commands.md](docs/07-verification-commands.md) for the
full set of commands (including `--compile-model` and norm-preserving weight
ablation, which runs automatically inside `run_full_pipeline`).

Check cost, logs, or cached data without leaving the terminal:

```bash
modal app list                                     # recent runs
modal app logs <app-id>                            # replay a run's full output
modal volume ls abliteration-eval-cache            # inspect cached data/weights/results
```

## Results (Qwen2.5-7B-Instruct)

All numbers below are post chat-template-fix and reflect the current best-known
operating point, not the first thing tried.

**Headline finding**: at **layer 16, coefficient 1.0**, permanent weight-level
ablation drops refusal on a 292-prompt, 12-category eval set from a ~95-97%
baseline down to:

| direction variant | refusal rate | drop from baseline | capability |
|---|---|---|---|
| raw (plain difference-in-means) | 43.2% | ~52pp | 1.000 (unchanged) |
| **purified** (Gram-Schmidt against the harmless direction) | **28.8%** | **~67pp** | 1.000 (unchanged) |

Capability — measured on an independent ARC-Easy-style MCQ subset — never moves off
1.000 in either case. The model isn't getting dumber; it's specifically losing the
tendency to refuse.

**Two things had to be discovered to get here, not just assumed:**

1. **The direction generalizes across prompt phrasing much better than it first
   appeared to.** An earlier run at layer 21 (picked by a sweep that only checked
   AdvBench-style prompts) suppressed refusal by just 3.4-8.9pp on this same
   templated eval set — looking like the technique barely worked outside its
   training distribution. Layer 16 alone gets 52pp on identical prompts. Layer
   *choice*, not just the direction itself, turned out to be the dominant factor.
2. **Purifying the direction against the harmless-prompt direction (Gram-Schmidt)
   adds a further ~15pp of suppression on top of that**, for free — same
   activations already extracted, no extra GPU forward passes. It also turns an
   illusory "0% refusal" result at layer 1 (which was actually just capability
   collapsing into garbled text) into an honest no-op that leaves the model intact.
3. A cheap per-layer signal-quality score (`SNR × (1 − cosine) × purity`, computed
   from the same activations) correctly flags layers 0-13 as weak candidates before
   spending any sweep GPU-time — though it's a coarse filter, not a fine-grained
   ranker: it doesn't reliably pick the single best layer among the good ones.

**Honest caveat**: these numbers rely heavily on the model's own self-judge tier
(~96-97% of verdicts, vs. ~20-27% in earlier runs) rather than the cheap rule-based
grader, and the weight-ablated method's judge runs on the same model that's already
been permanently ablated — a known, not-yet-fixed asymmetry (the judge is a closure
over the model object, and `ablate_weights_` mutates that object in place, so the
"self-judge" for the permanently-ablated variant is judging its own already-altered
output). Directionally consistent with the runtime-hook variant's numbers, but
treat the exact percentage-point figures as provisional until that's resolved.

**Reproducibility + norm-preserving ablation, checked against this exact headline
point**: re-running layer 16/coefficient 1.0/purified on the post-upgrade codebase
reproduced the 28.8% refusal rate essentially exactly (28.77%), confirming the
four-part upgrade (norm-preserving weight ablation, multi-domain extraction,
torch.compile, Pareto-frontier tooling) didn't change existing, non-opt-in
behavior. The new **norm-preserving** weight ablation variant — which rescales the
orthogonal projection to restore the original weight matrix's norm instead of
letting it shrink — scored 29.1% at the same operating point, a ~0.3pp difference
from plain weight ablation. Same small gap held at layer 21 too. So
norm-preservation's stability benefit doesn't come at a meaningful
refusal-suppression cost — though this harness doesn't yet measure matrix norms or
generation coherence directly, so that benefit itself is asserted from the
implementation and its unit tests, not measured live. See
[docs/05-results-log.md](docs/05-results-log.md) Runs 13-14 for full detail,
including the multi-domain and torch.compile live-verification results.

## Future initiatives

- **Fix the weight-ablated self-judge bias** noted above — likely by judging with
  a model snapshot captured before `ablate_weights_` mutates it in place, rather
  than the same (already-ablated) model object.
- **Verify the four-part upgrade live on Modal** — norm-preserving weight ablation,
  multi-domain extraction, and `torch.compile` are implemented and unit-tested
  (54/54 passing locally) but not yet exercised on a real GPU run; see
  [docs/07-verification-commands.md](docs/07-verification-commands.md) for the exact
  commands, none of which have been run yet as of this README.
- **A generalized qualitative compare tool** (gap-analysis #4) — widen the current
  one-prompt debug helper into a reusable side-by-side (baseline vs. ablated, across
  prompts, coefficients, and ablation methods) rather than a bespoke script per
  question.
- **Second-model repeat** — re-run the full pipeline (extract → sweep → full
  pipeline) on Llama-3.1-8B-Instruct now that the chat-template fix is in place;
  the existing Llama numbers predate that fix and are invalid.
- **30B+ flagship confirmation run** — `flagship_run` exists but hasn't been
  executed yet; confirms the technique and memory footprint hold at flagship scale.

## Development & Testing

```bash
# Run the full unit test suite (pure PyTorch, CPU-only, no GPU/network required)
python3.11 -m pytest tests/ -v

# Syntax/import check without deploying
python3.11 -m py_compile modal_app.py
python3.11 -c "import modal_app"
```

54 unit tests, all pure `torch` on CPU — no GPU, no network, no HF access needed.
No linter or build step is configured — this is a script/pipeline project, not a
packaged library. See [docs/06-testing.md](docs/06-testing.md) for what's covered,
file by file.

## Environment Variables

This project uses **Modal Secrets**, not a `.env` file — nothing reads `os.environ` directly except `HF_HOME`, which the code sets itself for cache placement.

| Variable | Description | Default / Required |
|---|---|---|
| `HF_TOKEN` | Hugging Face access token, injected via the `huggingface-secret` Modal secret (`modal secret create huggingface-secret HF_TOKEN=...`). Required to download gated datasets (`walledai/AdvBench`) and gated models (e.g. Llama). | **Required** — no default; every `@app.function` in `modal_app.py` declares `secrets=[HF_SECRET]` |
| `HF_HOME` | Where Hugging Face caches downloads inside the Modal container. | Set automatically to `/cache/huggingface` (on the persistent `abliteration-eval-cache` volume) — not user-configurable |

## Project Structure

```
AblateBench/
├── modal_app.py              # Entry point — all Modal functions (prepare_data, extract,
│                              # run_sweep, run_full_pipeline, flagship_run) + local_entrypoint
├── src/
│   ├── model_utils.py        # Shared decoder-layer introspection + weight snapshot/restore
│   ├── extract_direction.py  # Activation extraction + per-layer direction computation,
│   │                          # direction purification, signal-quality scoring, multi-domain directions
│   ├── layer_sweep.py        # Layer/coefficient sweep
│   ├── ablate_runtime.py     # Reversible forward-hook ablation
│   ├── ablate_weights.py     # Permanent weight-level ablation (plain + norm-preserving)
│   ├── rescale.py            # Capability rescaling + tradeoff-curve tracking
│   ├── eval_refusal.py       # Two-tier refusal grader (rule-based + judge fallback)
│   ├── eval_capability.py    # MCQ capability harness (ARC-Easy style)
│   └── pareto.py             # Pareto-frontier computation, shared by SweepResult/TradeoffCurve
├── scripts/
│   └── plot_pareto.py        # Local-only: plots a saved sweep/results JSON's Pareto frontier
├── data/
│   ├── prepare_prompts.py              # Downloads/splits AdvBench + Alpaca (needs network)
│   ├── build_eval_categories.py        # Generates ~292 category-labeled refusal-eval prompts
│   ├── build_extraction_categories.py  # Same categories, distinct phrasing, for multi-domain extraction
│   ├── build_capability_subset.py      # Real ARC-Easy sample (falls back to a static set offline)
│   ├── eval_categories/                # Generated output, 12 categories
│   ├── extraction_categories/          # Generated output, 12 categories (disjoint phrasing from eval)
│   └── capability_subset.json          # Generated output, 50-100 MCQ capability subset
├── tests/                    # 54 unit tests, pure PyTorch CPU, no GPU/network needed
└── requirements.txt
```

## License

[MIT](LICENSE)
