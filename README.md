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
extract_and_save_direction(model, purified=False)     (Modal, Blackwell B200)
  extract_activations() on train prompts (single forward pass per prompt)
  -> compute_directions() -> per-layer unit directions [num_layers, hidden_dim]
  -> layer_signal_quality() printed (free — same activations, no extra GPU cost)
  saved to /cache/directions_<model>.pt
  if purified=True: also purify_direction() (Gram-Schmidt against the
    harmless-mean direction) -> /cache/directions_purified_<model>.pt
    (raw AND purified both saved, so both can be swept/compared)
            |
            v
run_sweep(model, directions_path)           (Modal, Blackwell B200)
  for each candidate layer:
    broadcast that ONE layer's direction to every layer of the model
    for each coefficient in [1.0, 0.85, 0.7, 0.5]:
      install runtime ablation hooks -> generate -> grade -> measure capability
  -> sweep_result_<model>.json (pick the winning (layer, coefficient) by hand)
            |
            v
run_full_pipeline(model, best_layer, best_coefficient)  (Modal, Blackwell B200)
  baseline (no ablation): generate + grade + capability, across 12 eval categories
  for each coefficient: runtime-ablate -> generate -> capability -> grade
  permanent weight-level ablation at best_coefficient -> generate -> capability -> grade
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
```

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

## Future initiatives

- **Fix the weight-ablated self-judge bias** noted above — likely by judging with
  a model snapshot captured before `ablate_weights_` mutates it in place, rather
  than the same (already-ablated) model object.
- **Norm-preserving weight ablation** — decompose each ablated weight row into
  magnitude and direction, rescale back to original magnitude after removing the
  refusal component, as a third capability-protection lever alongside the existing
  coefficient blend.
- **A generalized qualitative compare tool** — widen the current one-prompt debug
  helper into a reusable side-by-side (baseline vs. ablated, across prompts,
  coefficients, and ablation methods) rather than a bespoke script per question.
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

39 unit tests, all pure `torch` on CPU — no GPU, no network, no HF access needed.
No linter or build step is configured — this is a script/pipeline project, not a
packaged library.

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
│   ├── model_utils.py        # Shared decoder-layer introspection, used by every stage below
│   ├── extract_direction.py  # Activation extraction + per-layer direction computation,
│   │                          # direction purification, signal-quality scoring
│   ├── layer_sweep.py        # Layer/coefficient sweep
│   ├── ablate_runtime.py     # Reversible forward-hook ablation
│   ├── ablate_weights.py     # Permanent weight-level ablation
│   ├── rescale.py            # Capability rescaling + tradeoff-curve tracking
│   ├── eval_refusal.py       # Two-tier refusal grader (rule-based + judge fallback)
│   └── eval_capability.py    # MCQ capability harness (ARC-Easy style)
├── data/
│   ├── prepare_prompts.py           # Downloads/splits AdvBench + Alpaca (needs network)
│   ├── build_eval_categories.py     # Generates ~292 category-labeled refusal-eval prompts
│   ├── build_capability_subset.py   # Real ARC-Easy sample (falls back to a static set offline)
│   ├── eval_categories/             # Generated output, 12 categories
│   └── capability_subset.json       # Generated output, 50-100 MCQ capability subset
├── tests/                    # 39 unit tests, pure PyTorch CPU, no GPU/network needed
└── requirements.txt
```

## License

[MIT](LICENSE)
