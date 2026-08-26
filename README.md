# abliteration-eval

Finds the internal "refuse this" direction in an open-weight LLM, removes it two ways (reversible hook / permanent weight edit), and proves the before/after tradeoff with a self-built refusal-rate + capability eval harness — all running on rented B200 GPUs via Modal.

`Python 3.11` · `PyTorch` · `Transformers` · `Modal (B200)` · `pytest`

## Features

- **Difference-in-means direction extraction** — per-layer refusal direction computed from harmful (AdvBench) vs. harmless (Alpaca) activations, no gradients required
- **Layer/coefficient sweep** — automatically shortlists candidate layers and ablation strengths, scored by refusal-rate drop vs. capability cost
- **Two ablation methods** — a reversible forward-hook (`ablate_runtime.py`) and a permanent weight-level orthogonalization (`ablate_weights.py`), proven numerically equivalent in `tests/`
- **Two-tier refusal grader** — cheap regex pass first, LLM-judge fallback (the model judging itself) only for ambiguous responses
- **Capability regression guard** — every ablation variant is scored on an ARC-Easy multiple-choice subset so "suppressed refusal" can't secretly mean "broke the model"
- **Modal-native orchestration** — one `modal_app.py` drives data prep, extraction, sweep, full pipeline, and a 30B+ flagship confirmation run, all cached in a shared `modal.Volume`

## Quickstart

### Prerequisites

- Python 3.11
- A [Modal](https://modal.com) account (compute runs on Modal's B200 GPUs, not locally)
- A [Hugging Face](https://huggingface.co) account + access token (some models/datasets used here are gated)

### Setup

```bash
git clone <this-repo-url>
cd abliteration-eval

pip install -r requirements.txt

# One-time Modal auth (opens a browser)
modal setup

# Store your HF token as a Modal secret — see Environment Variables below
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Run

```bash
# 1. Pull real AdvBench/Alpaca/ARC-Easy data onto the shared volume (CPU only)
modal run modal_app.py --stage prepare_data

# 2. Extract the refusal direction for a model (first GPU spend)
modal run modal_app.py --stage extract --model-name Qwen/Qwen2.5-7B-Instruct

# 3. Sweep layers/coefficients to find the best ablation operating point
modal run modal_app.py::run_sweep --model-name Qwen/Qwen2.5-7B-Instruct \
    --directions-path /cache/directions_Qwen_Qwen2.5-7B-Instruct.pt
```

## Results (Qwen2.5-7B-Instruct)

All numbers below are post chat-template-fix (see `docs/03-bugs-and-fixes.md`, Bug
#3) and reflect the current best-known operating point, not the first thing tried.

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

## Environment Variables

This project uses **Modal Secrets**, not a `.env` file — nothing reads `os.environ` directly except `HF_HOME`, which the code sets itself for cache placement.

| Variable | Description | Default / Required |
|---|---|---|
| `HF_TOKEN` | Hugging Face access token, injected via the `huggingface-secret` Modal secret (`modal secret create huggingface-secret HF_TOKEN=...`). Required to download gated datasets (`walledai/AdvBench`) and gated models (e.g. Llama). | **Required** — no default; every `@app.function` in `modal_app.py` declares `secrets=[HF_SECRET]` |
| `HF_HOME` | Where Hugging Face caches downloads inside the Modal container. | Set automatically to `/cache/huggingface` (on the persistent `abliteration-eval-cache` volume) — not user-configurable |

## Project Structure

```
abliteration-eval/
├── modal_app.py              # Entry point — all Modal functions (prepare_data, extract,
│                              # run_sweep, run_full_pipeline, flagship_run) + local_entrypoint
├── src/
│   ├── model_utils.py        # Shared decoder-layer introspection, used by every stage below
│   ├── extract_direction.py  # Activation extraction + per-layer direction computation
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

## Usage

Run the full pipeline for one model once you have a winning (layer, coefficient) from the sweep:

```bash
modal run modal_app.py::run_full_pipeline \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --best-layer 21 --best-coefficient 1.0
```

Confirm the technique holds at 30B+ scale (extraction + both ablation methods only, no eval sweep):

```bash
modal run modal_app.py --stage flagship --model-name Qwen/Qwen2.5-32B-Instruct
```

Check cost, logs, or cached data without leaving the terminal:

```bash
modal app list                                     # recent runs
modal app logs <app-id>                            # replay a run's full output
modal volume ls abliteration-eval-cache            # inspect cached data/weights/results
```

## Development & Testing

```bash
# Run the full unit test suite (pure PyTorch, CPU-only, no GPU/network required)
python3.11 -m pytest tests/ -v

# Syntax/import check without deploying
python3.11 -m py_compile modal_app.py
python3.11 -c "import modal_app"
```

No linter or build step is configured — this is a script/pipeline project, not a packaged library.
