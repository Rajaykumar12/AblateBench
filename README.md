# AblateBench

`Python 3.11` · `PyTorch` · `Transformers` · `Modal` · `NVIDIA H100 / B200` · `pytest`

A complete pipeline to find, remove, and evaluate the internal "refusal direction" in open-weight LLMs — a technique known as **abliteration**. It reproduces the difference-in-means technique from [Arditi et al. (2024)](https://arxiv.org/abs/2406.11717) and wraps it in a rigorous, self-built evaluation harness, running entirely on rented GPUs via [Modal](https://modal.com) — B200 by default, any Modal GPU per run with `--gpu`.

**Released model:** [rajaykumar12959/qwen3.8-27b-abliterated](https://huggingface.co/rajaykumar12959/qwen3.8-27b-abliterated) — refusal 99.3% → 0.0%, MMLU-Pro 65.1% → 61.8%.

**Key features:**

- **Two-tier refusal grader** — rule-based classification with an LLM-judge fallback for ambiguous cases.
- **Independent capability guard** — ARC-Easy as a quick sanity check, plus MMLU-Pro and harmless-prompt KL divergence for the cost ARC-Easy is too easy to show.
- **Two ablation methods** — a reversible runtime hook and a permanent weight edit, proven numerically equivalent.
- **Layer/coefficient sweep** — picks the operating point on the refusal-vs-capability tradeoff curve, not a guess.
- **Architecture coverage** — Llama, Qwen2.5, Qwen3.5–3.8 hybrid linear-attention models, and Gemma 2/3 sandwich-norm models, each with tests that fail on the naive edit.

## Requirements

- **Python 3.11**
- A [Modal](https://modal.com) account — all GPU compute runs on Modal-hosted GPUs, not locally (the full 27B run below cost ~$8 on an H100)
- A [Hugging Face](https://huggingface.co) account + access token — some models/datasets are gated
- Pinned versions in `requirements.txt`: `modal>=1.5.0`, `torch>=2.4.0`, `transformers>=5.19.0`, `accelerate>=0.33.0`, `datasets>=2.20.0`, `numpy>=1.26.0`, `pytest>=8.0.0`

## Quickstart

```bash
git clone <this-repo-url>
cd AblateBench

pip install -r requirements.txt

# One-time Modal auth (opens a browser)
modal setup

# Store your HF token as a Modal secret
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

## Usage

```bash
# 1. Pull AdvBench/Alpaca/ARC-Easy data onto the shared volume (CPU only)
modal run modal_app.py --stage prepare_data

# 2. Extract raw + purified refusal directions for a model
modal run modal_app.py::extract_and_save_direction \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --output-name Qwen_Qwen2.5-7B-Instruct \
    --purified

# 3. Sweep layers/coefficients to find the best operating point
modal run modal_app.py::run_sweep --model-name Qwen/Qwen2.5-7B-Instruct \
    --directions-path /cache/directions_Qwen_Qwen2.5-7B-Instruct.pt

# 4. Run the full pipeline at the sweep's winning (layer, coefficient)
modal run modal_app.py::run_full_pipeline \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --best-layer 16 --best-coefficient 1.0

# 4b. Add --save-checkpoint to persist the weight-ablated model, then publish
#     it separately (never automatic):
modal run modal_app.py::run_full_pipeline \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --best-layer 16 --best-coefficient 1.0 --save-checkpoint
modal run modal_app.py::push_checkpoint_to_hub \
    --checkpoint-path /cache/checkpoints/Qwen_Qwen2.5-7B-Instruct/weight_coeff1.0 \
    --repo-id your-username/qwen2.5-7b-abliterated

# 5. Confirm the technique holds at 30B+ scale (smoke check, no eval sweep)
modal run modal_app.py --stage flagship --model-name Qwen/Qwen2.5-32B-Instruct
```

The `main` entrypoint runs every stage with one flag set, including `--gpu` to size the GPU to the model (bf16 weights take ~2 GB per 1B parameters). The Qwen3.8-27B release was produced with:

```bash
M=Qwen/Qwen3.8-27B
modal run --detach modal_app.py --stage extract    --model-name $M --purified --gpu H100
modal run --detach modal_app.py --stage sweep      --model-name $M --purified --gpu H100 \
    --layer-shortlist 33,37,41,47,53,56
modal run --detach modal_app.py --stage full       --model-name $M --purified --gpu H100 \
    --best-layer 37 --best-coefficient 1.0 --save-checkpoint
modal run --detach modal_app.py --stage benchmarks --model-name $M --purified --gpu H100 \
    --best-layer 37 --best-coefficient 1.0          # MMLU-Pro + harmless-prompt KL
modal run --detach modal_app.py --stage push \
    --checkpoint-path /cache/checkpoints/Qwen_Qwen3.8-27B/weight_coeff1.0 \
    --repo-id your-username/qwen3.8-27b-abliterated
```

Use `--detach` for anything longer than a few minutes: without it, the run is tied to your terminal, and a laptop going to sleep stops the container (Modal restarts the job from scratch when the client reconnects).

Check cost, logs, or cached data without leaving the terminal:

```bash
modal app list                              # recent runs
modal app logs <app-id>                     # replay a run's full output
modal volume ls abliteration-eval-cache     # inspect cached data/weights/results
```

## Results (Qwen2.5-7B-Instruct)

Evaluated on a 292-prompt, 12-category eval set. Ablation performed at **layer 16, coefficient 1.0**.

| Direction variant | Refusal rate | Drop from baseline | Capability (ARC-Easy) |
|---|---|---|---|
| **Baseline** (no ablation) | ~96.0% | — | 1.000 |
| **Raw** (plain diff-in-means) | 43.2% | ~52pp | 1.000 (unchanged) |
| **Purified** (Gram-Schmidt) | **28.8%** | **~67pp** | 1.000 (unchanged) |

**Key insights:**

- **Purification matters.** Gram-Schmidt orthogonalization against the harmless-prompt direction adds ~15pp of suppression for free, filtering generic "a prompt was given" signal out of the refusal-specific signal.
- **Layer selection is dominant.** The direction only generalizes across prompt phrasings when the right layer is picked — layer 21 suppressed refusal by just ~5pp on this eval set, while layer 16 hit 52pp on identical prompts.
- **Signal-quality scoring pre-filters cheaply.** A per-layer SNR/purity score, computed from activations already extracted, flags weak layers before any sweep GPU-time is spent.

**Caveat:** the weight-ablated variant's self-judge currently evaluates itself *after* mutation — a known bias, tracked below, that hasn't been corrected yet. Numbers are directionally consistent with the runtime-hook variant but should be treated as provisional.

## Results (Qwen3.8-27B)

Hybrid linear/full-attention model (48 Gated DeltaNet + 16 full-attention layers), evaluated in non-thinking mode. Purified direction from **layer 37 of 64**, coefficient 1.0, on an H100. Total Modal cost ~$8.

| Metric | Original | Abliterated |
|---|---|---|
| Refusal rate (292 prompts, 12 categories) | 99.3% | **0.0%** (every category) |
| MMLU-Pro (490 questions, 14 subjects; chance 11.7%) | 65.1% | **61.8%** (−3.3pp) |
| KL divergence on harmless prompts (first token) | — | **0.086 nats** (92% same top-1 token) |
| ARC-Easy (100 questions) | 1.000 | 1.000 |

| Layer (depth) | Validation refusal at c=1.0 |
|---|---|
| 33 (52%) | 14% |
| **37 (58%)** | **0%** |
| 41 (64%) | 0% |
| 47 (73%) | 0% |
| 53 (83%) | 36% |
| 56 (88%) | 70% |

**Key insights:**

- **Highest signal quality ≠ best layer.** Layers 53 and 56 scored highest on the per-layer signal-quality metric yet suppressed refusal the least — the same pattern as layer 21 vs. 16 on Qwen2.5-7B. Signal quality is a useful pre-filter, not a selector.
- **The winning layer transfers by relative depth.** Layer 37/64 (58%) sits at the same relative depth as Qwen2.5-7B's layer 16/28 (57%).
- **ARC-Easy can't see the capability cost.** It reads 1.000 before and after; MMLU-Pro shows a 3.3pp drop. Hybrid models also need their linear-attention output projections edited — orthogonalizing only `o_proj`/`down_proj` would silently leave 48 of 64 attention writes untouched.

## How it works

1. **`prepare_data`** (CPU) — builds harmful/harmless prompt sets and a capability MCQ subset on a shared Modal Volume.
2. **`extract_and_save_direction`** (B200) — one forward pass per prompt, computes a per-layer difference-in-means direction, optionally purifies it via Gram-Schmidt against the harmless direction.
3. **`run_sweep`** (B200) — broadcasts each candidate layer's direction model-wide, tests it across coefficients, grades refusal + capability.
4. **`run_full_pipeline`** (B200) — runs baseline, runtime-hook ablation, and permanent weight ablation at the sweep's winning operating point; outputs the refusal/capability tradeoff curve.
5. **`run_benchmarks`** (GPU) — MMLU-Pro accuracy and harmless-prompt KL divergence for the base model and the same weight edit, in one container.
6. **`flagship_run`** (B200) — independent smoke test confirming the technique and memory footprint hold at 30B+ scale.
7. **`push_checkpoint_to_hub`** (CPU) — reshards any checkpoint shard over 5 GB (streaming, no full-model RAM), then uploads the folder as-is.

**Design decisions:**

- **Global broadcasting.** Directions are computed per layer but the single best layer's vector is broadcast model-wide — a single-layer edit alone gets reintroduced downstream through the residual stream.
- **Two ablation methods, proven equivalent.** The reversible forward-hook and the permanent weight edit are tested to produce numerically identical output for a single layer, so the weight edit can be trusted to reproduce what the hook demonstrated.
- **Two-tier refusal grading.** A cheap rule-based pass handles clear cases; only ambiguous responses escalate to an LLM-judge fallback.
- **Fixed sweep grid.** Layer/coefficient candidates are a bounded shortlist, not an exhaustive search, so sweep cost stays predictable regardless of model depth.
- **Architecture-aware weight edit.** Residual-write projections include `linear_attn.out_proj` for hybrid models. For sandwich-norm models (Gemma 2/3), where an RMSNorm sits between the projection and the residual add, the projection is orthogonalized against `normalize(g ⊙ d̂)` — the norm's per-channel scale would otherwise rotate a `d̂` component back in.
- **Non-thinking mode.** Prompts use `enable_thinking=False`, so hybrid-reasoning models answer directly instead of spending the generation budget inside `<think>`.

## Future work

- Fix the weight-ablated self-judge bias by judging against a pre-mutation model snapshot.
- Implement norm-preserving weight ablation to protect model capability, and test whether it (or a gentler coefficient like 0.7) recovers Qwen3.8-27B's 3.3pp MMLU-Pro drop.
- Build a general baseline-vs-ablated qualitative compare tool.
- Re-run the full pipeline on Llama-3.1-8B-Instruct post chat-template-fix.
- Add serving/inference for a saved checkpoint (deferred; preferred design is a bearer-token-gated endpoint).

## Development & Testing

```bash
# Full unit test suite (pure PyTorch, CPU-only, no GPU/network required)
python3.11 -m pytest tests/ -v

# Syntax/import check without deploying
python3.11 -m py_compile modal_app.py
python3.11 -c "import modal_app"
```

60 unit tests on CPU — no GPU, no network, no HF access needed. Architecture tests build tiny random Gemma 3 / Qwen3.5 models from `transformers` configs. No linter or build step is configured — this is a script/pipeline project, not a packaged library.

## Environment Variables

This project uses **Modal Secrets**, not a `.env` file — nothing reads `os.environ` directly except `HF_HOME`, which the code sets itself for cache placement.

| Variable | Description | Default / Required |
|---|---|---|
| `HF_TOKEN` | Hugging Face access token, injected via the `huggingface-secret` Modal secret (`modal secret create huggingface-secret HF_TOKEN=...`). Read access is enough for every function except `push_checkpoint_to_hub`, which needs a **write-scoped** token (regenerate the same secret with `--force` before publishing). | **Required** — every `@app.function` in `modal_app.py` declares `secrets=[HF_SECRET]` |
| `HF_HOME` | Where Hugging Face caches downloads inside the Modal container. | Set automatically to `/cache/huggingface` (persistent `abliteration-eval-cache` volume) — not user-configurable |

## Project Structure

```
AblateBench/
├── modal_app.py              # Entry point — all Modal functions (prepare_data, extract,
│                              # run_sweep, run_full_pipeline, run_benchmarks,
│                              # flagship_run, push_checkpoint_to_hub) + local_entrypoint
├── src/
│   ├── model_utils.py        # Shared decoder-layer introspection, used by every stage below
│   ├── extract_direction.py  # Activation extraction + per-layer direction computation,
│   │                          # direction purification, signal-quality scoring
│   ├── layer_sweep.py        # Layer/coefficient sweep
│   ├── ablate_runtime.py     # Reversible forward-hook ablation
│   ├── ablate_weights.py     # Permanent weight-level ablation
│   ├── rescale.py            # Capability rescaling + tradeoff-curve tracking
│   ├── eval_refusal.py       # Two-tier refusal grader (rule-based + judge fallback)
│   ├── eval_capability.py    # MCQ capability harness (ARC-Easy, MMLU-Pro; up to 10 options)
│   ├── eval_kl.py            # Base-vs-ablated KL divergence on harmless prompts
│   └── checkpoint_utils.py   # Streaming safetensors resharding for Hub upload
├── data/
│   ├── prepare_prompts.py           # Downloads/splits AdvBench + Alpaca (needs network)
│   ├── build_eval_categories.py     # Generates ~292 category-labeled refusal-eval prompts
│   ├── build_capability_subset.py   # Real ARC-Easy sample (falls back to a static set offline)
│   ├── build_mmlu_pro_subset.py     # Category-stratified MMLU-Pro sample (needs network)
│   ├── eval_categories/             # Generated output, 12 categories
│   └── capability_subset.json       # Generated output, 50-100 MCQ capability subset
├── tests/                    # 60 unit tests, CPU only, no GPU/network needed
└── requirements.txt
```

## License

[MIT](LICENSE)
