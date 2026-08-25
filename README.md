# Mini-Abliteration + Refusal-Rate Eval Harness

Reproduction/extension of the difference-in-means refusal-direction technique from
Arditi et al., ["Refusal in Language Models Is Mediated by a Single
Direction"](https://arxiv.org/abs/2406.11717) (arXiv:2406.11717, 2024; reference
implementation: [github.com/andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)).
Full requirements, architecture, execution plan, and test strategy are in
[`../abliteration-eval-prd.md`](../abliteration-eval-prd.md) — this README is filled
in with real numbers as each pipeline stage completes on Modal, per PRD §11.

## Status

Implementation scaffold complete and unit-tested locally (29/29 tests passing,
plus an end-to-end smoke test against a real `transformers` Llama architecture with
random weights — see `tests/` and the smoke test referenced in the session log).
**No GPU runs have executed yet** — the sections below are placeholders to be filled
in once `modal run` stages complete on Modal's B200s.

## Running

```bash
pip install -r requirements.txt
modal setup                      # one-time auth (opens a browser to modal.com)

modal run modal_app.py --stage prepare_data
modal run modal_app.py --stage extract --model-name Qwen/Qwen2.5-7B-Instruct
modal run modal_app.py::run_sweep --model-name Qwen/Qwen2.5-7B-Instruct \
    --directions-path /cache/directions_Qwen_Qwen2.5-7B-Instruct.pt
# inspect the sweep table, pick best (layer, coefficient), then:
modal run modal_app.py::run_full_pipeline --model-name Qwen/Qwen2.5-7B-Instruct \
    --best-layer <N> --best-coefficient <C>

modal run modal_app.py --stage flagship --model-name Qwen/Qwen2.5-32B-Instruct
```

Local unit tests (no GPU/network needed):

```bash
python3.11 -m pytest tests/ -v
```

## Refusal-rate before/after (per category, per model)

_TODO — fill in after `run_full_pipeline` completes for both models._

## Capability retention (with/without rescaling) — tradeoff curve

_TODO — fill in from `results_<model>.json`'s `tradeoff_curve` field._

## Per-layer sweep results

_TODO — fill in from `sweep_result_<model>.json`._

## Second-model divergence

_TODO — what transferred from model 1's best (layer, coefficient), what didn't, and
what was changed to compensate._

## Flagship 30B+ confirmation

_TODO — fill in from `flagship_run`'s summary (peak GPU memory, fraction of B200's
~180GB HBM used)._

## Grader validation

_TODO — rule-based vs. manual-label agreement rate; judge-fallback invocation rate
(PRD §9.2)._

## Citation

Arditi, A. et al., "Refusal in Language Models Is Mediated by a Single Direction,"
arXiv:2406.11717 (2024). This project is a reproduction/extension at smaller scale
with an added 30B+ confirmation run; it makes no claim of novel methodology.
