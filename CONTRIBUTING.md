# Contributing to AblateBench

Thanks for your interest in improving AblateBench. Bug reports, new model architectures, better evaluations and documentation fixes are all welcome.

Please read the [Responsible use](README.md#responsible-use) section first. AblateBench is a research tool for studying how refusal works in language models and how robust safety training is. Contributions whose main purpose is making harmful output easier to get, such as prompt collections of real attack instructions or "jailbreak packs", will not be accepted.

## Getting set up

You need Python 3.11. GPU work runs on [Modal](https://modal.com), but **you do not need Modal, a GPU or a Hugging Face token to work on most of the code**: the test suite runs on CPU using tiny random models.

```bash
git clone https://github.com/Rajaykumar12/AblateBench.git
cd AblateBench
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

To run real pipeline stages, also follow the Modal and Hugging Face steps in the [Quickstart](README.md#quickstart).

## Running the tests

```bash
python3.11 -m pytest tests/ -q          # CPU only, no GPU / network / HF access
python3.11 -m py_compile modal_app.py   # modal_app.py is mostly exercised on Modal, so at least check it compiles
```

`tests/test_compare.py` is skipped when `modal` isn't installed. `tests/conftest.py` puts `src/` and `data/` on the import path, so modules are imported by name (`import ablate_weights`), the same way `modal_app.py` imports them inside the container.

All tests must pass before a pull request is merged.

## Where things live

- `modal_app.py`: every Modal stage plus the `main` CLI entrypoint
- `src/`: the core library (direction extraction, hooks, weight edits, MoE routing, evaluators)
- `data/`: dataset builders and the generated refusal-category prompts
- `scripts/`: utilities such as model-card generation
- `tests/`: CPU unit tests

[TECHNICAL.md](TECHNICAL.md) explains each stage, the full command reference, the grader design and the project structure in detail.

## Making a change

1. **Open an issue first** for anything bigger than a small fix, so we can agree on the approach before you spend GPU time.
2. **Branch from `master`** with a descriptive name, e.g. `gemma3-sandwich-norm-fix`.
3. **Add or update tests.** Ablation code fails silently: a wrong axis or a leaky rescale still produces a model, just a worse one. Every change to the math should come with a CPU test on a toy model that checks the property directly, for example that the edited weights satisfy `d̂ᵀW' = 0` or that hook-based and weight-based edits give the same outputs.
4. **Keep the results reproducible.** If a change affects refusal, KL or capability numbers, say which model, layer and flags you ran, and include the `compare` output (or the relevant JSON) in the pull request.
5. **Update the docs.** New CLI flags, stages or metrics belong in [TECHNICAL.md](TECHNICAL.md). The [README](README.md) is for newcomers, so only change it when the user-facing story changes.

### Adding support for a new model architecture

Architecture discovery lives in `src/model_utils.py`. See `tests/test_gemma_support.py` and `tests/test_hybrid_attention_support.py` for the pattern: build a tiny random config of the new architecture, then check that decoder layers and residual-write projections are found and that the edit is applied correctly.

### Adding or changing an evaluation

Evaluators live in `src/eval_*.py`. Keep graders deterministic where possible and document any change to scoring rules, because they change how results compare across models. If you touch the refusal grader, add cases for both false positives (compliant answers flagged as refusals) and false negatives.

## Code style

- Follow the style of the surrounding code: type hints on public functions, short docstrings explaining *why*, not *what*.
- Keep functions usable without Modal. GPU- and Modal-specific logic belongs in `modal_app.py`; `src/` should stay importable and testable on CPU.
- Don't commit large files: checkpoints (`*.pt`, safetensors), sweep outputs (`sweep-results/`) and generated graphs are git-ignored for a reason.
- Write commit messages in the imperative mood ("Add router orthogonalization", not "Added…").

## Pull requests

A good pull request:
- explains what changed and why
- links the related issue
- lists how you tested it: unit tests, plus any Modal runs and their results
- keeps unrelated changes out

## Reporting bugs

Open a GitHub issue with:
- the model name and the exact `modal run` command
- the error output or the unexpected metrics
- your package versions (`pip freeze | grep -E "torch|transformers|modal"`)

For grader problems, include the prompt and the response that was graded wrongly.

## License

AblateBench is licensed under [Apache-2.0](LICENSE). By submitting a contribution, you agree that it is licensed under the same terms (Apache-2.0, Section 5).
