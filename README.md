# AblateBench

**Find the part of an AI model that makes it refuse, remove it, and measure what changed.**

`Python 3.11` · `PyTorch` · `Transformers` · `Modal` · `NVIDIA H100` · `pytest`

## What is this?

Chat models like Qwen or Llama are trained to refuse some requests ("I can't help with that"). Research from 2024 ([Arditi et al.](https://arxiv.org/abs/2406.11717)) found that this refusal behaviour is controlled by a surprisingly simple thing inside the model: a single *direction* in its internal activations. You can picture it as one dial among thousands. When the model reads a request it decides to refuse, that dial turns up.

**Abliteration** is the technique of finding that dial and permanently removing it from the model's weights, with no retraining. The model keeps its knowledge and skills; it just stops refusing.

The hard part is not removing the dial. It's **proving that nothing else broke**, and measuring refusal honestly. AblateBench is a full pipeline for both:

1. **Find** the refusal direction by comparing how the model processes harmful vs. harmless prompts.
2. **Remove** it from every place the model writes it, as a permanent weight edit.
3. **Measure** the result with a test battery:
   - refusal rate across 12 categories
   - general knowledge (MMLU-Pro)
   - math (GSM8K)
   - security skills
   - how much normal, harmless answers changed

Everything runs on rented cloud GPUs through [Modal](https://modal.com), at roughly $5–10 per model.

## Results

| Model | Type | Refusals before → after | Capability | Model |
|---|---|---|---|---|
| **Qwen3.5-35B-A3B** | Mixture-of-Experts, 256 experts | 99.7% → **0%** ¹ | Unchanged (MMLU-Pro 62.9% → 63.1%, GSM8K 94–95%) | [link](https://huggingface.co/rajaykumar12959/qwen3.5-35b-a3b-abliterated) |
| **Qwen3.8-27B** | Hybrid attention | 99.3% → **0%** | Small cost (MMLU-Pro 65.1% → 61.8%) | [link](https://huggingface.co/rajaykumar12959/qwen3.8-27b-abliterated) |
| **Qwen2.5-7B-Instruct** | Dense, first prototype | ~96% → 43% | Unchanged on the quick check (ARC-Easy) | [link](https://huggingface.co/rajaykumar12959/qwen2.5-7b-abliterated) |

¹ Measured on 292 test prompts with full-length (256-token) answers, and also with the model's "thinking" mode on. If answers are cut at 128 tokens it reads 3.4%, because a few answers open with a disclaimer before complying. The model card reports the 128-token number.

You can chat with the 7B and 27B models in the [Hugging Face Space](https://huggingface.co/spaces/rajaykumar12959/Abliterated).

## What makes this project different

- **It measures carefully.** A naive grader that looks for "I cannot" overcounts badly: on the 35B model, it reported 22.9% refusals where most replies were real answers with a disclaimer on top. AblateBench grades each reply with an AI judge that sees the original request. The judge is always the *unmodified* model, so the edited model never grades itself. Every graded refusal is also saved, so a person can check it.
- **It proves nothing else broke.** Each model is tested on general knowledge, math, security tasks, and how much its answers to ordinary prompts changed. Easy tests hide damage. The 27B model scored a perfect 1.000 on a simple quiz before and after, while a harder test (MMLU-Pro) showed a 3.3-point drop.
- **It handles modern architectures.**
  - *Mixture-of-Experts models* route each word to a few of 256 "expert" sub-networks, and every expert has to be edited.
  - *Hybrid attention models* have two kinds of attention layers.
  - *Thinking models* reason before answering.
  - Each case has tests that fail if the edit misses part of the model.
- **It searches instead of guessing.** For the hardest model (the 35B MoE), the simple recipe of one direction at full strength everywhere left 14–88% refusals, depending on the layer. An automatic optimizer (inspired by [Heretic](https://github.com/p-e-w/heretic)) instead tunes how strongly each layer is edited. It reached 0% refusal while changing harmless answers less than half as much as the simple recipe did.

## Quickstart

You need Python 3.11, a [Modal](https://modal.com) account and a [Hugging Face](https://huggingface.co) token.

```bash
git clone https://github.com/Rajaykumar12/AblateBench.git
cd AblateBench
pip install -r requirements.txt

modal setup                                                     # one-time login
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxx     # your HF token
```

Abliterate a small model end to end:

```bash
M=Qwen/Qwen2.5-7B-Instruct
modal run modal_app.py --stage prepare_data                                     # download prompt sets
modal run --detach modal_app.py --stage extract --model-name $M --purified      # find the direction
modal run --detach modal_app.py --stage sweep   --model-name $M --purified      # pick the best layer
modal run --detach modal_app.py --stage full    --model-name $M --purified \
    --best-layer 16 --save-checkpoint                                           # edit, evaluate, save
```

`--detach` keeps a job running on Modal even if your laptop sleeps. Watch it with `modal app logs <app-id>`.

For larger models, the optimizer, the full evaluation, and publishing to Hugging Face, see **[TECHNICAL.md](TECHNICAL.md)**.

## Responsible use

Abliteration removes safety behaviour on purpose. Models produced with it **will comply with harmful requests** that the original model refuses. This project exists for research:
- understanding how refusal works inside language models
- testing how robust safety training is
- building better evaluations for both

Released models carry this warning on their model cards. If you deploy one, you are responsible for how it's used.

## Learn more

- **[TECHNICAL.md](TECHNICAL.md)**:
  - how each pipeline stage works
  - the full command reference
  - per-model results and findings
  - MoE and router-level editing
  - grader design
  - testing
- Arditi et al., [*Refusal in Language Models Is Mediated by a Single Direction*](https://arxiv.org/abs/2406.11717) (2024): the paper this builds on.

## License

- **Code:** [Apache-2.0](LICENSE). See [NOTICE](NOTICE).
- **Released models:** Apache-2.0, inherited from the Qwen base models. Each model card states its license.
- **Datasets:** downloaded at runtime and kept under their own licenses. Note that [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) is CC BY-NC 4.0 (non-commercial); check dataset terms before any commercial use.
