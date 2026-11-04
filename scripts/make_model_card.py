"""Render a Hugging Face model card (README.md) for an abliterated checkpoint from
the JSON written by `modal run modal_app.py --stage compare`.

Stdlib only, so it runs locally without the Modal/torch stack:

    python scripts/make_model_card.py \
        --compare-json compare_Qwen_Qwen3.5-35B-A3B_vs_you_Qwen3.5-35B-A3B-abliterated.json \
        --repo-id you/Qwen3.5-35B-A3B-abliterated \
        --layer 36 --coefficient 0.7 --purified \
        --out README.md
"""

import argparse
import json


def _pct(x) -> str:
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _delta_pp(base, abl) -> str:
    if not isinstance(base, (int, float)) or not isinstance(abl, (int, float)):
        return "n/a"
    return f"{100 * (abl - base):+.1f} pp"


def _row(name: str, base, abl) -> str:
    return f"| {name} | {_pct(base)} | {_pct(abl)} | {_delta_pp(base, abl)} |"


def render_card(
    results: dict,
    repo_id: str,
    layer: int,
    coefficient: float,
    purified: bool,
    license_id: str = "apache-2.0",
    source_url: str = "",
    eval_note: str = "",
) -> str:
    base_id = results["baseline_id"]
    refusal = results["refusal_by_category"]
    base_ref, abl_ref = refusal["baseline"], refusal["abliterated"]
    mmlu = results["mmlu_pro"]
    defensive = results["defensive"]
    kl = results["kl_harmless"]

    header = "| Metric | Base | Abliterated | Δ |\n|---|---|---|---|"
    categories = sorted(c for c in base_ref if c != "aggregate")
    refusal_rows = [_row(c, base_ref.get(c), abl_ref.get(c)) for c in categories]
    refusal_rows.append(_row("**aggregate**", base_ref.get("aggregate"), abl_ref.get("aggregate")))

    mmlu_cats = sorted(mmlu["baseline"].get("by_category", {}))
    mmlu_rows = [
        _row(c, mmlu["baseline"]["by_category"].get(c), mmlu["abliterated"].get("by_category", {}).get(c))
        for c in mmlu_cats
    ]

    sec, vuln, patch = defensive["security_mcq"], defensive["vuln_detection"], defensive["patch_synthesis"]
    direction_kind = "purified (Gram-Schmidt orthogonalized against the harmless mean)" if purified else "raw"

    lines = [
        "---",
        f"base_model: {base_id}",
        f"license: {license_id}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "  - abliterated",
        "  - refusal-direction-ablation",
        "  - moe",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"Weight-ablated (\"abliterated\") variant of [{base_id}](https://huggingface.co/{base_id}). "
        "A single refusal direction was projected out of every residual-stream write "
        "(token embeddings, attention / linear-attention output projections, and all MoE expert "
        "down-projections), with per-row norm preservation. No fine-tuning; architecture and "
        "tokenizer are unchanged.",
        "",
        "## Method",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Base model | `{base_id}` |",
        f"| Direction source layer | {layer} |",
        f"| Ablation coefficient | {coefficient} |",
        f"| Direction | {direction_kind} |",
        "| Norm preservation | yes (rows rescaled to their original L2 norm) |",
        "| Edited modules | token embeddings + residual-write projections in every decoder layer, incl. all experts |",
        "",
        *([f"Produced with [AblateBench]({source_url}).", ""] if source_url else []),
        "## Benchmarks",
        "",
        "Base and abliterated models were evaluated by identical code on identical data."
        + (f" {eval_note}" if eval_note else ""),
        "",
        "### Refusal rate by category (lower = fewer refusals)",
        "",
        f"Judge: {refusal.get('judge', 'n/a')}. {refusal.get('caveat', '')}",
        "",
        header,
        *refusal_rows,
        "",
        f"### General capability: MMLU-Pro ({mmlu.get('num_questions', 'n/a')} questions)",
        "",
        header,
        _row("**overall**", mmlu["baseline"].get("accuracy"), mmlu["abliterated"].get("accuracy")),
        *mmlu_rows,
        "",
        "### Defensive-security capability",
        "",
        header,
        _row("Security MCQ", sec["baseline"].get("accuracy"), sec["abliterated"].get("accuracy")),
        _row("Vulnerability detection", vuln["baseline"].get("accuracy"), vuln["abliterated"].get("accuracy")),
        _row("Patch synthesis (fix rate)", patch["baseline"].get("fix_rate"), patch["abliterated"].get("fix_rate")),
        "",
        f"Patch grading judge: {patch.get('judge', 'n/a')}.",
        "",
        f"### Distribution shift on harmless prompts ({kl.get('num_prompts', 'n/a')} prompts)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean first-token KL(base ‖ abliterated) | {kl['mean_kl_base_to_abliterated_nats']:.4f} nats |",
        f"| First-token top-1 agreement | {_pct(kl['first_token_top1_agreement'])} |",
        "",
        "## Usage",
        "",
        "```python",
        "from transformers import AutoModelForImageTextToText, AutoTokenizer",
        "",
        f'model = AutoModelForImageTextToText.from_pretrained("{repo_id}", dtype="bfloat16", device_map="auto")',
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")',
        "```",
        "",
        "## Limitations and intended use",
        "",
        "- This model's refusal behaviour has been deliberately reduced. It will comply with "
        "requests the base model declines; you are responsible for how it is deployed and used.",
        "- Intended for research on refusal mechanisms, red-teaming, and evaluation.",
        "- Benchmarks are small, fixed subsets; treat deltas of a few points as noise.",
        "- All other limitations and the license of the base model apply.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare-json", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--coefficient", type=float, required=True)
    p.add_argument("--purified", action="store_true")
    p.add_argument("--license", default="apache-2.0", help="must match the base model's license")
    p.add_argument("--source-url", default="", help="optional link to the code that produced the model")
    p.add_argument("--eval-note", default="", help="optional sentence on eval settings")
    p.add_argument("--out", default="README.md")
    args = p.parse_args()

    with open(args.compare_json) as f:
        results = json.load(f)
    card = render_card(
        results, args.repo_id, args.layer, args.coefficient, args.purified, args.license,
        source_url=args.source_url, eval_note=args.eval_note,
    )
    with open(args.out, "w") as f:
        f.write(card)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
