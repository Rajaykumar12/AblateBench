#!/usr/bin/env python3.11
"""Plot the Pareto frontier of a refusal-rate/capability tradeoff curve.

Reads either:
  - a `sweep_result_*.json` (the direct output of SweepResult.as_table_rows(): a plain
    list of {"layer", "coefficient", "refusal_rate", "capability_score"} rows), or
  - a `results_*.json` (the output of run_full_pipeline: a dict with a
    "tradeoff_curve" key holding TradeoffCurve.as_table_rows() rows) via --tradeoff-curve.

Computes the Pareto frontier (src/pareto.py) over all points, scatter-plots every
point, highlights the frontier, and connects frontier points with a dashed line
(sorted by refusal_rate so the line traces the curve left-to-right). Local-only —
matplotlib is not part of the Modal image; this reads a JSON file already downloaded
from a Modal run and never touches Modal itself.

Usage:
    python3.11 scripts/plot_pareto.py sweep_result_<tag>.json --out /tmp/pareto.png
    python3.11 scripts/plot_pareto.py results_<tag>.json --tradeoff-curve --out /tmp/pareto.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable as top-level modules, matching tests/conftest.py's convention.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pareto import pareto_frontier  # noqa: E402


def _load_points(path: Path, tradeoff_curve: bool) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if tradeoff_curve:
        return data["tradeoff_curve"]
    return data


def plot_pareto(points: list[dict], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    frontier = pareto_frontier(points)
    frontier_keys = {id(p) for p in frontier}
    dominated = [p for p in points if id(p) not in frontier_keys]

    fig, ax = plt.subplots(figsize=(7, 5))

    if dominated:
        ax.scatter(
            [p["refusal_rate"] for p in dominated],
            [p["capability_score"] for p in dominated],
            color="tab:gray",
            alpha=0.6,
            label="dominated",
        )

    frontier_sorted = sorted(frontier, key=lambda p: p["refusal_rate"])
    ax.scatter(
        [p["refusal_rate"] for p in frontier_sorted],
        [p["capability_score"] for p in frontier_sorted],
        color="tab:orange",
        edgecolors="black",
        zorder=3,
        label="Pareto frontier",
    )
    ax.plot(
        [p["refusal_rate"] for p in frontier_sorted],
        [p["capability_score"] for p in frontier_sorted],
        color="tab:orange",
        linestyle="--",
        zorder=2,
    )

    ax.set_xlabel("refusal_rate (lower is better)")
    ax.set_ylabel("capability_score (higher is better)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path} ({len(frontier)}/{len(points)} points on the frontier)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="sweep_result_*.json or results_*.json")
    parser.add_argument(
        "--tradeoff-curve",
        action="store_true",
        help="Read points from the input's \"tradeoff_curve\" key (results_*.json shape) instead of treating the input as a plain list (sweep_result_*.json shape).",
    )
    parser.add_argument("--out", type=Path, default=Path("pareto.png"), help="Output PNG path (default: pareto.png)")
    args = parser.parse_args()

    points = _load_points(args.input, args.tradeoff_curve)
    plot_pareto(points, args.out, title=args.input.name)


if __name__ == "__main__":
    main()
