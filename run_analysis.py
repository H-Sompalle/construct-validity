#!/usr/bin/env python3
"""Run construct validity analyses and regenerate paper tables/figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analyze import run_all_analyses
from src.figures import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce results from 'Construct Validity Failures in Agentic AI Benchmarks'."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for tables and figures",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_analyses()

    results["correlation_matrix"].to_csv(tables_dir / "correlation_matrix.csv")
    results["correlation_table"].to_csv(tables_dir / "correlations.csv", index=False)
    results["inversion_table"].to_csv(tables_dir / "inversion_rates.csv", index=False)
    results["rank_spread_table"].to_csv(tables_dir / "rank_spreads.csv", index=False)
    results["ranks"].to_csv(tables_dir / "model_ranks.csv")

    cd = results["convergent_discriminant"]
    summary_lines = [
        "Construct Validity Audit — Summary Statistics",
        "=" * 48,
        f"Mean Spearman ρ (all 10 pairs):     {cd['mean_all_pairs']:.2f}",
        f"Mean within-agent ρ:                 {cd['mean_within_agent']:.2f}",
        f"Mean cross-domain ρ:                {cd['mean_cross_domain']:.2f}",
        f"Mean inversion rate:                 {results['inversion_table'].iloc[-1]['rate_pct']:.1f}%",
        "",
        "Correlation matrix:",
        results["correlation_matrix"].round(2).to_string(),
        "",
        "Ranking inversions:",
        results["inversion_table"].to_string(index=False),
        "",
        "Rank spreads (models with ≥4 benchmarks):",
        results["rank_spread_table"].to_string(index=False),
    ]
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    figure_paths = generate_all_figures(figures_dir, results["scores"])

    print(f"Wrote tables to {tables_dir}")
    print(f"Wrote summary to {summary_path}")
    for name, path in figure_paths.items():
        print(f"Wrote {name} to {path}")


if __name__ == "__main__":
    main()
