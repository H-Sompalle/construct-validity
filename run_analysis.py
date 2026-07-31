#!/usr/bin/env python3
"""Run construct validity analyses and regenerate tables/figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analyze import run_all_analyses
from src.data import BENCHMARK_LABELS
from src.figures import generate_all_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run construct validity analyses across agentic AI benchmarks."
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
    results["percentile_spread_table"].to_csv(
        tables_dir / "percentile_rank_spreads.csv", index=False
    )
    results["ranks"].to_csv(tables_dir / "model_ranks.csv")
    results["percentile_ranks"].to_csv(tables_dir / "model_percentile_ranks.csv")

    cd = results["convergent_discriminant"]
    corr_ci_lines = [
        f"  {BENCHMARK_LABELS[r.benchmark_a]} × {BENCHMARK_LABELS[r.benchmark_b]}: "
        f"ρ={r.rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] (n={r.n}, p={r.p_value:.4f})"
        for r in results["correlations"]
    ]
    summary_lines = [
        "Construct Validity Audit — Summary Statistics",
        "=" * 48,
        f"Mean Spearman ρ (all 10 pairs):     {cd['mean_all_pairs']:.2f}",
        f"Mean within-agent ρ:                 {cd['mean_within_agent']:.2f}",
        f"Mean cross-domain ρ:                {cd['mean_cross_domain']:.2f}",
        f"Mean inversion rate:                 {results['inversion_table'].iloc[-1]['rate_pct']:.1f}%",
        "",
        "Pairwise Spearman ρ with bootstrap 95% CIs (10,000 resamples):",
        *corr_ci_lines,
        "",
        "Correlation matrix:",
        results["correlation_matrix"].round(2).to_string(),
        "",
        "Ranking inversions:",
        results["inversion_table"].to_string(index=False),
        "",
        "Percentile rank spreads (0=best, 100=worst; models with ≥4 benchmarks):",
        results["percentile_spread_table"].to_string(index=False),
        "",
        "Raw rank spreads (legacy):",
        results["rank_spread_table"].to_string(index=False),
    ]
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    figure_paths = generate_all_figures(
        figures_dir, results["scores"], correlations=results["correlations"]
    )

    print(f"Wrote tables to {tables_dir}")
    print(f"Wrote summary to {summary_path}")
    for name, path in figure_paths.items():
        print(f"Wrote {name} to {path}")


if __name__ == "__main__":
    main()
