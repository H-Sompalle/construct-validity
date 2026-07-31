"""Verify analysis outputs against independently transcribed expectations."""

from __future__ import annotations

import sys

from src.analyze import compute_ranks, rank_spreads, run_all_analyses
from src.data import BENCHMARK_COLUMNS, load_scores
from src.expectations import (
    CORRELATIONS,
    COVERAGE,
    DERIVED,
    EXPECTED_PERCENTILES,
    EXPECTED_RANKS,
    EXPECTED_SCORES,
    INVERSIONS,
    PROSE_CLAIMS,
    SUMMARY_STATS,
)


def _fail(failures: list[str], msg: str) -> None:
    failures.append(msg)


def verify_scores(scores, failures: list[str]) -> int:
    count = 0
    for model, expected_cols in EXPECTED_SCORES.items():
        for col, expected in expected_cols.items():
            count += 1
            actual = scores.loc[model, col]
            if expected is None:
                if not (actual != actual):  # NaN check
                    _fail(failures, f"score {model}/{col}: expected missing, got {actual}")
            elif actual != expected:
                _fail(failures, f"score {model}/{col}: expected {expected}, got {actual}")
    return count


def verify_ranks(ranks, spread_results, failures: list[str]) -> int:
    count = 0
    spreads = {r.model: r for r in spread_results}
    for model, expected in EXPECTED_RANKS.items():
        for col in BENCHMARK_COLUMNS:
            count += 1
            expected_rank = expected[col]
            actual_rank = ranks.loc[model, col]
            if expected_rank is None:
                if actual_rank == actual_rank:
                    _fail(failures, f"rank {model}/{col}: expected missing rank, got {int(actual_rank)}")
            elif int(actual_rank) != expected_rank:
                _fail(failures, f"rank {model}/{col}: expected rank {expected_rank}, got {int(actual_rank)}")
        count += 1
        expected_spread = expected["spread"]
        actual_spread = spreads[model].spread
        if actual_spread != expected_spread:
            _fail(failures, f"rank {model} spread: expected {expected_spread}, got {actual_spread}")
    return count


def verify_percentiles(spread_results, failures: list[str]) -> int:
    count = 0
    spreads = {r.model: r for r in spread_results}
    for model, expected in EXPECTED_PERCENTILES.items():
        r = spreads[model]
        for col in BENCHMARK_COLUMNS:
            count += 1
            expected_pct = expected[col]
            actual = r.percentile_ranks.get(col)
            if expected_pct is None:
                if actual is not None:
                    _fail(failures, f"percentile {model}/{col}: expected missing, got {actual}")
            else:
                actual_pct = round(100 * actual)
                if actual_pct != expected_pct:
                    _fail(
                        failures,
                        f"percentile {model}/{col}: expected {expected_pct}, got {actual_pct}",
                    )
        count += 1
        actual_spread = round(100 * r.percentile_spread)
        if actual_spread != expected["percentile_spread"]:
            _fail(
                failures,
                f"percentile {model} spread: expected {expected['percentile_spread']}, got {actual_spread}",
            )
    return count


def verify_correlations(corr_results, failures: list[str]) -> int:
    count = 0
    by_pair = {(r.benchmark_a, r.benchmark_b): r for r in corr_results}
    for pair, expected in CORRELATIONS.items():
        r = by_pair[pair]
        count += 1
        if round(r.rho, 2) != expected["rho"]:
            _fail(failures, f"ρ{pair}: expected {expected['rho']}, got {r.rho:.4f}")
        count += 1
        if r.n != expected["n"]:
            _fail(failures, f"n{pair}: expected {expected['n']}, got {r.n}")
        if expected.get("p_lt_001"):
            count += 1
            if not (r.p_value < 0.001):
                _fail(failures, f"p{pair}: expected < 0.001, got {r.p_value:.4f}")
        if expected.get("p_gt_005"):
            count += 1
            if not (r.p_value > 0.05):
                _fail(failures, f"p{pair}: expected > 0.05 (non-significant), got {r.p_value:.4f}")
        if "p_round_2" in expected:
            count += 1
            if round(r.p_value, 2) != expected["p_round_2"]:
                _fail(failures, f"p{pair}: expected ~{expected['p_round_2']}, got {r.p_value:.4f}")
        if "ci" in expected:
            count += 2
            exp_lo, exp_hi = expected["ci"]
            if round(r.ci_low, 2) != exp_lo:
                _fail(failures, f"CI lo{pair}: expected {exp_lo}, got {r.ci_low:.4f}")
            if round(r.ci_high, 2) != exp_hi:
                _fail(failures, f"CI hi{pair}: expected {exp_hi}, got {r.ci_high:.4f}")
    return count


def verify_inversions(inv_results, failures: list[str]) -> int:
    count = 0
    by_pair = {(r.benchmark_a, r.benchmark_b): r for r in inv_results}
    for pair, (exp_inv, exp_total, exp_rate) in INVERSIONS.items():
        r = by_pair[pair]
        count += 3
        if (r.inversions, r.total_pairs) != (exp_inv, exp_total):
            _fail(failures, f"inversions{pair}: expected {exp_inv}/{exp_total}, got {r.inversions}/{r.total_pairs}")
        if round(100 * r.rate, 1) != exp_rate:
            _fail(failures, f"inversion rate{pair}: expected {exp_rate}%, got {100*r.rate:.1f}%")
    return count


def verify_summary(results, failures: list[str]) -> int:
    count = 0
    cd = results["convergent_discriminant"]
    rhos = [r.rho for r in results["correlations"]]

    checks = [
        ("mean ρ all pairs", round(cd["mean_all_pairs"], 2), SUMMARY_STATS["mean_rho_all"]),
        ("mean within-agent ρ", round(cd["mean_within_agent"], 2), SUMMARY_STATS["mean_rho_within_agent"]),
        ("mean cross-domain ρ", round(cd["mean_cross_domain"], 2), SUMMARY_STATS["mean_rho_cross_domain"]),
        ("min ρ", round(min(rhos), 2), SUMMARY_STATS["rho_min"]),
        ("max ρ", round(max(rhos), 2), SUMMARY_STATS["rho_max"]),
        (
            "mean inversion rate",
            round(results["inversion_table"].iloc[-1]["rate_pct"], 1),
            SUMMARY_STATS["mean_inversion_rate_pct"],
        ),
        (
            "headline inversion rate (rounded)",
            round(results["inversion_table"].iloc[-1]["rate_pct"]),
            SUMMARY_STATS["abstract_inversion_rate_pct_rounded"],
        ),
    ]
    for label, actual, expected in checks:
        count += 1
        if actual != expected:
            _fail(failures, f"{label}: expected {expected}, got {actual}")
    return count


def verify_coverage(scores, failures: list[str]) -> int:
    count = 0
    for col, (exp_n, exp_total, exp_pct) in COVERAGE.items():
        count += 3
        n = int(scores[col].notna().sum())
        pct = round(100 * n / exp_total)
        if n != exp_n:
            _fail(failures, f"coverage {col}: expected {exp_n}/{exp_total}, got {n}/{exp_total}")
        if pct != exp_pct:
            _fail(failures, f"coverage {col}: expected {exp_pct}%, got {pct}%")
    return count


def verify_claims(ranks, scores, failures: list[str]) -> int:
    count = 0
    claim_checks = [
        ("o4-mini GPQA rank", int(ranks.loc["o4-mini", "gpqa"]), PROSE_CLAIMS["o4-mini_gpqa_rank"]),
        ("o4-mini MMLU rank", int(ranks.loc["o4-mini", "mmlu_pro"]), PROSE_CLAIMS["o4-mini_mmlu_rank"]),
        ("o4-mini τ-Retail rank", int(ranks.loc["o4-mini", "tau_retail"]), PROSE_CLAIMS["o4-mini_tau_retail_rank"]),
        ("o4-mini τ-Airline rank", int(ranks.loc["o4-mini", "tau_airline"]), PROSE_CLAIMS["o4-mini_tau_airline_rank"]),
        ("o1 MMLU rank", int(ranks.loc["o1", "mmlu_pro"]), PROSE_CLAIMS["o1_mmlu_rank"]),
        ("o1 GPQA rank", int(ranks.loc["o1", "gpqa"]), PROSE_CLAIMS["o1_gpqa_rank"]),
        ("o1 τ-Retail rank", int(ranks.loc["o1", "tau_retail"]), PROSE_CLAIMS["o1_tau_retail_rank"]),
        ("o1 τ-Airline rank", int(ranks.loc["o1", "tau_airline"]), PROSE_CLAIMS["o1_tau_airline_rank"]),
        ("o1 SWE rank", int(ranks.loc["o1", "swe_bench"]), PROSE_CLAIMS["o1_swe_rank"]),
        ("o3-mini MMLU rank", int(ranks.loc["o3-mini", "mmlu_pro"]), PROSE_CLAIMS["o3-mini_mmlu_rank"]),
        ("o3-mini GPQA rank", int(ranks.loc["o3-mini", "gpqa"]), PROSE_CLAIMS["o3-mini_gpqa_rank"]),
        ("o3-mini τ-Retail rank", int(ranks.loc["o3-mini", "tau_retail"]), PROSE_CLAIMS["o3-mini_tau_retail_rank"]),
        (
            "Claude Opus 4.1 τ-Retail rank",
            int(ranks.loc["Claude Opus 4.1", "tau_retail"]),
            PROSE_CLAIMS["claude_opus_41_tau_retail_rank"],
        ),
        (
            "Claude Opus 4.1 τ-Airline rank",
            int(ranks.loc["Claude Opus 4.1", "tau_airline"]),
            PROSE_CLAIMS["claude_opus_41_tau_airline_rank"],
        ),
    ]
    for label, actual, expected in claim_checks:
        count += 1
        if actual != expected:
            _fail(failures, f"claim {label}: expected {expected}, got {actual}")

    count += 1
    cs45_ranks = ranks.loc["Claude Sonnet 4.5", BENCHMARK_COLUMNS].dropna().astype(int)
    if not all(r == PROSE_CLAIMS["claude_sonnet_45_all_ranks"] for r in cs45_ranks):
        _fail(failures, f"claim Claude Sonnet 4.5 all ranks 1: got {dict(cs45_ranks)}")

    count += 2
    if not (scores.loc["GPT-4.5", "tau_retail"] > scores.loc["o3-mini", "tau_retail"]):
        _fail(failures, "claim GPT-4.5 should beat o3-mini on τ-Retail")
    if not (scores.loc["GPT-4.5", "swe_bench"] < scores.loc["o3-mini", "swe_bench"]):
        _fail(failures, "claim GPT-4.5 should lose to o3-mini on SWE-bench")
    count += 2
    if not (scores.loc["GPT-4.1", "tau_airline"] > scores.loc["o4-mini", "tau_airline"]):
        _fail(failures, "claim GPT-4.1 should beat o4-mini on τ-Airline")
    if not (scores.loc["GPT-4.1", "swe_bench"] < scores.loc["o4-mini", "swe_bench"]):
        _fail(failures, "claim GPT-4.1 should lose to o4-mini on SWE-bench")

    return count


def verify_derived(failures: list[str]) -> int:
    count = 1
    actual = round(1 - 0.75**2, 2)
    expected = DERIVED["tau_retail_tau_airline_unexplained_variance"]
    if actual != expected:
        _fail(failures, f"unexplained variance: expected {expected}, got {actual}")
    return count


def main() -> int:
    scores = load_scores()
    results = run_all_analyses(scores)
    ranks = compute_ranks(scores)
    spreads = rank_spreads(scores)
    failures: list[str] = []
    sections: list[tuple[str, int]] = []

    sections.append(("Scores (75 cells)", verify_scores(scores, failures)))
    sections.append(("Raw ranks & spreads", verify_ranks(ranks, spreads, failures)))
    sections.append(("Percentile ranks & spreads", verify_percentiles(spreads, failures)))
    sections.append(("Correlations (ρ, n, p, CI)", verify_correlations(results["correlations"], failures)))
    sections.append(("Inversions", verify_inversions(results["inversions"], failures)))
    sections.append(("Summary statistics", verify_summary(results, failures)))
    sections.append(("Coverage", verify_coverage(scores, failures)))
    sections.append(("Narrative claims", verify_claims(ranks, scores, failures)))
    sections.append(("Derived values", verify_derived(failures)))

    total = sum(n for _, n in sections)
    print(f"Ran {total} checks across {len(sections)} sections:\n")
    for name, n in sections:
        print(f"  {name}: {n} checks")

    if failures:
        print(f"\nVERIFICATION FAILED ({len(failures)} failures):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nAll {total} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
