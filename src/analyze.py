"""Construct validity analyses from the KDD workshop paper."""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from .data import AGENT_BENCHMARKS, BENCHMARK_COLUMNS, BENCHMARK_LABELS, load_scores


class CorrelationResult(NamedTuple):
    benchmark_a: str
    benchmark_b: str
    rho: float
    p_value: float
    n: int


class InversionResult(NamedTuple):
    benchmark_a: str
    benchmark_b: str
    inversions: int
    total_pairs: int
    rate: float


class RankSpreadResult(NamedTuple):
    model: str
    ranks: dict[str, int]
    spread: int


def pairwise_spearman(scores: pd.DataFrame) -> list[CorrelationResult]:
    """Spearman rank correlations on pairwise-complete observations."""
    results: list[CorrelationResult] = []
    for a, b in combinations(BENCHMARK_COLUMNS, 2):
        pair = scores[[a, b]].dropna()
        n = len(pair)
        if n < 2:
            rho, p = np.nan, np.nan
        else:
            rho, p = stats.spearmanr(pair[a], pair[b])
        results.append(CorrelationResult(a, b, float(rho), float(p), n))
    return results


def correlation_matrix(scores: pd.DataFrame) -> pd.DataFrame:
    """Symmetric Spearman correlation matrix with benchmark labels."""
    mat = pd.DataFrame(index=BENCHMARK_COLUMNS, columns=BENCHMARK_COLUMNS, dtype=float)
    for col in BENCHMARK_COLUMNS:
        mat.loc[col, col] = 1.0
    for result in pairwise_spearman(scores):
        mat.loc[result.benchmark_a, result.benchmark_b] = result.rho
        mat.loc[result.benchmark_b, result.benchmark_a] = result.rho
    mat.index = [BENCHMARK_LABELS[c] for c in mat.index]
    mat.columns = [BENCHMARK_LABELS[c] for c in mat.columns]
    return mat


def mean_correlation(results: list[CorrelationResult]) -> float:
    values = [r.rho for r in results if not np.isnan(r.rho)]
    return float(np.mean(values)) if values else float("nan")


def convergent_discriminant_analysis(scores: pd.DataFrame) -> dict[str, float]:
    """Compare mean within-agent vs cross-domain Spearman correlations."""
    results = pairwise_spearman(scores)
    within_agent: list[float] = []
    cross_domain: list[float] = []
    for r in results:
        a_agent = r.benchmark_a in AGENT_BENCHMARKS
        b_agent = r.benchmark_b in AGENT_BENCHMARKS
        if a_agent and b_agent:
            within_agent.append(r.rho)
        elif (a_agent and not b_agent) or (b_agent and not a_agent):
            cross_domain.append(r.rho)
    return {
        "mean_within_agent": float(np.mean(within_agent)),
        "mean_cross_domain": float(np.mean(cross_domain)),
        "mean_all_pairs": mean_correlation(results),
    }


def ranking_inversions(scores: pd.DataFrame) -> list[InversionResult]:
    """Count model pairs whose relative order swaps between two benchmarks.

    Tied scores on either benchmark are not counted as inversions, but tied
    pairs remain in the denominator (all C(n, 2) model pairs).
    """
    results: list[InversionResult] = []
    for a, b in combinations(BENCHMARK_COLUMNS, 2):
        pair = scores[[a, b]].dropna()
        models = pair.index.tolist()
        inversions = 0
        total = len(models) * (len(models) - 1) // 2
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                s1a, s2a = pair.loc[m1, a], pair.loc[m2, a]
                s1b, s2b = pair.loc[m1, b], pair.loc[m2, b]
                if s1a == s2a or s1b == s2b:
                    continue
                if (s1a > s2a) != (s1b > s2b):
                    inversions += 1
        rate = inversions / total if total else float("nan")
        results.append(InversionResult(a, b, inversions, total, rate))
    return results


def compute_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-benchmark ranks (1 = best) among models with observed scores."""
    ranks = pd.DataFrame(index=scores.index, columns=BENCHMARK_COLUMNS, dtype=float)
    for col in BENCHMARK_COLUMNS:
        observed = scores[col].dropna().sort_values(ascending=False, kind="mergesort")
        ranks.loc[observed.index, col] = range(1, len(observed) + 1)
    return ranks


def rank_spreads(scores: pd.DataFrame, min_benchmarks: int = 4) -> list[RankSpreadResult]:
    """Rank spread (max rank − min rank) for models scored on ≥ min_benchmarks."""
    ranks = compute_ranks(scores)
    results: list[RankSpreadResult] = []
    for model in scores.index:
        model_ranks = ranks.loc[model].dropna()
        if len(model_ranks) < min_benchmarks:
            continue
        rank_dict = {col: int(model_ranks[col]) for col in model_ranks.index}
        spread = int(model_ranks.max() - model_ranks.min())
        results.append(RankSpreadResult(model, rank_dict, spread))
    results.sort(key=lambda r: r.spread, reverse=True)
    return results


def format_correlation_table(results: list[CorrelationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "benchmark_a": BENCHMARK_LABELS[r.benchmark_a],
                "benchmark_b": BENCHMARK_LABELS[r.benchmark_b],
                "rho": round(r.rho, 2),
                "p_value": r.p_value,
                "n": r.n,
                "bonferroni_significant": r.p_value < 0.005,
                "low_power": r.n < 8,
            }
        )
    return pd.DataFrame(rows)


def format_inversion_table(results: list[InversionResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "benchmark_a": BENCHMARK_LABELS[r.benchmark_a],
                "benchmark_b": BENCHMARK_LABELS[r.benchmark_b],
                "inversions": r.inversions,
                "total_pairs": r.total_pairs,
                "rate_pct": round(100 * r.rate, 1),
            }
        )
    df = pd.DataFrame(rows)
    mean_rate = df["rate_pct"].mean()
    df.loc[len(df)] = {
        "benchmark_a": "Mean",
        "benchmark_b": "across all pairs",
        "inversions": "",
        "total_pairs": "",
        "rate_pct": round(mean_rate, 1),
    }
    return df


def format_rank_spread_table(results: list[RankSpreadResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r.model, "spread": r.spread}
        for col in BENCHMARK_COLUMNS:
            row[BENCHMARK_LABELS[col]] = r.ranks.get(col, "")
        rows.append(row)
    return pd.DataFrame(rows)


def run_all_analyses(scores: pd.DataFrame | None = None) -> dict:
    if scores is None:
        scores = load_scores()
    corr_results = pairwise_spearman(scores)
    inv_results = ranking_inversions(scores)
    spread_results = rank_spreads(scores)
    return {
        "scores": scores,
        "correlations": corr_results,
        "correlation_matrix": correlation_matrix(scores),
        "correlation_table": format_correlation_table(corr_results),
        "convergent_discriminant": convergent_discriminant_analysis(scores),
        "inversions": inv_results,
        "inversion_table": format_inversion_table(inv_results),
        "ranks": compute_ranks(scores),
        "rank_spreads": spread_results,
        "rank_spread_table": format_rank_spread_table(spread_results),
    }
