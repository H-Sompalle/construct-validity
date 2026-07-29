"""Construct validity analyses from the KDD workshop paper."""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

from .data import AGENT_BENCHMARKS, BENCHMARK_COLUMNS, BENCHMARK_LABELS, load_scores

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


class CorrelationResult(NamedTuple):
    benchmark_a: str
    benchmark_b: str
    rho: float
    p_value: float
    n: int
    ci_low: float
    ci_high: float


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
    percentile_ranks: dict[str, float]
    percentile_spread: float


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman ρ (pairwise-complete vectors)."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        # Constant resamples yield undefined Spearman ρ; skip them.
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            boots[i] = np.nan
            continue
        rho, _ = stats.spearmanr(xb, yb)
        boots[i] = rho if np.isfinite(rho) else np.nan
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def pairwise_spearman(
    scores: pd.DataFrame,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[CorrelationResult]:
    """Spearman rank correlations on pairwise-complete observations with bootstrap CIs."""
    results: list[CorrelationResult] = []
    for a, b in combinations(BENCHMARK_COLUMNS, 2):
        pair = scores[[a, b]].dropna()
        n = len(pair)
        if n < 2:
            rho, p, lo, hi = np.nan, np.nan, np.nan, np.nan
        else:
            x = pair[a].to_numpy()
            y = pair[b].to_numpy()
            rho, p = stats.spearmanr(x, y)
            lo, hi = _bootstrap_spearman_ci(x, y, n_resamples=n_resamples, seed=seed)
        results.append(
            CorrelationResult(a, b, float(rho), float(p), n, float(lo), float(hi))
        )
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


def compute_percentile_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks in [0, 1]: 0 = best, 1 = worst, via (rank − 1) / (n − 1).

    Normalizes for unequal field sizes (e.g. τ-Airline n=10 vs τ-Retail n=15).
    """
    ranks = compute_ranks(scores)
    pct = ranks.copy()
    for col in BENCHMARK_COLUMNS:
        n = int(scores[col].notna().sum())
        if n <= 1:
            pct[col] = np.nan
        else:
            pct[col] = (ranks[col] - 1) / (n - 1)
    return pct


def rank_spreads(scores: pd.DataFrame, min_benchmarks: int = 4) -> list[RankSpreadResult]:
    """Rank and percentile-rank spreads for models scored on ≥ min_benchmarks."""
    ranks = compute_ranks(scores)
    pct = compute_percentile_ranks(scores)
    results: list[RankSpreadResult] = []
    for model in scores.index:
        model_ranks = ranks.loc[model].dropna()
        if len(model_ranks) < min_benchmarks:
            continue
        model_pct = pct.loc[model, model_ranks.index]
        rank_dict = {col: int(model_ranks[col]) for col in model_ranks.index}
        pct_dict = {col: float(model_pct[col]) for col in model_ranks.index}
        spread = int(model_ranks.max() - model_ranks.min())
        pct_spread = float(model_pct.max() - model_pct.min())
        results.append(
            RankSpreadResult(model, rank_dict, spread, pct_dict, pct_spread)
        )
    results.sort(key=lambda r: r.percentile_spread, reverse=True)
    return results


def format_correlation_table(results: list[CorrelationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "benchmark_a": BENCHMARK_LABELS[r.benchmark_a],
                "benchmark_b": BENCHMARK_LABELS[r.benchmark_b],
                "rho": round(r.rho, 2),
                "ci_low": round(r.ci_low, 2),
                "ci_high": round(r.ci_high, 2),
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
    """Raw ranks and raw spread (legacy Table 5 layout)."""
    rows = []
    for r in results:
        row = {"model": r.model, "spread": r.spread}
        for col in BENCHMARK_COLUMNS:
            row[BENCHMARK_LABELS[col]] = r.ranks.get(col, "")
        rows.append(row)
    return pd.DataFrame(rows)


def format_percentile_spread_table(results: list[RankSpreadResult]) -> pd.DataFrame:
    """Percentile ranks (0--100, 0 = best) and percentile spread."""
    rows = []
    for r in results:
        row = {
            "model": r.model,
            "percentile_spread": round(100 * r.percentile_spread),
            "raw_spread": r.spread,
        }
        for col in BENCHMARK_COLUMNS:
            if col in r.percentile_ranks:
                row[BENCHMARK_LABELS[col]] = round(100 * r.percentile_ranks[col])
            else:
                row[BENCHMARK_LABELS[col]] = ""
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
        "percentile_ranks": compute_percentile_ranks(scores),
        "rank_spreads": spread_results,
        "rank_spread_table": format_rank_spread_table(spread_results),
        "percentile_spread_table": format_percentile_spread_table(spread_results),
    }
