"""Expected values for automated verification of analysis outputs."""

from __future__ import annotations

# Cross-benchmark scores. None = no published score.
EXPECTED_SCORES: dict[str, dict[str, float | None]] = {
    "Claude 3.5 Sonnet": {
        "tau_retail": 69.2,
        "tau_airline": 46.0,
        "swe_bench": 49.0,
        "gpqa": 65.0,
        "mmlu_pro": 78.0,
    },
    "Claude 3.7 Sonnet": {
        "tau_retail": 81.2,
        "tau_airline": 58.4,
        "swe_bench": 62.3,
        "gpqa": 68.0,
        "mmlu_pro": 78.2,
    },
    "Claude Sonnet 4": {
        "tau_retail": 80.5,
        "tau_airline": 60.0,
        "swe_bench": 72.7,
        "gpqa": 75.4,
        "mmlu_pro": None,
    },
    "Claude Opus 4": {
        "tau_retail": 81.4,
        "tau_airline": 59.6,
        "swe_bench": 72.5,
        "gpqa": 79.6,
        "mmlu_pro": None,
    },
    "Claude Opus 4.1": {
        "tau_retail": 82.4,
        "tau_airline": 56.0,
        "swe_bench": None,
        "gpqa": None,
        "mmlu_pro": None,
    },
    "Claude Sonnet 4.5": {
        "tau_retail": 86.2,
        "tau_airline": 70.0,
        "swe_bench": 77.2,
        "gpqa": 83.4,
        "mmlu_pro": None,
    },
    "GPT-4o": {
        "tau_retail": 60.3,
        "tau_airline": None,
        "swe_bench": 33.2,
        "gpqa": 53.6,
        "mmlu_pro": 72.6,
    },
    "GPT-4.5": {
        "tau_retail": 68.4,
        "tau_airline": 50.0,
        "swe_bench": 38.0,
        "gpqa": 71.4,
        "mmlu_pro": 73.6,
    },
    "GPT-4.1": {
        "tau_retail": 68.0,
        "tau_airline": 49.4,
        "swe_bench": 54.6,
        "gpqa": 56.5,
        "mmlu_pro": 73.6,
    },
    "GPT-4.1 mini": {
        "tau_retail": 55.8,
        "tau_airline": None,
        "swe_bench": 28.9,
        "gpqa": 43.7,
        "mmlu_pro": 64.2,
    },
    "GPT-4.1 nano": {
        "tau_retail": 22.6,
        "tau_airline": None,
        "swe_bench": None,
        "gpqa": None,
        "mmlu_pro": 55.6,
    },
    "o1": {
        "tau_retail": 70.8,
        "tau_airline": 50.0,
        "swe_bench": 48.9,
        "gpqa": 78.0,
        "mmlu_pro": 83.6,
    },
    "o3-mini": {
        "tau_retail": 57.6,
        "tau_airline": None,
        "swe_bench": 49.3,
        "gpqa": 77.0,
        "mmlu_pro": 80.0,
    },
    "o4-mini": {
        "tau_retail": 71.8,
        "tau_airline": 49.2,
        "swe_bench": 68.1,
        "gpqa": 81.4,
        "mmlu_pro": 81.4,
    },
    "Claude 3.5 Haiku": {
        "tau_retail": 51.0,
        "tau_airline": None,
        "swe_bench": 40.6,
        "gpqa": 41.0,
        "mmlu_pro": 65.0,
    },
}

# Raw ranks (1 = best).
EXPECTED_RANKS: dict[str, dict[str, int | None]] = {
    "o3-mini": {
        "tau_retail": 12,
        "tau_airline": None,
        "swe_bench": 7,
        "gpqa": 5,
        "mmlu_pro": 3,
        "spread": 9,
    },
    "o1": {
        "tau_retail": 7,
        "tau_airline": 7,
        "swe_bench": 9,
        "gpqa": 4,
        "mmlu_pro": 1,
        "spread": 8,
    },
    "o4-mini": {
        "tau_retail": 6,
        "tau_airline": 9,
        "swe_bench": 4,
        "gpqa": 2,
        "mmlu_pro": 2,
        "spread": 7,
    },
    "GPT-4.5": {
        "tau_retail": 9,
        "tau_airline": 6,
        "swe_bench": 11,
        "gpqa": 7,
        "mmlu_pro": 6,
        "spread": 5,
    },
    "Claude Opus 4": {
        "tau_retail": 3,
        "tau_airline": 3,
        "swe_bench": 3,
        "gpqa": 3,
        "mmlu_pro": None,
        "spread": 0,
    },
    "Claude Sonnet 4.5": {
        "tau_retail": 1,
        "tau_airline": 1,
        "swe_bench": 1,
        "gpqa": 1,
        "mmlu_pro": None,
        "spread": 0,
    },
}

# Percentile-normalized ranks: 0 = best, 100 = worst; spread = max − min.
# Values are round(100 * (rank − 1) / (n − 1)).
EXPECTED_PERCENTILES: dict[str, dict[str, int | None]] = {
    "o4-mini": {
        "tau_retail": 36,
        "tau_airline": 89,
        "swe_bench": 25,
        "gpqa": 8,
        "mmlu_pro": 10,
        "percentile_spread": 81,
    },
    "o1": {
        "tau_retail": 43,
        "tau_airline": 67,
        "swe_bench": 67,
        "gpqa": 25,
        "mmlu_pro": 0,
        "percentile_spread": 67,
    },
    "o3-mini": {
        "tau_retail": 79,
        "tau_airline": None,
        "swe_bench": 50,
        "gpqa": 33,
        "mmlu_pro": 20,
        "percentile_spread": 59,
    },
    "GPT-4.5": {
        "tau_retail": 57,
        "tau_airline": 56,
        "swe_bench": 83,
        "gpqa": 50,
        "mmlu_pro": 50,
        "percentile_spread": 33,
    },
    "Claude Opus 4": {
        "tau_retail": 14,
        "tau_airline": 22,
        "swe_bench": 17,
        "gpqa": 17,
        "mmlu_pro": None,
        "percentile_spread": 8,
    },
    "Claude Sonnet 4.5": {
        "tau_retail": 0,
        "tau_airline": 0,
        "swe_bench": 0,
        "gpqa": 0,
        "mmlu_pro": None,
        "percentile_spread": 0,
    },
}

# Spearman ρ (rounded to 2 decimals), n, p-values, bootstrap CIs.
# Optional p_* / ci_* keys cover claims stated explicitly in the analysis summary.
CORRELATIONS: dict[tuple[str, str], dict] = {
    ("tau_retail", "tau_airline"): {"rho": 0.75, "n": 10, "ci": (0.23, 0.95)},
    ("tau_retail", "swe_bench"): {"rho": 0.81, "n": 13, "ci": (0.44, 0.95)},
    ("tau_retail", "gpqa"): {"rho": 0.76, "n": 13, "ci": (0.27, 0.98)},
    ("tau_retail", "mmlu_pro"): {"rho": 0.80, "n": 11, "ci": (0.29, 0.99)},
    ("tau_airline", "swe_bench"): {"rho": 0.66, "n": 9, "ci": (-0.19, 1.00)},
    ("tau_airline", "gpqa"): {"rho": 0.49, "n": 9, "p_gt_005": True, "ci": (-0.41, 0.95)},
    ("tau_airline", "mmlu_pro"): {
        "rho": 0.10,
        "n": 6,
        "p_round_2": 0.85,
        "ci": (-0.87, 0.95),
    },
    ("swe_bench", "gpqa"): {"rho": 0.72, "n": 13, "ci": (0.26, 0.93)},
    ("swe_bench", "mmlu_pro"): {"rho": 0.69, "n": 10, "ci": (0.07, 0.98)},
    ("gpqa", "mmlu_pro"): {"rho": 0.92, "n": 10, "p_lt_001": True, "ci": (0.65, 1.00)},
}

SUMMARY_STATS = {
    "mean_rho_all": 0.67,
    "mean_rho_within_agent": 0.74,
    "mean_rho_cross_domain": 0.59,
    "rho_min": 0.10,
    "rho_max": 0.92,
    "mean_inversion_rate_pct": 22.1,
    "abstract_inversion_rate_pct_rounded": 22,
}

# Ranking inversion counts and rates.
INVERSIONS: dict[tuple[str, str], tuple[int, int, float]] = {
    ("tau_retail", "tau_airline"): (10, 45, 22.2),
    ("tau_retail", "swe_bench"): (14, 78, 17.9),
    ("tau_retail", "gpqa"): (14, 78, 17.9),
    ("tau_retail", "mmlu_pro"): (9, 55, 16.4),
    ("tau_airline", "swe_bench"): (8, 36, 22.2),
    ("tau_airline", "gpqa"): (11, 36, 30.6),
    ("tau_airline", "mmlu_pro"): (6, 15, 40.0),
    ("swe_bench", "gpqa"): (18, 78, 23.1),
    ("swe_bench", "mmlu_pro"): (10, 45, 22.2),
    ("gpqa", "mmlu_pro"): (4, 45, 8.9),
}

COVERAGE: dict[str, tuple[int, int, int]] = {
    "tau_retail": (15, 15, 100),
    "tau_airline": (10, 15, 67),
}

# Rank/score claims used in the analysis narrative.
PROSE_CLAIMS = {
    "o4-mini_gpqa_rank": 2,
    "o4-mini_mmlu_rank": 2,
    "o4-mini_tau_retail_rank": 6,
    "o4-mini_tau_airline_rank": 9,
    "o1_mmlu_rank": 1,
    "o1_gpqa_rank": 4,
    "o1_tau_retail_rank": 7,
    "o1_tau_airline_rank": 7,
    "o1_swe_rank": 9,
    "o3-mini_mmlu_rank": 3,
    "o3-mini_gpqa_rank": 5,
    "o3-mini_tau_retail_rank": 12,
    "claude_sonnet_45_all_ranks": 1,
    "claude_opus_41_tau_retail_rank": 2,
    "claude_opus_41_tau_airline_rank": 5,
}

DERIVED = {
    "tau_retail_tau_airline_unexplained_variance": 0.44,  # 1 - 0.75^2
}
