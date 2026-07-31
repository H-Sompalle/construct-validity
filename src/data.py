"""Load and label the cross-benchmark score matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BENCHMARK_COLUMNS = ["tau_retail", "tau_airline", "swe_bench", "gpqa", "mmlu_pro"]

BENCHMARK_LABELS = {
    "tau_retail": "τ-Retail",
    "tau_airline": "τ-Airline",
    "swe_bench": "SWE-bench",
    "gpqa": "GPQA",
    "mmlu_pro": "MMLU-Pro",
}

AGENT_BENCHMARKS = {"tau_retail", "tau_airline", "swe_bench"}
REASONING_BENCHMARKS = {"gpqa", "mmlu_pro"}


def load_scores(path: Path | str | None = None) -> pd.DataFrame:
    """Return model × benchmark score matrix with NaN for missing scores."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "scores.csv"
    df = pd.read_csv(path)
    df = df.set_index("model")
    for col in BENCHMARK_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[BENCHMARK_COLUMNS]
