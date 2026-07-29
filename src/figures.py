"""Generate Figures 1–3 from the construct validity audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .data import BENCHMARK_COLUMNS, BENCHMARK_LABELS, load_scores
from .analyze import compute_ranks, pairwise_spearman

# Short labels for scatter-plot annotations.
SHORT_LABELS = {
    "Claude 3.5 Sonnet": "C.3.5 Sonnet",
    "Claude 3.7 Sonnet": "C.3.7 Sonnet",
    "Claude Sonnet 4": "C.Sonnet 4",
    "Claude Opus 4": "C.Opus 4",
    "Claude Opus 4.1": "C.Opus 4.1",
    "Claude Sonnet 4.5": "C.Sonnet 4.5",
    "GPT-4o": "G4o",
    "GPT-4.5": "G4.5",
    "GPT-4.1": "G4.1",
    "GPT-4.1 mini": "G4.1 mini",
    "GPT-4.1 nano": "G4.1 nano",
    "o1": "o1",
    "o3-mini": "o3 mini",
    "o4-mini": "o4 mini",
    "Claude 3.5 Haiku": "C.3.5 Haiku",
}


def _rho_for_pair(scores: pd.DataFrame, a: str, b: str) -> float:
    pair = scores[[a, b]].dropna()
    rho, _ = stats.spearmanr(pair[a], pair[b])
    return float(rho)


def figure1_correlation_heatmap(
    scores: pd.DataFrame,
    output: Path,
    correlations: list | None = None,
) -> None:
    """Figure 1: cross-benchmark Spearman correlation heatmap with bootstrap CIs."""
    from .analyze import pairwise_spearman

    labels = [BENCHMARK_LABELS[c] for c in BENCHMARK_COLUMNS]
    mat = np.eye(len(BENCHMARK_COLUMNS))
    ci = {(i, i): None for i in range(len(BENCHMARK_COLUMNS))}
    if correlations is None:
        correlations = pairwise_spearman(scores)
    corr_by_pair = {(r.benchmark_a, r.benchmark_b): r for r in correlations}
    for i, a in enumerate(BENCHMARK_COLUMNS):
        for j, b in enumerate(BENCHMARK_COLUMNS):
            if i < j:
                r = corr_by_pair[(a, b)]
                mat[i, j] = r.rho
                mat[j, i] = r.rho
                ci[(i, j)] = (r.ci_low, r.ci_high)
                ci[(j, i)] = (r.ci_low, r.ci_high)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
            else:
                lo, hi = ci[(i, j)]
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}\n[{lo:.2f}, {hi:.2f}]",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=7,
                    linespacing=1.15,
                )
    ax.set_title("Cross-Benchmark Rank Correlations\n(Spearman ρ with bootstrap 95% CIs)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _scatter_with_labels(
    ax: plt.Axes,
    scores: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
) -> None:
    pair = scores[[x_col, y_col]].dropna()
    rho = _rho_for_pair(scores, x_col, y_col)
    ax.scatter(pair[x_col], pair[y_col], s=55, color="#1f77b4", edgecolors="white", linewidths=0.5)
    for model, row in pair.iterrows():
        ax.annotate(
            SHORT_LABELS.get(model, model),
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xlabel(f"{BENCHMARK_LABELS[x_col]} Score")
    ax.set_ylabel(f"{BENCHMARK_LABELS[y_col]} Score")
    ax.set_title(f"{title} (ρ={rho:.2f})")
    ax.grid(True, alpha=0.25)


def figure2_score_relationships(scores: pd.DataFrame, output: Path) -> None:
    """Figure 2: τ-Retail vs SWE-bench and τ-Retail vs GPQA scatter plots."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _scatter_with_labels(
        axes[0],
        scores,
        "tau_retail",
        "swe_bench",
        "Tool-Use vs Code Agent",
    )
    _scatter_with_labels(
        axes[1],
        scores,
        "tau_retail",
        "gpqa",
        "Tool-Use vs Reasoning",
    )
    fig.suptitle("Cross-Benchmark Score Relationships", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure3_rank_profiles(scores: pd.DataFrame, output: Path) -> None:
    """Figure 3: model rank profiles across benchmarks."""
    ranks = compute_ranks(scores)
    labels = [BENCHMARK_LABELS[c] for c in BENCHMARK_COLUMNS]
    x = np.arange(len(BENCHMARK_COLUMNS))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in ranks.index:
        model_ranks = ranks.loc[model].dropna()
        if model_ranks.empty:
            continue
        xs = [BENCHMARK_COLUMNS.index(c) for c in model_ranks.index]
        ys = [model_ranks[c] for c in model_ranks.index]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Model Rankings Across Benchmarks")
    ax.invert_yaxis()
    ax.set_ylim(14, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    output_dir: Path | str | None = None,
    scores: pd.DataFrame | None = None,
    correlations: list | None = None,
) -> dict[str, Path]:
    if scores is None:
        scores = load_scores()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "figure1": output_dir / "figure1_correlation_heatmap.png",
        "figure1_pdf": output_dir / "figure1_correlation_heatmap.pdf",
        "figure2": output_dir / "figure2_score_relationships.png",
        "figure3": output_dir / "figure3_rank_profiles.png",
    }
    figure1_correlation_heatmap(scores, paths["figure1"], correlations=correlations)
    figure2_score_relationships(scores, paths["figure2"])
    figure3_rank_profiles(scores, paths["figure3"])
    return paths
