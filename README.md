# Construct Validity

Analysis code for auditing construct validity across agentic AI benchmarks.

Repository: https://github.com/H-Sompalle/construct-validity

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_analysis.py
python verify.py
```

`verify.py` runs automated checks against expected values in `src/expectations.py` (scores, ranks, correlations, p-values, and related claims).

Outputs are written to `outputs/`:

| Output | Description |
|--------|-------------|
| `tables/correlation_matrix.csv` | Pairwise Spearman correlations |
| `tables/inversion_rates.csv` | Ranking inversion rates |
| `tables/rank_spreads.csv` | Per-model rank spreads |
| `figures/figure1_correlation_heatmap.png` | Correlation heatmap |
| `figures/figure2_score_relationships.png` | Score relationship scatters |
| `figures/figure3_rank_profiles.png` | Rank profiles across benchmarks |
| `summary.txt` | Headline statistics |

## Data

The cross-benchmark score matrix is in `data/scores.csv`. Scores were collected from public leaderboards and model cards (May–June 2026). Missing entries are left blank and excluded from pairwise analyses.

## Analyses

1. **Spearman rank correlations** — pairwise-complete observations per benchmark pair
2. **Convergent vs. discriminant validity** — mean within-agent vs. cross-domain correlations
3. **Ranking inversions** — fraction of model pairs whose relative order swaps
4. **Rank spread** — max rank − min rank for models with scores on ≥4 benchmarks

## Updating

To replicate on newer leaderboard snapshots, edit `data/scores.csv` and re-run `python run_analysis.py`. The pairwise-complete methodology automatically handles new models and missing scores.
