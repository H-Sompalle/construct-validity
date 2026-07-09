# Construct Validity Audit — Reproduction Code

Reproduces the empirical analyses from **Construct Validity Failures in Agentic AI Benchmarks: An Empirical Audit** (KDD '26 Workshop).

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

`verify.py` runs **197 automated checks** against values transcribed independently in `src/paper_expectations.py` (Table 3 scores, Table 4–5, correlations, p-values, prose claims).

Outputs are written to `outputs/`:

| Output | Paper reference |
|--------|-----------------|
| `tables/correlation_matrix.csv` | Figure 1, §4.1 |
| `tables/inversion_rates.csv` | Table 4, §4.2 |
| `tables/rank_spreads.csv` | Table 5, §4.4 |
| `figures/figure1_correlation_heatmap.png` | Figure 1 |
| `figures/figure2_score_relationships.png` | Figure 2 |
| `figures/figure3_rank_profiles.png` | Figure 3 |
| `summary.txt` | Headline statistics |

## Data

The cross-benchmark score matrix (`data/scores.csv`) matches Table 3 in the paper. Scores were collected from public leaderboards and model cards (May–June 2026). Missing entries (`—` in the paper) are left blank and excluded from pairwise analyses.

## Analyses

1. **Spearman rank correlations** — pairwise-complete observations per benchmark pair (§3.3, Analysis 1)
2. **Convergent vs. discriminant validity** — mean within-agent vs. cross-domain correlations (§3.3, Analysis 2)
3. **Ranking inversions** — fraction of model pairs whose relative order swaps (§3.3, Analysis 3)
4. **Rank spread** — max rank − min rank for models with scores on ≥4 benchmarks (§3.3, Analysis 4)

## Updating

To replicate on newer leaderboard snapshots, edit `data/scores.csv` and re-run `python run_analysis.py`. The pairwise-complete methodology automatically handles new models and missing scores.
