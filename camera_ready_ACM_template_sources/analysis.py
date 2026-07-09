import numpy as np
from scipy.stats import spearmanr
from itertools import combinations

models = [
    "Claude 3.5 Sonnet", "Claude 3.7 Sonnet", "Claude Sonnet 4", "Claude Opus 4",
    "Claude Opus 4.1", "Claude Sonnet 4.5", "GPT-4o", "GPT-4.5", "GPT-4.1",
    "GPT-4.1 mini", "GPT-4.1 nano", "o1", "o3-mini", "o4-mini", "Claude 3.5 Haiku"
]

# CORRECTED DATA (verified July 2026 against primary sources)
data = {
    "tau_retail":  [69.2, 81.2, 80.5, 81.4, 82.4, 86.2, 60.3, 68.4, 68.0, 55.8, 22.6, 70.8, 57.6, 71.8, 51.0],
    "tau_airline": [46.0, 58.4, 60.0, 59.6, 56.0, 70.0, None, 50.0, 49.4, None, None, 50.0, None, 49.2, None],
    "swe_bench":   [49.0, 62.3, 72.7, 72.5, None, 77.2, 33.2, 38.0, 54.6, 28.9, None, 48.9, 49.3, 68.1, 40.6],
    # GPQA corrections: Sonnet4 75.4, Opus4 79.6, Sonnet4.5 83.4 (added), o3-mini 77.0, o4-mini 81.4
    "gpqa":        [65.0, 68.0, 75.4, 79.6, None, 83.4, 53.6, 71.4, 56.5, 43.7, None, 78.0, 77.0, 81.4, 41.0],
    "mmlu_pro":    [78.0, 78.2, None, None, None, None, 72.6, 73.6, 73.6, 64.2, 55.6, 83.6, 80.0, 81.4, 65.0],
}
bench_names = ["tau_retail", "tau_airline", "swe_bench", "gpqa", "mmlu_pro"]
labels = {"tau_retail": "τ-Retail", "tau_airline": "τ-Airline", "swe_bench": "SWE-bench",
          "gpqa": "GPQA", "mmlu_pro": "MMLU-Pro"}

print("=" * 78)
print("PAIRWISE SPEARMAN CORRELATIONS (corrected data)")
print("=" * 78)
corr = {}
for a, b in combinations(bench_names, 2):
    pairs = [(x, y) for x, y in zip(data[a], data[b]) if x is not None and y is not None]
    xs, ys = zip(*pairs)
    rho, p = spearmanr(xs, ys)
    corr[(a, b)] = (rho, p, len(pairs))
    sig = "sig" if p < 0.05 else "ns "
    bonf = "B" if p < 0.005 else " "
    print(f"{labels[a]:11s} x {labels[b]:11s}  rho={rho:6.2f}  p={p:.4f} {sig}{bonf}  n={len(pairs)}")

rhos = [v[0] for v in corr.values()]
print(f"\nMean rho = {np.mean(rhos):.3f}  Range = {min(rhos):.2f} to {max(rhos):.2f}")

agent = ["tau_retail", "tau_airline", "swe_bench"]
reason = ["gpqa", "mmlu_pro"]
within_agent = [corr[k][0] for k in corr if k[0] in agent and k[1] in agent]
within_reason = [corr[k][0] for k in corr if k[0] in reason and k[1] in reason]
cross = [corr[k][0] for k in corr if (k[0] in agent) != (k[1] in agent)]
print(f"Within-agent mean rho   = {np.mean(within_agent):.3f}  ({[round(r,2) for r in within_agent]})")
print(f"Within-reasoning rho    = {np.mean(within_reason):.3f}")
print(f"Cross-domain mean rho   = {np.mean(cross):.3f}  ({[round(r,2) for r in cross]})")

print("\n" + "=" * 78)
print("RANKING INVERSIONS")
print("=" * 78)
def inversions(a, b):
    pairs_models = [i for i in range(len(models)) if data[a][i] is not None and data[b][i] is not None]
    inv, tot = 0, 0
    for i, j in combinations(pairs_models, 2):
        da = data[a][i] - data[a][j]
        db = data[b][i] - data[b][j]
        if da == 0 or db == 0:
            tot += 1
            continue
        tot += 1
        if (da > 0) != (db > 0):
            inv += 1
    return inv, tot

inv_rates = []
for a, b in combinations(bench_names, 2):
    inv, tot = inversions(a, b)
    rate = 100 * inv / tot
    inv_rates.append(rate)
    print(f"{labels[a]:11s} x {labels[b]:11s}  {inv}/{tot}  ({rate:.1f}%)")
print(f"\nMean inversion rate = {np.mean(inv_rates):.1f}%")

print("\n" + "=" * 78)
print("PER-BENCHMARK RANKINGS")
print("=" * 78)
ranks = {}
for b in bench_names:
    scored = [(models[i], data[b][i]) for i in range(len(models)) if data[b][i] is not None]
    scored.sort(key=lambda x: -x[1])
    ranks[b] = {m: r + 1 for r, (m, s) in enumerate(scored)}
    print(f"\n{labels[b]} (n={len(scored)}):")
    for r, (m, s) in enumerate(scored, 1):
        print(f"  {r:2d}. {m:22s} {s:.1f}")

print("\n" + "=" * 78)
print("RANK SPREADS (models on 4+ benchmarks)")
print("=" * 78)
for m in models:
    mranks = {labels[b]: ranks[b][m] for b in bench_names if m in ranks[b]}
    if len(mranks) >= 4:
        spread = max(mranks.values()) - min(mranks.values())
        print(f"{m:22s} spread={spread:2d}  {mranks}")

print("\n" + "=" * 78)
print("KEY CLAIM CHECKS")
print("=" * 78)
for m in ["o4-mini", "o1", "o3-mini", "Claude Sonnet 4.5", "Claude 3.7 Sonnet", "Claude Opus 4.1", "GPT-4.5", "GPT-4.1"]:
    mranks = {labels[b]: ranks[b].get(m) for b in bench_names if m in ranks[b]}
    print(f"{m:22s} {mranks}")
