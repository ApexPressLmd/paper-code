"""Statistical procedures (Sections 4.3, 4.5, 4.7, 4.8; Appendix C).

* Question-level paired comparison (Table 5): per-question scores are first averaged over the five seeds;
  the effect is the mean difference in metric points; 95% interval by paired bootstrap with the
  question as resampling unit (10,000 resamples, percentile); two-sided P by paired sign-flip
  permutation (10,000 permutations); Holm adjustment across the pre-specified comparisons.
* Friedman test over methods x datasets, Nemenyi critical difference CD = q_alpha sqrt(k(k+1)/(6N)).
* Wilson score intervals, Cohen's kappa, exact McNemar test.
* Calibration: ECE with equal-frequency bins (10 bins), Brier score, negative log-likelihood, AUROC.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int, rng: np.random.Generator, level: float = 0.95,
                        chunk: int = 1000) -> tuple[float, float]:
    diff = np.asarray(diff, float)
    n = len(diff)
    means = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        means[s:s + b] = diff[rng.integers(0, n, size=(b, n))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def sign_flip_pvalue(diff: np.ndarray, n_perm: int, rng: np.random.Generator, chunk: int = 1000) -> float:
    diff = np.asarray(diff, float)
    obs = abs(diff.mean())
    count = 0
    for s in range(0, n_perm, chunk):
        b = min(chunk, n_perm - s)
        signs = rng.choice((-1.0, 1.0), size=(b, len(diff)))
        count += int((np.abs((signs * diff).mean(axis=1)) >= obs - 1e-12).sum())
    return (count + 1) / (n_perm + 1)


def holm(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_comparison(a: np.ndarray, b: np.ndarray, n_boot: int, n_perm: int, seed: int, level: float = 0.95) -> dict:
    """a, b: per-question scores in [0, 1] (already averaged over seeds). Effects in metric points."""
    rng = np.random.default_rng(seed)
    diff = (np.asarray(a, float) - np.asarray(b, float)) * 100.0
    lo, hi = paired_bootstrap_ci(diff, n_boot, rng, level)
    p = sign_flip_pvalue(diff, n_perm, rng)
    return {"n": int(len(diff)), "delta": float(diff.mean()), "ci_low": lo, "ci_high": hi, "p_raw": p}


def friedman_nemenyi(scores: np.ndarray, alpha: float = 0.05) -> dict:
    """scores: (N datasets, k methods), higher is better. Ranks: 1 = best, ties averaged."""
    scores = np.asarray(scores, float)
    N, k = scores.shape
    ranks = np.vstack([stats.rankdata(-row) for row in scores])
    stat, p = stats.friedmanchisquare(*scores.T)
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    cd = q * math.sqrt(k * (k + 1) / (6.0 * N))
    return {"mean_ranks": ranks.mean(axis=0), "chi2": float(stat), "p": float(p), "q_alpha": float(q), "cd": float(cd)}


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - level) / 2)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a, b = list(a), list(b)
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant counts b and c."""
    return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


# ---------------------------------------------------------------------------- calibration metrics
def ece_equal_frequency(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    order = np.argsort(p, kind="mergesort")
    total = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            total += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(total)


def ece_equal_width(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    order = np.argsort(np.asarray(p, float), kind="mergesort")
    out = []
    for idx in np.array_split(order, n_bins):
        if len(idx):
            out.append({"n": int(len(idx)), "mean_score": float(np.mean(p[idx])), "accuracy": float(np.mean(y[idx]))})
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
