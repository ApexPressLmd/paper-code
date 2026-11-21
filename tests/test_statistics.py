"""Statistical procedures checked against counts stated in the manuscript (arithmetic identities only)."""
import numpy as np
import pytest

from peak_rag.eval import stats as S
from peak_rag.eval.selective import abstention_metrics, risk_coverage


def test_wilson_intervals_reported_in_paper():
    assert [round(100 * x, 1) for x in S.wilson(216, 250)] == [81.6, 90.1]      # human evaluation
    assert [round(100 * x, 1) for x in S.wilson(442, 500)] == [85.3, 90.9]      # audited claims
    assert [round(100 * x, 1) for x in S.wilson(89, 100)] == [81.4, 93.7]       # typed edges
    assert [round(100 * x, 1) for x in S.wilson(64, 100)] == [54.2, 72.7]       # co-occurrence edges
    assert [round(100 * x, 1) for x in S.wilson(41, 100)] == [31.9, 50.8]       # first-mention edges


def test_nemenyi_critical_difference_k16_n5():
    rng = np.random.default_rng(0)
    res = S.friedman_nemenyi(rng.random((5, 16)))
    assert round(res["cd"], 2) == 10.32                                          # Figure 6
    assert round(1.20 + res["cd"], 2) == 11.52


def test_exact_mcnemar_from_discordant_errors():
    assert round(S.mcnemar_exact(14, 10), 4) == 0.5413                           # Supplementary Table S18


def test_abstention_chain_at_default_operating_point():
    # 4,500 answerable test questions with 187 false refusals; 500 unanswerable with 413 abstentions
    unans = np.r_[np.zeros(4500, bool), np.ones(500, bool)]
    abst = np.r_[np.arange(4500) < 187, np.arange(500) < 413]
    m = abstention_metrics(abst, unans)
    assert m["abstentions"] == 600
    assert round(100 * m["coverage"], 1) == 88.0
    assert round(100 * m["false_refusal_rate"], 2) == 4.16
    assert round(m["f1"], 4) == 0.7509
    assert round(100 * m["unanswerable_answered_rate"], 1) == 17.4


def test_holm_and_resampling():
    adj = S.holm([0.01, 0.04, 0.03, 0.2, 0.5])
    assert np.allclose(adj, [0.05, 0.12, 0.12, 0.4, 0.5])
    rng = np.random.default_rng(1)
    diff = rng.normal(0, 1, 900)
    lo, hi = S.paired_bootstrap_ci(diff, 2000, np.random.default_rng(2))
    assert lo < diff.mean() < hi
    p = S.sign_flip_pvalue(diff + 5, 999, np.random.default_rng(3))
    assert p == pytest.approx(1 / 1000)


def test_calibration_metrics():
    p = np.linspace(0.05, 0.95, 1000)
    y = (np.random.default_rng(4).random(1000) < p).astype(float)
    assert S.ece_equal_frequency(p, y, 10) < 0.06
    assert S.brier(p, y) < 0.25 and S.nll(p, y) < 0.7 and S.auroc(p, y) > 0.7
    cov, risk, aurc = risk_coverage(p, y)
    assert cov[-1] == 1.0 and 0 <= aurc <= 1
