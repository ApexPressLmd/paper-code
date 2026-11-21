"""Gate calibration (Section 3.6).

Target    y = 1 when the candidate answer is correct under the primary metric of its dataset
          (normalised exact correctness for entity and exact-answer items, exact label correctness for
          yes/no items, token-level F1 >= 0.80 for token-overlap metrics), y = 0 otherwise; on the test
          split any answer to an unanswerable question has y = 0. The development split has no
          unanswerable questions.
Features  (v_branch, 1 - c_conf, u_faith).
Fit       non-negativity-constrained logistic regression minimising the mean binary cross-entropy plus
          (eta / 2) ||w||^2 with eta = 0.01 on w1..w3 (not on b), subject to w1, w2, w3 >= 0.
Thresholds theta_low  = 10% quantile of the development scores (10% abstention budget);
           theta_high = smallest development score at which the accepted answers (s >= theta_high)
                        reach 85% precision.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize

FEATURES = ("v_branch", "one_minus_c_conf", "u_faith")


def features_from_records(records: Sequence[dict]) -> np.ndarray:
    return np.array([[r["v_branch"], 1.0 - r["c_conf"], r["u_faith"]] for r in records], dtype=float)


def fit_gate(X: np.ndarray, y: np.ndarray, eta: float = 0.01, mask: Sequence[bool] = (True, True, True)) -> dict:
    """Returns {"w1", "w2", "w3", "b"}; masked features get weight 0 (Supplementary Table S17)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    m = np.array(mask, bool)
    Xm = X[:, m]

    def loss(theta):
        w, b = theta[:-1], theta[-1]
        z = Xm @ w + b
        # mean BCE written with logaddexp for stability
        bce = np.mean(np.logaddexp(0, z) - y * z)
        return bce + 0.5 * eta * np.dot(w, w)

    def grad(theta):
        w, b = theta[:-1], theta[-1]
        p = 1.0 / (1.0 + np.exp(-(Xm @ w + b)))
        gw = Xm.T @ (p - y) / len(y) + eta * w
        gb = np.mean(p - y)
        return np.append(gw, gb)

    k = int(m.sum())
    res = minimize(loss, np.append(np.full(k, 0.5), 0.0), jac=grad, method="L-BFGS-B",
                   bounds=[(0.0, None)] * k + [(None, None)])
    w = np.zeros(3)
    w[m] = res.x[:-1]
    return {"w1": float(w[0]), "w2": float(w[1]), "w3": float(w[2]), "b": float(res.x[-1]),
            "converged": bool(res.success), "loss": float(res.fun)}


def score(X: np.ndarray, weights: dict) -> np.ndarray:
    z = X @ np.array([weights["w1"], weights["w2"], weights["w3"]]) + weights["b"]
    return 1.0 / (1.0 + np.exp(-z))


def select_thresholds(s: np.ndarray, y: np.ndarray, abstention_budget: float = 0.10,
                      precision_target: float = 0.85) -> tuple[float, float]:
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    theta_low = float(np.quantile(s, abstention_budget))
    order = np.argsort(-s)
    ss, yy = s[order], y[order]
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)      # precision of {s >= ss[i]}
    ok = np.nonzero(precision >= precision_target)[0]
    theta_high = float(ss[ok[-1]]) if len(ok) else 1.0
    return theta_low, max(theta_high, theta_low)


def correctness_label(metric: str, prediction: str, golds: Sequence[str], *, f1_threshold: float = 0.80,
                      unanswerable: bool = False) -> int:
    """Binary target y of the composite correctness criterion, computed with the scorer of the dataset's
    primary metric: token-overlap metrics (F1) count as correct at F1 >= 0.80; Hits@1, accuracy (exact
    label for yes/no items) and exact match count as correct at a score of 1. For TempRAG, whose primary
    metric (context faithfulness) is not a correctness measure, normalised exact correctness against the
    reference answer is used. An abstention, a statement of insufficiency, or any answer to an
    unanswerable question has y = 0."""
    from .eval.metrics import answer_score
    from .text import is_insufficient
    if unanswerable or not prediction or is_insufficient(prediction):
        return 0
    if metric == "f1":
        return int(answer_score("f1", prediction, golds) >= f1_threshold)
    if metric == "context_faithfulness":
        metric = "exact_match"
    return int(answer_score(metric, prediction, golds) >= 1.0)
