"""Selective prediction and abstention metrics (Sections 4.1 and 4.7).

Abstaining on an evidence-removal unanswerable question is a true positive. Reported: abstention
precision, recall and F1; false-refusal rate on answerable questions; coverage; selective risk; the
risk-coverage curve and the area under it (AURC).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def abstention_metrics(abstained: Sequence[bool], unanswerable: Sequence[bool]) -> dict:
    a = np.asarray(abstained, bool)
    u = np.asarray(unanswerable, bool)
    tp = int((a & u).sum())
    fp = int((a & ~u).sum())
    fn = int((~a & u).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n_ans = int((~u).sum())
    return {
        "true_abstentions": tp, "false_refusals": fp, "unanswerable_answered": fn,
        "abstentions": tp + fp, "precision": precision, "recall": recall, "f1": f1,
        "false_refusal_rate": fp / n_ans if n_ans else 0.0,
        "coverage": 1.0 - (tp + fp) / len(a) if len(a) else 0.0,
        "unanswerable_answered_rate": fn / int(u.sum()) if u.sum() else 0.0,
    }


def risk_coverage(confidence: Sequence[float], correct: Sequence[float]) -> tuple[np.ndarray, np.ndarray, float]:
    """Answers sorted by decreasing confidence; at coverage i/n the selective risk is the error rate of
    the i most confident answers. AURC is the mean selective risk over the n coverage levels."""
    c = np.asarray(confidence, float)
    y = np.asarray(correct, float)
    order = np.argsort(-c, kind="mergesort")
    err = 1.0 - y[order]
    n = len(c)
    risk = np.cumsum(err) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    return coverage, risk, float(risk.mean())


def selective_risk(answered_correct: Sequence[float]) -> float:
    a = np.asarray(answered_correct, float)
    return float(1.0 - a.mean()) if len(a) else 0.0
