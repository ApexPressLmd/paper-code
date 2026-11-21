"""Regenerate the reported tables from per-question prediction records.

Prediction records (one row per dataset x question x system x seed):
    dataset, qid, system, seed, unanswerable, abstained, primary_score
Calibration records (PEAK-RAG; one row per dataset x question x split):
    dataset, qid, split, unanswerable, v_branch, c_conf, u_faith, s, y, action_initial, action_final

Functions map to: Table 4 (main_table), Table 5 (paired_table), Supplementary Table S14
(all_pairs_table), Figure 6 (rank_analysis), Figure 9 and Supplementary Table S15 (calibration_report),
Supplementary Table S16 (lodo_table), Table 6 / ablations (main_table on variant systems).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..calibration import fit_gate, score, select_thresholds
from . import stats as S
from .selective import abstention_metrics, risk_coverage


def main_table(pred: pd.DataFrame, datasets: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mean and standard deviation over seeds of 100 x the mean full-coverage primary score."""
    per_seed = pred.groupby(["system", "dataset", "seed"])["primary_score"].mean().mul(100)
    agg = per_seed.groupby(["system", "dataset"]).agg(["mean", "std"])
    mean, sd = agg["mean"].unstack(), agg["std"].unstack()
    if datasets:
        mean, sd = mean[list(datasets)], sd[list(datasets)]
    fmt = mean.round(1).astype(str) + " ± " + sd.round(2).astype(str)
    return mean, sd, fmt


def _per_question(pred: pd.DataFrame, dataset: str, systems: Sequence[str]) -> pd.DataFrame:
    sub = pred[(pred.dataset == dataset) & pred.system.isin(systems)]
    return sub.groupby(["qid", "system"])["primary_score"].mean().unstack().dropna()


def paired_table(pred: pd.DataFrame, comparators: Mapping[str, str], system: str = "PEAK-RAG",
                 n_boot: int = 10000, n_perm: int = 10000, seed: int = 20261003, level: float = 0.95) -> pd.DataFrame:
    rows = []
    for i, (dataset, comp) in enumerate(comparators.items()):
        q = _per_question(pred, dataset, [system, comp])
        r = S.paired_comparison(q[system].to_numpy(), q[comp].to_numpy(), n_boot, n_perm, seed + i, level)
        rows.append({"dataset": dataset, "comparator": comp, **r})
    out = pd.DataFrame(rows)
    out["p_holm"] = S.holm(out.p_raw.to_numpy())
    return out


def all_pairs_table(pred: pd.DataFrame, system: str = "PEAK-RAG", n_boot: int = 10000, n_perm: int = 10000,
                    seed: int = 20261004, level: float = 0.95) -> pd.DataFrame:
    rows = []
    k = 0
    for dataset in sorted(pred.dataset.unique()):
        for comp in sorted(s for s in pred.system.unique() if s != system):
            q = _per_question(pred, dataset, [system, comp])
            if q.empty:
                continue
            r = S.paired_comparison(q[system].to_numpy(), q[comp].to_numpy(), n_boot, n_perm, seed + k, level)
            rows.append({"dataset": dataset, "comparator": comp, **r})
            k += 1
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm_all"] = S.holm(out.p_raw.to_numpy())
    return out


def abstention_by_system(pred: pd.DataFrame) -> pd.DataFrame:
    """Abstention precision / recall / F1, false-refusal rate and coverage per system, pooled over the
    datasets, as means over seeds."""
    rows = []
    for (system, seed), g in pred.groupby(["system", "seed"]):
        m = abstention_metrics(g.abstained.astype(bool), g.unanswerable.astype(bool))
        rows.append({"system": system, "seed": seed, **m})
    return pd.DataFrame(rows).groupby("system").mean(numeric_only=True).drop(columns="seed")


def ablation_table(pred: pd.DataFrame, datasets: Sequence[str] | None = None) -> pd.DataFrame:
    """Table 6 layout: primary metric per dataset, path recall, abstention F1 and hallucination rate
    (mean +- std over seeds)."""
    mean, sd, _ = main_table(pred, datasets)
    out = mean.round(1).astype(str) + " ± " + sd.round(2).astype(str)
    per_seed = pred.groupby(["system", "seed"]).agg(PathRec=("path_recall", "mean"),
                                                    Hallu=("hallucinated", lambda x: 100 * pd.to_numeric(x).mean()))
    ab = pd.DataFrame([{"system": s, "seed": sd_, "AbstF1": abstention_metrics(g.abstained.astype(bool),
                                                                               g.unanswerable.astype(bool))["f1"]}
                       for (s, sd_), g in pred.groupby(["system", "seed"])]).set_index(["system", "seed"])
    per_seed = per_seed.join(ab)
    for col in ("PathRec", "AbstF1", "Hallu"):
        agg = per_seed[col].groupby("system").agg(["mean", "std"])
        out[col] = agg["mean"].round(3).astype(str) + " ± " + agg["std"].round(3).astype(str)
    return out


def aurc_by_system(pred: pd.DataFrame) -> pd.DataFrame:
    """Risk-coverage area per system on the pooled test split, ranking answers by the system's
    confidence (the gate score s for PEAK-RAG) with correctness y."""
    rows = []
    for system, g in pred[pred.confidence.notna()].groupby("system"):
        _, _, aurc = risk_coverage(g.confidence.to_numpy(float), g.y.to_numpy(float))
        rows.append({"system": system, "aurc": aurc, "n": len(g)})
    return pd.DataFrame(rows)


def rank_analysis(mean: pd.DataFrame, alpha: float = 0.05) -> dict:
    """mean: systems x datasets. Returns Friedman statistic, mean ranks and the Nemenyi CD."""
    res = S.friedman_nemenyi(mean.T.to_numpy(), alpha)
    res["mean_ranks"] = dict(zip(mean.index, map(float, res["mean_ranks"])))
    return res


# ---------------------------------------------------------------------------- calibration
def calibration_report(cal: pd.DataFrame, theta_low: float, theta_high: float, n_bins: int = 10) -> dict:
    dev, test = cal[cal.split == "dev"], cal[cal.split == "test"]
    out = {
        "dev_ece": S.ece_equal_frequency(dev.s.to_numpy(), dev.y.to_numpy(), n_bins),
        "test_ece": S.ece_equal_frequency(test.s.to_numpy(), test.y.to_numpy(), n_bins),
        "test_brier": S.brier(test.s, test.y), "test_nll": S.nll(test.s, test.y), "test_auroc": S.auroc(test.s, test.y),
        "test_reliability": S.reliability_bins(test.s.to_numpy(), test.y.to_numpy(), n_bins),
        "per_dataset": {},
    }
    hi = test[test.s >= theta_high]
    out["high_score_precision"] = float(hi.y.mean()) if len(hi) else float("nan")
    out["high_score_n"] = int(len(hi))
    for d, g in test.groupby("dataset"):
        h = g[g.s >= theta_high]
        out["per_dataset"][d] = {"ece": S.ece_equal_frequency(g.s.to_numpy(), g.y.to_numpy(), n_bins),
                                 "high_score_precision": float(h.y.mean()) if len(h) else float("nan")}
    ab = abstention_metrics(test.action_final == "abstain", test.unanswerable.astype(bool))
    out["abstention"] = ab
    _, _, aurc = risk_coverage(test.s.to_numpy(), test.y.to_numpy())
    out["aurc"] = aurc
    out["dev_abstain_initial_rate"] = float((dev.s < theta_low).mean())
    out["test_answerable_below_theta_low"] = float((test[test.unanswerable == 0].s < theta_low).mean())
    return out


def lodo_table(cal: pd.DataFrame, eta: float = 0.01, n_bins: int = 10, budget: float = 0.10,
               target: float = 0.85) -> pd.DataFrame:
    """Leave-one-dataset-out stress test (Supplementary Table S16).

    Protocol of this release: for each held-out dataset, the gate (weights and thresholds) is fitted on
    the development records of the other four datasets ("LODO") or of all five ("pooled"), and evaluated
    on all test records of the held-out dataset with ten equal-width bins. Equal-width bins differ from
    the equal-frequency bins of the per-dataset test ECE, so the two ECE scales are not comparable.
    """
    rows = []
    feats = lambda d: np.c_[d.v_branch, 1 - d.c_conf, d.u_faith]
    dev = cal[cal.split == "dev"]
    for held in sorted(cal.dataset.unique()):
        test = cal[(cal.split == "test") & (cal.dataset == held)]
        for mode, train in (("LODO", dev[dev.dataset != held]), ("pooled", dev)):
            w = fit_gate(feats(train), train.y.to_numpy(), eta)
            s_tr = score(feats(train), w)
            _, th = select_thresholds(s_tr, train.y.to_numpy(), budget, target)
            s_te = score(feats(test), w)
            hi = s_te >= th
            rows.append({"held_out": held, "fit": mode, "ece": S.ece_equal_width(s_te, test.y.to_numpy(), n_bins),
                         "theta_high": th, "n_high": int(hi.sum()),
                         "high_score_precision": float(test.y.to_numpy()[hi].mean()) if hi.any() else float("nan"),
                         "n_high_correct": int(test.y.to_numpy()[hi].sum())})
    out = pd.DataFrame(rows)
    macro = out.groupby("fit").agg(ece=("ece", "mean"), n_high=("n_high", "sum"), n_high_correct=("n_high_correct", "sum")).reset_index()
    macro["high_score_precision"] = macro.n_high_correct / macro.n_high
    macro["held_out"] = "MACRO (ECE) / overall (precision)"
    return pd.concat([out, macro], ignore_index=True)
