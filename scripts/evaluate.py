#!/usr/bin/env python3
"""Regenerate the result tables from scored per-question records.

python scripts/evaluate.py --records runs/x/*.jsonl --out runs/x/tables \
    [--system PEAK-RAG] [--comparators-json '{"MLPQ": "Yuan et al.", ...}'] [--theta 0.35 0.70] \
    [--threshold-systems dense+selfcheck dense+semantic-entropy PARKA]

Writes
  main_table.csv          Table 4  (primary metric, mean +- std over seeds; test split)
  paired_comparisons.csv  Table 5  (question-level paired tests, Holm across the pre-specified comparisons)
  all_pairs.csv           Supplementary Table S14
  ranks.json              Figure 6 (Friedman test, mean ranks, Nemenyi CD)
  ablation_table.csv      Table 6 layout (primary metrics, path recall, abstention F1, hallucination rate)
  abstention.csv          abstention precision / recall / F1, false-refusal rate, coverage per system
  auxiliary.csv           context faithfulness, context relevance, retrieval and path recall, hallucination
  aurc.csv                risk-coverage areas (Figure 9, right), including the thresholded alternatives
  calibration.json        Figure 9 (left), Supplementary Table S15, dev/test rejection rates
  lodo.csv                Supplementary Table S16
  efficiency.csv          Table 8 (calls, tokens, latency per question)
"""
import argparse
import json
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_rag.config import load_config  # noqa: E402
from peak_rag.data import read_jsonl  # noqa: E402
from peak_rag.eval import tables as T  # noqa: E402
from peak_rag.runner import calibration_frame, prediction_frame, thresholded_systems  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="PEAK-RAG")
    ap.add_argument("--comparators-json", default=None)
    ap.add_argument("--theta", type=float, nargs=2, default=None)
    ap.add_argument("--threshold-systems", nargs="*", default=["dense+selfcheck", "dense+semantic-entropy"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    ev = cfg.evaluation
    recs = [r for f in a.records for r in read_jsonl(f)]
    os.makedirs(a.out, exist_ok=True)
    pred = prediction_frame(recs)
    pred = pd.concat([pred, thresholded_systems(pred, cfg.gating.abstention_budget, a.threshold_systems)],
                     ignore_index=True)
    pred.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    test = pred[pred.split == "test"]
    for (system, ds, seed), n in test.groupby(["system", "dataset", "seed"]).size().items():
        if ds in ev.primary_metric and n != ev.test_questions_per_dataset:
            warnings.warn(f"{system} / {ds} / seed {seed}: {n} test questions, expected {ev.test_questions_per_dataset}")

    mean, sd, fmt = T.main_table(test)
    fmt.to_csv(os.path.join(a.out, "main_table.csv"))
    comparators = json.loads(a.comparators_json) if a.comparators_json else dict(ev.comparators)
    comparators = {d: c for d, c in comparators.items() if d in set(test.dataset) and c in set(test.system)}
    if comparators:
        T.paired_table(test, comparators, a.system, ev.n_bootstrap, ev.n_permutations, ev.resampling_seed_main,
                       ev.confidence_level).to_csv(os.path.join(a.out, "paired_comparisons.csv"), index=False)
    T.all_pairs_table(test, a.system, ev.n_bootstrap, ev.n_permutations, ev.resampling_seed_supp,
                      ev.confidence_level).to_csv(os.path.join(a.out, "all_pairs.csv"), index=False)
    if mean.shape[0] >= 3 and mean.shape[1] >= 2:
        with open(os.path.join(a.out, "ranks.json"), "w") as fh:
            json.dump(T.rank_analysis(mean.dropna()), fh, indent=2)
    T.ablation_table(test).to_csv(os.path.join(a.out, "ablation_table.csv"))
    T.abstention_by_system(test).to_csv(os.path.join(a.out, "abstention.csv"))
    aux_cols = ["context_faithfulness", "context_relevance", "retrieval_recall", "path_recall", "hallucinated"]
    test[["system"] + aux_cols].assign(**{c: pd.to_numeric(test[c]) for c in aux_cols}) \
        .groupby("system").mean().to_csv(os.path.join(a.out, "auxiliary.csv"))
    T.aurc_by_system(test).to_csv(os.path.join(a.out, "aurc.csv"), index=False)
    test.groupby("system")[["n_calls", "tokens", "latency_s"]].mean().to_csv(os.path.join(a.out, "efficiency.csv"))

    cal = calibration_frame([r for r in recs if r["system"] == a.system])
    if len(cal):
        cal.to_csv(os.path.join(a.out, "calibration_records.csv"), index=False)
        lo, hi = a.theta or (cfg.gating.theta_low, cfg.gating.theta_high)
        if (cal.split == "test").any() and (cal.split == "dev").any():
            rep = T.calibration_report(cal, lo, hi, ev.ece_bins)
            with open(os.path.join(a.out, "calibration.json"), "w") as fh:
                json.dump(rep, fh, indent=2)
            if cal.dataset.nunique() >= 2:
                T.lodo_table(cal, cfg.gating.l2_eta, ev.ece_bins, cfg.gating.abstention_budget,
                             cfg.gating.precision_target).to_csv(os.path.join(a.out, "lodo.csv"), index=False)
    print(fmt.to_string())
    print(f"tables written to {a.out}")


if __name__ == "__main__":
    main()
