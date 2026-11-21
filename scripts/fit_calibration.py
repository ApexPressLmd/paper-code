#!/usr/bin/env python3
"""Fit the gate calibration on development records (Section 3.6).

python scripts/fit_calibration.py --records runs/x/peak_dev.jsonl --out runs/x/calibration.json
Only records with split == dev are used. Feature masks reproduce the signal ablations of
Supplementary Table S17 (e.g. --drop v_branch).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_rag.calibration import fit_gate, score, select_thresholds  # noqa: E402
from peak_rag.config import load_config  # noqa: E402
from peak_rag.data import read_jsonl  # noqa: E402
from peak_rag.eval.stats import ece_equal_frequency  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--drop", nargs="*", default=[], choices=["v_branch", "c_conf", "u_faith"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    recs = [r for f in a.records for r in read_jsonl(f) if r["split"] == "dev" and r.get("signals")]
    if not recs:
        sys.exit("no development records with gate signals")
    X = np.array([[r["signals"]["v_branch"], 1 - r["signals"]["c_conf"], r["signals"]["u_faith"]] for r in recs])
    y = np.array([r["y"] for r in recs], float)
    mask = [n not in a.drop for n in ("v_branch", "c_conf", "u_faith")]
    w = fit_gate(X, y, cfg.gating.l2_eta, mask)
    s = score(X, w)
    lo, hi = select_thresholds(s, y, cfg.gating.abstention_budget, cfg.gating.precision_target)
    out = {"weights": {k: w[k] for k in ("w1", "w2", "w3", "b")}, "theta_low": lo, "theta_high": hi,
           "n_records": len(recs), "n_positive": int(y.sum()),
           "dev_ece": ece_equal_frequency(s, y, cfg.evaluation.ece_bins), "converged": w["converged"],
           "dropped_features": a.drop}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
