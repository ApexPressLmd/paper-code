#!/usr/bin/env python3
"""Add a semantic-entropy confidence to records of an external system (e.g. PARKA + semantic entropy).

Each input record must carry the evaluation fields (dataset, qid, system, seed, split, unanswerable,
abstained, primary_score, y) and ``samples``: the k = 5 answers sampled at T = 0.7 under the system's
own configuration. The output records can be passed to evaluate.py with --threshold-systems.

python scripts/external_confidence.py --records parka.jsonl --out parka_se.jsonl [--backend hf]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_rag.backends import make_backends  # noqa: E402
from peak_rag.baselines import semantic_entropy_confidence  # noqa: E402
from peak_rag.config import load_config  # noqa: E402
from peak_rag.data import read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="offline", choices=["offline", "hf"])
    a = ap.parse_args()
    cfg = load_config()
    _, _, nli = make_backends(cfg, a.backend)
    recs = read_jsonl(a.records)
    for r in recs:
        r["confidence"] = semantic_entropy_confidence(r.get("samples", []), nli)
    write_jsonl(a.out, recs)
    print(f"{len(recs)} records written to {a.out}")


if __name__ == "__main__":
    main()
