#!/usr/bin/env python3
"""Construct the evidence-removal (corpus-relative) unanswerable subset (Section 4.1).

python scripts/make_unanswerable.py --corpus corpus.jsonl --questions questions.jsonl \
    --out-dir data/x_unans --per-dataset 100
Writes corpus.jsonl (inference corpus with the supporting documents removed), questions.jsonl (with the
``unanswerable`` flag) and unanswerable_ids.txt (the identifiers listed in Supplementary Table S9).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_rag.config import load_config  # noqa: E402
from peak_rag.data import make_evidence_removal_subset, read_jsonl, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-dataset", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    n = a.per_dataset or cfg.evaluation.unanswerable_per_dataset
    corpus, questions = make_evidence_removal_subset(read_jsonl(a.corpus), read_jsonl(a.questions), n,
                                                     cfg.evaluation.resampling_seed_supp)
    os.makedirs(a.out_dir, exist_ok=True)
    write_jsonl(os.path.join(a.out_dir, "corpus.jsonl"), corpus)
    write_jsonl(os.path.join(a.out_dir, "questions.jsonl"), questions)
    ids = [q["qid"] for q in questions if q["unanswerable"]]
    with open(os.path.join(a.out_dir, "unanswerable_ids.txt"), "w") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"{len(ids)} unanswerable questions; corpus {len(corpus)} documents")


if __name__ == "__main__":
    main()
