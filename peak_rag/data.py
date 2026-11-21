"""Data schema and the evidence-removal unanswerable subset (Section 4.1).

Each benchmark is converted once into two JSONL files (conversion per corpus: Supplementary Table S9).

corpus.jsonl     {"doc_id", "title", "text"}
questions.jsonl  {"qid", "dataset", "split": "dev"|"test", "question", "answers": [...],
                  "route": "parallel"|"chained" (optional, oracle router), "gold_support": [passage ids],
                  "gold_path": [concepts or fact ids] (optional), "unanswerable": 0|1}

The graph is built from the corpus alone; no question text, reference answer, gold supporting evidence
or correctness label enters graph mining, de-duplication or indexing.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np

from .text import normalize_answer


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def make_evidence_removal_subset(corpus: list[dict], questions: list[dict], n_per_dataset: int = 100,
                                 seed: int = 20261004) -> tuple[list[dict], list[dict]]:
    """Mark ``n_per_dataset`` test questions per dataset unanswerable by removing the documents that hold
    their gold supporting evidence from the inference corpus; a question is eligible only if, after the
    removal, no remaining document contains any of its reference answers. Returns (corpus', questions')."""
    rng = np.random.default_rng(seed)
    removed: set[str] = set()
    chosen: set[str] = set()
    by_ds: dict[str, list[dict]] = {}
    for q in questions:
        if q["split"] == "test" and q.get("gold_support"):
            by_ds.setdefault(q["dataset"], []).append(q)
    docs = {d["doc_id"]: d for d in corpus}
    needed_by: dict[str, set[str]] = {}
    for q in questions:
        for s in q.get("gold_support", []):
            needed_by.setdefault(s.split("#")[0], set()).add(q["qid"])
    for ds, qs in sorted(by_ds.items()):
        picked = 0
        for i in rng.permutation(len(qs)):
            q = qs[i]
            support_docs = {s.split("#")[0] for s in q["gold_support"]}
            # removal must not take away the evidence of another question that stays answerable
            if any(needed_by.get(d, set()) - chosen - {q["qid"]} for d in support_docs):
                continue
            remaining = [d for k, d in docs.items() if k not in removed | support_docs]
            ans = [normalize_answer(a) for a in q["answers"] if normalize_answer(a)]
            if any(a and a in normalize_answer(d["text"]) for d in remaining for a in ans):
                continue
            removed |= support_docs
            chosen.add(q["qid"])
            picked += 1
            if picked >= n_per_dataset:
                break
    new_corpus = [d for d in corpus if d["doc_id"] not in removed]
    new_q = [dict(q, unanswerable=int(q["qid"] in chosen)) for q in questions]
    return new_corpus, new_q
