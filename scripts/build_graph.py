#!/usr/bin/env python3
"""Offline stage (Algorithm 1): build the indexing graph G and the prerequisite digraph P of one corpus.

python scripts/build_graph.py --corpus data/<name>/corpus.jsonl --out runs/<name>/kb.pkl [--backend hf]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_rag.backends import make_backends  # noqa: E402
from peak_rag.config import ablation_preset, load_config, parse_sets  # noqa: E402
from peak_rag.data import read_jsonl  # noqa: E402
from peak_rag.offline.build import build_knowledge_base  # noqa: E402
from peak_rag.offline.prereq import source_shares  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default="offline", choices=["offline", "hf"])
    ap.add_argument("--variant", default="full", help="e.g. typed-only, cooc-only, order-only")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    a = ap.parse_args()
    cfg = load_config(a.config, {**ablation_preset(a.variant), **parse_sets(a.set)})
    gen, emb, nli = make_backends(cfg, a.backend)
    t0 = time.time()
    kb = build_knowledge_base(read_jsonl(a.corpus), cfg, gen, emb, nli)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    kb.save(a.out)
    summary = {"passages": len(kb.passages), "facts": len(kb.facts), "active_facts": len(kb.active_facts),
               "concepts": len(kb.concepts), "prerequisite_edges": len(kb.prereq),
               "edge_source_shares": source_shares(kb.prereq), "graph_nodes": kb.graph.number_of_nodes(),
               "graph_edges": kb.graph.number_of_edges(), "build_calls": kb.build_calls,
               "conflict_decisions": {d: sum(1 for c in kb.conflict_log if c["decision"] == d)
                                      for d in ("keep_both", "drop_first", "drop_second", "merge", "unresolved")},
               "wall_clock_s": round(time.time() - t0, 2)}
    with open(os.path.splitext(a.out)[0] + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
