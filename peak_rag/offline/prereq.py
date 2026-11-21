"""Prerequisite mining (Section 3.3; Algorithm 1, lines 14-19).

Three signals of decreasing reliability propose a candidate edge c' -> c ("c' is a prerequisite of c"):

* typed relation   omega = 1.0   requires / precedes / based_on (direction below)
* co-occurrence    omega = 0.5   c' is mentioned before c inside one passage
* first-mention    omega = 0.3 / (j - i)   from the first-appearance order i < j of c', c

Duplicate edges keep the maximum omega; the signal that supplied the maximum is stored as the edge
``source`` (used for the per-source audit and the teleport-mass shares of Section 4.5).

Direction of typed relations. ``X requires Y`` and ``X based_on Y`` make Y a prerequisite of X
(edge Y -> X); ``X precedes Y`` makes X a prerequisite of Y (edge X -> Y).

Concept pairs "observed in the ontology layer" are, in this implementation, pairs of active concepts
that occur in the same document. The first-mention ranks i < j are the concepts' first-appearance ranks in
the corpus D (documents in corpus order, passages in reading order).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

SIGNAL_ORDER = ("typed", "cooc", "order")   # tie-break when two signals give the same omega


def mine_prerequisites(facts: Iterable[dict], mentions: dict[str, list[tuple[str, str, int]]],
                       concepts: set[str], cfg, signals: Iterable[str] = SIGNAL_ORDER,
                       corpus_rank: dict[str, int] | None = None) -> dict[tuple[str, str], dict]:
    """
    facts:    active facts with canonical ``head``, ``relation``, ``tail``.
    mentions: doc_id -> ordered list of (concept, passage_id, position) for every concept mention,
              sorted by (passage order, character offset).
    concepts: the concept vocabulary of the ontology layer.
    corpus_rank: first-appearance rank of each concept in D; if omitted, ranks are taken from the
              documents in the order of ``mentions``.
    Returns {(c_prime, c): {"omega": float, "source": str}}.
    """
    pm = cfg.prerequisite_mining
    if pm.duplicate_merge != "max":
        raise ValueError("only the max rule for duplicate edges is implemented")
    if corpus_rank is None:
        corpus_rank = {}
        for doc_mentions in mentions.values():
            for concept, _pid, _pos in doc_mentions:
                corpus_rank.setdefault(concept, len(corpus_rank))
    signals = set(signals)
    edges: dict[tuple[str, str], dict] = {}

    def add(cp: str, c: str, omega: float, source: str) -> None:
        if cp == c or cp not in concepts or c not in concepts or omega <= 0:
            return
        cur = edges.get((cp, c))
        better = cur is None or omega > cur["omega"] or (
            omega == cur["omega"] and SIGNAL_ORDER.index(source) < SIGNAL_ORDER.index(cur["source"]))
        if better:                                   # duplicate-edge merge: max omega
            edges[(cp, c)] = {"omega": float(omega), "source": source}

    if "typed" in signals:
        direction = dict(pm.typed_direction)     # relation -> "tail_to_head" | "head_to_tail"
        for f in facts:
            rel = f["relation"]
            if rel not in set(pm.typed_relations):
                continue
            if direction[rel] == "tail_to_head":
                add(f["tail"], f["head"], pm.omega_typed, "typed")
            else:
                add(f["head"], f["tail"], pm.omega_typed, "typed")

    if "cooc" in signals:
        for doc_mentions in mentions.values():
            by_passage: dict[str, list[tuple[int, str]]] = defaultdict(list)
            for concept, pid, pos in doc_mentions:
                by_passage[pid].append((pos, concept))
            for items in by_passage.values():
                first: dict[str, int] = {}
                for pos, concept in sorted(items):
                    first.setdefault(concept, pos)
                ordered = sorted(first.items(), key=lambda kv: kv[1])
                for a in range(len(ordered)):
                    for b in range(a + 1, len(ordered)):
                        if ordered[a][1] < ordered[b][1]:      # earlier mention -> later mention
                            add(ordered[a][0], ordered[b][0], pm.omega_cooc, "cooc")

    if "order" in signals:
        for doc_mentions in mentions.values():
            present = sorted({c for c, _pid, _pos in doc_mentions}, key=lambda c: corpus_rank[c])
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    i, j = corpus_rank[present[a]], corpus_rank[present[b]]
                    add(present[a], present[b], pm.omega_order_numerator / (j - i), "order")
    return edges


def foundationality(prereq: dict[tuple[str, str], dict]) -> dict[str, int]:
    """f(c') = prerequisite out-degree of c' in P (Eq. 1 with omega * rel = 1)."""
    out: dict[str, int] = defaultdict(int)
    for (cp, _c) in prereq:
        out[cp] += 1
    return dict(out)


def source_shares(prereq: dict[tuple[str, str], dict]) -> dict[str, float]:
    n = max(1, len(prereq))
    counts = defaultdict(int)
    for e in prereq.values():
        counts[e["source"]] += 1
    return {s: counts[s] / n for s in SIGNAL_ORDER}
