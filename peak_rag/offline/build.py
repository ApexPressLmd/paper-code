"""Offline knowledge-base construction (Section 3.2; Algorithm 1).

Steps: split passages -> ExtractionAgent -> candidate conflicts (same head and relation with a
different tail, or fact-embedding similarity > delta = 0.83) -> KeepDropOrMerge grounded in the evidence
passages -> three-layer memory (ontology, fact, passage) -> thematic denoising (schemas promoted only
above the P40 frequency percentile) -> hierarchical indexing multi-digraph of entities, types and
passages -> structural unification (merge nodes with similarity >= 0.86; bridge weights 1.0 / 0.6 / 0.4)
-> prerequisite mining (prereq.py). The fact embeddings used for candidate conflicts are computed on
the extracted surface names, before unification, as in the order of Algorithm 1.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np
from scipy import sparse

from .. import prompts
from ..backends.base import CallLog, CountingLLM
from ..text import content_tokens, normalize_answer, split_sentences
from .prereq import foundationality, mine_prerequisites


@dataclass
class KnowledgeBase:
    passages: list[dict]
    facts: list[dict]                               # all facts; ``active`` marks those kept
    schema_freq: dict[str, int]
    active_schemas: set[str]
    canonical: dict[str, str]                       # surface entity name -> canonical concept
    concepts: list[str]
    prereq: dict[tuple[str, str], dict]
    graph: nx.MultiDiGraph
    nodes: list[str]
    node_index: dict[str, int]
    transition: sparse.csr_matrix                   # column-stochastic P = A D^-1
    dangling: np.ndarray                            # boolean mask of nodes without out-edges
    concept_emb: np.ndarray
    fact_emb: np.ndarray
    passage_emb: np.ndarray
    conflict_log: list[dict] = field(default_factory=list)
    build_calls: dict = field(default_factory=dict)
    foundationality: dict[str, int] = field(default_factory=dict)
    passage_rep: dict[str, str] = field(default_factory=dict)
    type_rep: dict[str, str] = field(default_factory=dict)

    @property
    def active_facts(self) -> list[dict]:
        return [f for f in self.facts if f["active"]]

    def fact_text(self, f: dict) -> str:
        return f"{f['head']} {f['relation'].replace('_', ' ')} {f['tail']}"

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "KnowledgeBase":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------- helpers
class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def split_passages(corpus: list[dict], max_sentences: int = 4) -> list[dict]:
    """Split each document into passages of at most ``max_sentences`` sentences, keeping order."""
    passages = []
    for doc in corpus:
        paras = [p for p in doc["text"].split("\n\n") if p.strip()] or [doc["text"]]
        order = 0
        for para in paras:
            sents = split_sentences(para)
            for i in range(0, len(sents), max_sentences):
                passages.append({"pid": f"{doc['doc_id']}#{order}", "doc_id": doc["doc_id"], "order": order,
                                 "text": " ".join(sents[i:i + max_sentences])})
                order += 1
    return passages


def _parse_json(text: str, default):
    try:
        start = min([i for i in (text.find("["), text.find("{")) if i >= 0])
        return json.loads(text[start:])
    except (ValueError, json.JSONDecodeError):
        return default


def _similar_pairs(emb: np.ndarray, threshold: float, strict: bool, chunk: int = 2048) -> list[tuple[int, int]]:
    pairs = []
    n = emb.shape[0]
    for s in range(0, n, chunk):
        sims = emb[s:s + chunk] @ emb.T
        rows, cols = np.nonzero(sims > threshold if strict else sims >= threshold)
        for r, c in zip(rows, cols):
            i = s + r
            if i < c:
                pairs.append((int(i), int(c)))
    return pairs


# ---------------------------------------------------------------------------- conflict resolution
def _source_based(justification: str, passages: list[str]) -> bool:
    """A justification is source-based when it quotes or paraphrases one of the evidence passages
    (at least three content tokens in common) and does not rest on similarity."""
    if not justification.strip() or "similar" in justification.lower():
        return False
    jt = set(content_tokens(justification))
    return any(len(jt & set(content_tokens(p))) >= 3 for p in passages)


def keep_drop_or_merge(fa: dict, fb: dict, passages: dict[str, str], llm: CountingLLM, nli,
                       max_new_tokens: int = 256) -> tuple[str, str, float]:
    """KeepDropOrMerge (Section 3.2). Returns (decision, justification, nli_contradiction).

    Facts with different time intervals or scopes are retained; similarity alone never justifies a drop;
    a passage-level NLI contradiction only triggers review (it is passed to the resolution agent and never
    deletes a fact by itself); a deletion requires an explicit source-based justification; unresolved pairs
    stay in the graph with their provenance and are flagged uncertain; duplicates merge only when meaning
    and qualifiers are equivalent."""
    pa, pb = passages[fa["pid"]], passages[fb["pid"]]
    contradiction = float(nli.entail([pa], [pb])[0, 2])
    for q in ("time", "scope"):
        if (fa.get(q) or "") != (fb.get(q) or ""):
            return "keep_both", f"different {q}", contradiction
    req = prompts.resolve_conflict({k: fa[k] for k in ("head", "relation", "tail")},
                                   {k: fb[k] for k in ("head", "relation", "tail")}, pa, pb, contradiction,
                                   max_new_tokens)
    resp = _parse_json(llm(req), {})
    decision = resp.get("decision", "unresolved") if isinstance(resp, dict) else "unresolved"
    justification = str(resp.get("justification", "")) if isinstance(resp, dict) else ""
    if decision in ("drop_first", "drop_second"):
        if not _source_based(justification, [pa, pb]):
            return "unresolved", "deletion without a source-based justification is not applied", contradiction
        return decision, justification, contradiction
    if decision == "merge":
        ta = f"{fa['head']} {fa['relation']} {fa['tail']}"
        tb = f"{fb['head']} {fb['relation']} {fb['tail']}"
        if normalize_answer(ta) == normalize_answer(tb):
            return "merge", justification, contradiction
        probs = nli.entail([ta, tb], [tb, ta])
        if probs[0, 0] >= 0.5 and probs[1, 0] >= 0.5:
            return "merge", justification, contradiction
        return "keep_both", "merge refused: meanings not equivalent", contradiction
    if decision == "keep_both":
        return "keep_both", justification, contradiction
    return "unresolved", justification or "no decision", contradiction


def _unify(names: list[str], embedder, threshold: float, weight: Counter) -> dict[str, str]:
    """Structural unification: merge items with embedding similarity >= threshold (union-find); the
    representative is the most frequent member (then the shortest, then alphabetical)."""
    if not names:
        return {}
    uf = _UnionFind(names)
    for i, j in _similar_pairs(embedder.encode(names), threshold, strict=False):
        uf.union(names[i], names[j])
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        groups[uf.find(n)].append(n)
    rep = {}
    for members in groups.values():
        r = sorted(members, key=lambda m: (-weight[m], len(m), m))[0]
        for m in members:
            rep[m] = r
    return rep


# ---------------------------------------------------------------------------- main build
def build_knowledge_base(corpus: list[dict], cfg, llm_backend, embedder, nli,
                         signals: Iterable[str] | None = None) -> KnowledgeBase:
    """Algorithm 1, in the order given there."""
    cc = cfg.construction
    log = CallLog()
    llm = CountingLLM(llm_backend, log)
    passages = split_passages(corpus)
    ptext = {p["pid"]: p["text"] for p in passages}

    # (1) ExtractionAgent: one call per passage; each fact carries its evidence passage
    facts: list[dict] = []
    for p in passages:
        for t in _parse_json(llm(prompts.extract_facts(cfg, p["text"])), []):
            if not isinstance(t, dict) or not t.get("head") or not t.get("tail") or not t.get("relation"):
                continue
            facts.append({"fid": len(facts), "head": t["head"].strip(), "relation": t["relation"].strip().lower(),
                          "tail": t["tail"].strip(), "pid": p["pid"], "time": t.get("time"), "scope": t.get("scope"),
                          "active": True, "uncertain": False, "merged_into": None})

    # (2) candidate conflicts: same head and relation with a different tail, or similarity > delta
    norm = lambda s: normalize_answer(s)  # noqa: E731
    candidates: set[tuple[int, int]] = set()
    by_hr: dict[tuple[str, str], list[int]] = defaultdict(list)
    for f in facts:
        by_hr[(norm(f["head"]), f["relation"])].append(f["fid"])
    for ids in by_hr.values():
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if norm(facts[ids[a]]["tail"]) != norm(facts[ids[b]]["tail"]):
                    candidates.add((ids[a], ids[b]))
    fact_texts = [f"{f['head']} {f['relation'].replace('_', ' ')} {f['tail']}" for f in facts]
    fact_emb = embedder.encode(fact_texts) if facts else np.zeros((0, embedder.dim), np.float32)
    candidates.update(_similar_pairs(fact_emb, cc.conflict_similarity_delta, strict=True))

    # (3) ConflictResolution grounded in the evidence passages
    conflict_log = []
    for a, b in sorted(candidates):
        fa, fb = facts[a], facts[b]
        if not (fa["active"] and fb["active"]):
            continue
        decision, why, contra = keep_drop_or_merge(fa, fb, ptext, llm, nli,
                                                   cfg.decoding.offline_max_new_tokens.resolve_conflict)
        if decision == "drop_first":
            fa["active"] = False
        elif decision == "drop_second":
            fb["active"] = False
        elif decision == "merge":
            fb["active"], fb["merged_into"] = False, fa["fid"]
        elif decision == "unresolved":
            fa["uncertain"] = fb["uncertain"] = True
        conflict_log.append({"a": a, "b": b, "decision": decision, "justification": why, "nli_contradiction": contra})

    # (4) three-layer memory; thematic denoising promotes a schema only above the tau-percentile
    schema_freq = Counter(f["relation"] for f in facts if f["active"])
    if schema_freq:
        cut = float(np.percentile(list(schema_freq.values()), cc.schema_percentile_tau))
        active_schemas = {s for s, n in schema_freq.items() if n > cut}
        if not active_schemas:
            # degenerate case not covered by the rule: all schema frequencies are equal, so none exceeds
            # the percentile; keep them all rather than deactivating the whole fact layer
            active_schemas = set(schema_freq)
    else:
        active_schemas = set()
    for f in facts:
        if f["relation"] not in active_schemas:
            f["active"] = False
    active = [f for f in facts if f["active"]]

    # (5) structural unification of entity, type and passage nodes (similarity >= 0.86)
    ent_freq = Counter([f["head"] for f in active] + [f["tail"] for f in active])
    canonical = _unify(sorted(ent_freq), embedder, cc.unification_threshold, ent_freq)
    type_rep = _unify(sorted(active_schemas), embedder, cc.unification_threshold, Counter(schema_freq))
    passage_rep = {p["pid"]: p["pid"] for p in passages}
    if passages:
        pemb = embedder.encode([p["text"] for p in passages])
        uf = _UnionFind([p["pid"] for p in passages])
        for i, j in _similar_pairs(pemb, cc.unification_threshold, strict=False):
            uf.union(passages[i]["pid"], passages[j]["pid"])   # representative: the earlier passage
        passage_rep = {p["pid"]: uf.find(p["pid"]) for p in passages}
    for f in facts:
        f["head_surface"], f["tail_surface"] = f["head"], f["tail"]
        f["head"], f["tail"] = canonical.get(f["head"], f["head"]), canonical.get(f["tail"], f["tail"])

    # concept vocabulary of the ontology layer and its mentions in reading order
    concepts = sorted({f["head"] for f in active} | {f["tail"] for f in active})
    cset = set(concepts)
    surface_by_concept: dict[str, set[str]] = defaultdict(set)
    for surf, can in canonical.items():
        if can in cset:
            surface_by_concept[can].add(normalize_answer(surf))
    mentions: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for p in passages:
        low = normalize_answer(p["text"])
        found = []
        for c in concepts:
            positions = [low.find(s) for s in surface_by_concept[c] if s and low.find(s) >= 0]
            if positions:
                found.append((min(positions), c))
        for pos, c in sorted(found):
            mentions[p["doc_id"]].append((c, p["pid"], p["order"] * 10_000_000 + pos))
    for doc in mentions:
        mentions[doc].sort(key=lambda m: m[2])
    corpus_rank: dict[str, int] = {}        # first-appearance order of each concept in D (corpus order)
    for doc in corpus:
        for c, _pid, _pos in mentions.get(doc["doc_id"], []):
            corpus_rank.setdefault(c, len(corpus_rank))

    # (6) prerequisite mining
    sig = list(signals) if signals is not None else list(cfg.ablation.mining_signals)
    prereq = mine_prerequisites(active, mentions, cset, cfg, sig, corpus_rank)

    # (7) hierarchical indexing multi-digraph of entities, types and passages, with bridge weights
    bw = cc.bridge_weights
    g = nx.MultiDiGraph()
    P = lambda pid: f"p:{passage_rep[pid]}"  # noqa: E731
    T = lambda s: f"t:{type_rep[s]}"         # noqa: E731
    for c in concepts:
        g.add_node(f"e:{c}", kind="entity")
    for s in active_schemas:
        g.add_node(T(s), kind="type")
    for p in passages:
        g.add_node(P(p["pid"]), kind="passage")
    for f in active:
        h, t = f"e:{f['head']}", f"e:{f['tail']}"
        g.add_edge(h, t, weight=bw.fact, view="fact", fid=f["fid"])
        g.add_edge(t, h, weight=bw.fact, view="fact", fid=f["fid"])
        for e in (h, t):
            g.add_edge(e, T(f["relation"]), weight=bw.ontology, view="ontology")
            g.add_edge(T(f["relation"]), e, weight=bw.ontology, view="ontology")
            g.add_edge(e, P(f["pid"]), weight=bw.source, view="source")
            g.add_edge(P(f["pid"]), e, weight=bw.source, view="source")
    for doc_mentions in mentions.values():
        for c, pid, _ in doc_mentions:
            if not g.has_edge(f"e:{c}", P(pid)):
                g.add_edge(f"e:{c}", P(pid), weight=bw.source, view="source")
                g.add_edge(P(pid), f"e:{c}", weight=bw.source, view="source")

    nodes = list(g.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    for pid, rep in passage_rep.items():          # merged passages resolve to their representative node
        index[f"p:{pid}"] = index[f"p:{rep}"]
    for s, rep in type_rep.items():
        index[f"t:{s}"] = index[f"t:{rep}"]
    rows, cols, vals = [], [], []
    for u, v, d in g.edges(data=True):
        rows.append(index[v])
        cols.append(index[u])
        vals.append(d["weight"])
    n = len(nodes)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))      # A[v, u] = weight(u -> v)
    out_w = np.asarray(A.sum(axis=0)).ravel()
    dangling = out_w == 0
    inv = np.where(dangling, 0.0, 1.0 / np.where(dangling, 1.0, out_w))
    transition = (A @ sparse.diags(inv)).tocsr()                     # column-stochastic P = A D^-1

    return KnowledgeBase(
        passages=passages, facts=facts, schema_freq=dict(schema_freq), active_schemas=active_schemas,
        canonical=canonical, concepts=concepts, prereq=prereq, graph=g, nodes=nodes, node_index=index,
        transition=transition, dangling=dangling,
        concept_emb=embedder.encode(concepts) if concepts else np.zeros((0, embedder.dim), np.float32),
        fact_emb=embedder.encode([f"{f['head']} {f['relation'].replace('_', ' ')} {f['tail']}" for f in facts])
        if facts else fact_emb, passage_emb=embedder.encode([p["text"] for p in passages]),
        conflict_log=conflict_log, build_calls=log.by_site(), foundationality=foundationality(prereq),
        passage_rep=passage_rep, type_rep=type_rep,
    )
