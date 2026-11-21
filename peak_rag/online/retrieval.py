"""Contribution 1: prerequisite-aware adaptive retrieval (Section 3.3; Algorithm 2).

* Eq. (1)  w_pre(c') = sum_{(c' -> c) in P} omega(c', c) * rel(c, q),
           rel(c, q) = max{0, min[1, cos(e(c), e(q))]} for concepts identified in the reasoning
           structure of q, and 0 otherwise.
* f(c')    foundationality = prerequisite out-degree of c'; used only to break ties in Kahn's sort.
* Eq. (2)  t~ = (1 - lambda) t + lambda normalize(w_pre); falls back to t when w_pre is all zero.
* PPR      r <- (1 - alpha) (P r + dangling * t~) + alpha t~,  P = A D^-1 column-stochastic,
           alpha = 0.15, at most 100 iterations or until ||delta r||_1 < 1e-8.
* Evidence top passages by r and top fact edges by endpoint mass, MMR de-duplication at >= 0.97,
           budget 12 items per sub-question; C = union over pi(q).
"""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field

import numpy as np

from .. import prompts
from ..backends.base import CountingLLM, clipped_cosine
from ..offline.build import KnowledgeBase, _parse_json
from ..text import content_tokens, normalize_answer


@dataclass
class Evidence:
    eid: str           # "p:<pid>" or "f:<fid>"
    kind: str          # passage | fact
    text: str
    score: float
    entities: tuple[str, ...] = ()
    pid: str = ""


@dataclass
class RetrievalResult:
    sub_questions: list[str]
    order: list[int]
    dag_edges: list[tuple[int, int]]
    concepts: list[list[str]]
    w_pre: dict[str, float]
    evidence: list[Evidence]
    per_sub: list[list[Evidence]] = field(default_factory=list)


# ---------------------------------------------------------------------------- reasoning structure
def decompose(llm: CountingLLM, cfg, question: str) -> list[str]:
    subs = _parse_json(llm(prompts.decompose_query(cfg, question)), [question])
    subs = [s for s in subs if isinstance(s, str) and s.strip()] if isinstance(subs, list) else []
    return (subs or [question])[: cfg.reasoning.max_chain_steps]


def reasoning_dag(llm: CountingLLM, cfg, subs: list[str]) -> list[tuple[int, int]]:
    """BuildReasoningDAG with an acyclicity guard: edges that would close a cycle are dropped."""
    raw = _parse_json(llm(prompts.build_reasoning_dag(cfg, subs)), [])
    edges: list[tuple[int, int]] = []
    adj: dict[int, set[int]] = {i: set() for i in range(len(subs))}

    def reaches(a: int, b: int) -> bool:
        stack, seen = [a], set()
        while stack:
            x = stack.pop()
            if x == b:
                return True
            if x not in seen:
                seen.add(x)
                stack.extend(adj[x])
        return False

    for e in raw if isinstance(raw, list) else []:
        try:
            i, j = int(e[0]), int(e[1])
        except (TypeError, ValueError, IndexError):
            continue
        if i == j or not (0 <= i < len(subs) and 0 <= j < len(subs)) or reaches(j, i):
            continue
        adj[i].add(j)
        edges.append((i, j))
    return edges


def kahn_sort(n: int, edges: list[tuple[int, int]], tie_score: list[float]) -> list[int]:
    """Kahn's algorithm; among ready sub-questions the largest foundationality sum goes first
    (then the original index)."""
    indeg = [0] * n
    succ: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in edges:
        indeg[j] += 1
        succ[i].append(j)
    heap = [(-tie_score[i], i) for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(heap, (-tie_score[j], j))
    return order


def identify_concepts(kb: KnowledgeBase, text: str) -> list[str]:
    """Concepts of the ontology layer whose surface name occurs in the text."""
    low = " " + normalize_answer(text) + " "
    surfaces: dict[str, list[str]] = {}
    for surf, can in kb.canonical.items():
        surfaces.setdefault(can, []).append(normalize_answer(surf))
    found = []
    for c in kb.concepts:
        if any(s and f" {s} " in low for s in surfaces.get(c, [c])):
            found.append(c)
    if not found:   # fall back to content-token containment for inflected mentions
        qt = set(content_tokens(text))
        for c in kb.concepts:
            ct = set(content_tokens(c))
            if ct and ct <= qt:
                found.append(c)
    return found


# ---------------------------------------------------------------------------- Eq. 1 and Eq. 2
def prerequisite_weights(kb: KnowledgeBase, question_concepts: list[str], rel: dict[str, float],
                         direction: str = "forward") -> dict[str, float]:
    """Eq. (1). ``direction`` implements the propagation variants of Supplementary Table S2:
    forward (prerequisites of question concepts), reverse (dependents), undirected (both), none."""
    w: dict[str, float] = {}
    if direction == "none":
        return w
    qset = set(question_concepts)
    for (cp, c), e in kb.prereq.items():
        if direction in ("forward", "undirected") and c in qset:
            w[cp] = w.get(cp, 0.0) + e["omega"] * rel.get(c, 0.0)
        if direction in ("reverse", "undirected") and cp in qset:
            w[c] = w.get(c, 0.0) + e["omega"] * rel.get(cp, 0.0)
    return w


def mix_teleport(t: np.ndarray, w_vec: np.ndarray, lam: float) -> np.ndarray:
    """Eq. (2); if every prerequisite weight is zero the normalised base teleport is used."""
    t = t / t.sum()
    s = w_vec.sum()
    if lam <= 0 or s <= 0:
        return t
    return (1.0 - lam) * t + lam * (w_vec / s)


def personalized_pagerank(kb: KnowledgeBase, teleport: np.ndarray, alpha: float, max_iter: int, tol: float,
                          engine: str = "power") -> np.ndarray:
    """Dangling-aware power iteration r <- (1 - alpha)(P r + dangling * t~) + alpha t~; the dangling mass is
    redistributed through the teleport. ``engine="networkx"`` runs nx.pagerank on the weighted
    MultiDiGraph with damping 1 - alpha, personalization and dangling set to t~ and NetworkX's tolerance
    rescaled so that the stopping rule is ||delta r||_1 < tol; ``engine="power"`` runs the same update in
    SciPy. If NetworkX has not converged after ``max_iter`` iterations, the iterate after ``max_iter``
    steps of the same update is returned, as Algorithm 2 prescribes."""
    if engine == "networkx":
        import networkx as nx
        pers = {n: float(teleport[i]) for i, n in enumerate(kb.nodes)}
        try:
            pr = nx.pagerank(kb.graph, alpha=1.0 - alpha, personalization=pers, dangling=pers, nstart=pers,
                             max_iter=max_iter, tol=tol / max(1, len(kb.nodes)), weight="weight")
            return np.array([pr[n] for n in kb.nodes])
        except nx.PowerIterationFailedConvergence:
            pass
    r = teleport.copy()
    P = kb.transition
    dmask = kb.dangling
    for _ in range(max_iter):
        new = (1.0 - alpha) * (P @ r + r[dmask].sum() * teleport) + alpha * teleport
        delta = np.abs(new - r).sum()
        r = new
        if delta < tol:
            break
    return r


# ---------------------------------------------------------------------------- evidence assembly
def base_teleport(kb: KnowledgeBase, sub_q_emb: np.ndarray, cfg) -> np.ndarray:
    """Mass on endpoints of the top-K retrieved facts (+0.5 on each fact's evidence passage) and on the
    top-K passages of the sub-question; uniform fallback."""
    rc = cfg.retrieval
    t = np.zeros(len(kb.nodes))
    active_ids = [f["fid"] for f in kb.facts if f["active"]]
    if active_ids:
        sims = clipped_cosine(sub_q_emb[None, :], kb.fact_emb[active_ids])[0]
        for k in np.argsort(-sims)[: rc.seed_top_k]:
            f = kb.facts[active_ids[k]]
            for e in (f["head"], f["tail"]):
                t[kb.node_index[f"e:{e}"]] += sims[k]
            t[kb.node_index[f"p:{f['pid']}"]] += rc.passage_seed_bonus * sims[k]
    if len(kb.passages):
        psims = clipped_cosine(sub_q_emb[None, :], kb.passage_emb)[0]
        for k in np.argsort(-psims)[: rc.seed_top_k]:
            t[kb.node_index[f"p:{kb.passages[k]['pid']}"]] += psims[k]
    if t.sum() <= 0:
        t[:] = 1.0
    return t


def context_pruning(items: list[Evidence], emb: np.ndarray, dedup: float, budget: int) -> list[Evidence]:
    """MMR-style pruning: greedy by relevance, skipping items with similarity >= dedup to a kept item."""
    order = np.argsort([-it.score for it in items])
    kept: list[int] = []
    for i in order:
        if any(float(emb[i] @ emb[j]) >= dedup for j in kept):
            continue
        kept.append(int(i))
        if len(kept) >= budget:
            break
    return [items[i] for i in kept]


def assemble_evidence(kb: KnowledgeBase, r: np.ndarray, cfg) -> tuple[list[Evidence], np.ndarray]:
    rc = cfg.retrieval
    pool = 2 * rc.context_budget
    pidx = [kb.node_index[f"p:{p['pid']}"] for p in kb.passages]
    items: list[Evidence] = []
    embs = []
    reps = [k for k, p in enumerate(kb.passages) if kb.passage_rep.get(p["pid"], p["pid"]) == p["pid"]]
    for k in sorted(reps, key=lambda k: -r[pidx[k]])[:pool]:
        p = kb.passages[k]
        items.append(Evidence(f"p:{p['pid']}", "passage", p["text"], float(r[pidx[k]]), pid=p["pid"]))
        embs.append(kb.passage_emb[k])
    facts = kb.active_facts
    if facts:
        mass = np.array([(r[kb.node_index[f"e:{f['head']}"]] + r[kb.node_index[f"e:{f['tail']}"]]) / 2 for f in facts])
        for k in np.argsort(-mass)[:pool]:
            f = facts[k]
            items.append(Evidence(f"f:{f['fid']}", "fact", kb.fact_text(f), float(mass[k]),
                                  (f["head"], f["tail"]), f["pid"]))
            embs.append(kb.fact_emb[f["fid"]])
    if not items:
        return [], np.zeros((0, kb.passage_emb.shape[1]))
    E = np.vstack(embs)
    return context_pruning(items, E, rc.mmr_dedup, rc.context_budget), E


def retrieve(kb: KnowledgeBase, question: str, llm: CountingLLM, embedder, cfg, *, route: str) -> RetrievalResult:
    """Algorithm 2. The reasoning DAG is built only for chained questions; the independent sub-questions
    of a parallel question have no dependency edges and are ordered by foundationality alone."""
    ab, rc = cfg.ablation, cfg.retrieval
    subs = decompose(llm, cfg, question)
    edges = reasoning_dag(llm, cfg, subs) if route == "chained" else []
    concepts = [identify_concepts(kb, s) for s in subs]
    q_concepts = sorted({c for cs in concepts for c in cs} | set(identify_concepts(kb, question)))
    q_emb = embedder.encode([question])[0]
    rel = {}
    if q_concepts:
        pos = {c: i for i, c in enumerate(kb.concepts)}
        cidx = [pos[c] for c in q_concepts]
        sims = clipped_cosine(q_emb[None, :], kb.concept_emb[cidx])[0]
        rel = dict(zip(q_concepts, map(float, sims)))
    direction = ab.propagation if ab.prereq_teleport else "none"
    w_pre = prerequisite_weights(kb, q_concepts, rel, direction)
    tie = [float(sum(kb.foundationality.get(c, 0) for c in cs)) if ab.prereq_tiebreak else 0.0 for cs in concepts]
    order = kahn_sort(len(subs), edges, tie)

    w_vec = np.zeros(len(kb.nodes))
    for c, w in w_pre.items():
        w_vec[kb.node_index[f"e:{c}"]] = w
    lam = rc.lambda_mix if ab.prereq_teleport else 0.0
    sub_embs = embedder.encode(subs)
    evidence: dict[str, Evidence] = {}
    per_sub = []
    for i in order:
        t = base_teleport(kb, sub_embs[i], cfg)
        tt = mix_teleport(t, w_vec, lam)
        r = personalized_pagerank(kb, tt, rc.alpha_teleport, rc.ppr_max_iter, rc.ppr_tol, rc.ppr_engine)
        ci, _ = assemble_evidence(kb, r, cfg)
        per_sub.append(ci)
        for ev in ci:
            if ev.eid not in evidence or evidence[ev.eid].score < ev.score:
                evidence[ev.eid] = ev
    return RetrievalResult(subs, order, edges, concepts, w_pre, list(evidence.values()), per_sub)


def evidence_to_json(ev: list[Evidence]) -> str:
    return json.dumps([e.__dict__ for e in ev])
